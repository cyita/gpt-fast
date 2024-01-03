# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import os
import copy
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import intel_extension_for_pytorch as ipex

# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer
from tp import maybe_init_dist

from transformers import LlamaTokenizer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList, top_k_top_p_filtering
import warnings
import logging
import inspect

logger = logging.getLogger("bigdl.llm.speculative")


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

# def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
#     probs = logits_to_probs(logits[0, -1], temperature, top_k)
#     idx_next = multinomial_sample_one_no_sync(probs)
#     return idx_next, probs

def sample(logits, return_probs: bool=False, do_sample: bool=False, top_k: int=50, top_p: float=0.7, temperature: float=0.7):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)
            
        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
            _logits = top_k_top_p_filtering(logits.view(-1, logits.size(-1)) / temperature, top_k=top_k, top_p=top_p)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(logits.shape[:-1])
        else:
            output_ids = torch.argmax(logits, dim=-1)
            
        return output_ids

def prefill(model, input_ids, past_key_values, generation_config):
    outputs = model(input_ids=input_ids,
                       past_key_values=past_key_values,
                       return_dict=True,
                       use_cache=True)
    next_token_logits = outputs.logits[:, -1, :]
    output_ids = sample(next_token_logits, do_sample=True, top_k=generation_config.top_k,
                        top_p=generation_config.top_p, temperature=generation_config.temperature)
    return output_ids, outputs['past_key_values']
    

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    # logits = model(x, input_pos)
    logits = model(x).logits
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(1, -1)
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x)
    # return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # xpu will raise error
    target_probs_cpu = target_probs.to('cpu')
    draft_probs_cpu = draft_probs.to('cpu')
    draft_tokens_cpu = draft_tokens.to('cpu')
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs_cpu[torch.arange(0, speculate_k, device='cpu'), draft_tokens_cpu]
    q = target_probs_cpu[torch.arange(0, speculate_k, device='cpu'), draft_tokens_cpu]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: Transformer,
    inputs: torch.Tensor,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    generation_config: Optional[GenerationConfig] = None,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    accept_counts = [0] * (speculate_k + 1)

    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
        # two conditions must be met
        # 1) the generation config must have been created from the model config (`_from_model_config` field);
        # 2) the generation config must have seen no modification since its creation (the hash is the same).
        if model.generation_config._from_model_config and model.generation_config._original_object_hash == hash(
            model.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(model.config)
            if new_generation_config != model.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                model.generation_config = new_generation_config
        generation_config = model.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**sampling_kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    model._validate_model_kwargs(model_kwargs.copy())

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

    # 2. Set generation parameters if not already defined
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not model.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )
    else:
        logger.error("encoder-decoder models are not supported now.")
        raise TypeError("encoder-decoder models are not supported now.")

    # yina: remove encoder_decoder part in the following code
    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
    
    # if streamer is not None:
    #     streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = sampling_kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    # generation_config.max_length = generation_config.max_length + speculate_k + 1 if is_speculative else generation_config.max_length
    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # Here we use sample generation mode
    # 8. prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
    )

    # 9. prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    # 11. prepare logits warper
    logits_warper = model._get_logits_warper(generation_config)

    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )

    generate_stats = {
        'accept_counts': accept_counts
    }

    # 13. run sample
    if is_speculative:
        n = input_ids_length
        past_key_values = None
        draft_past_key_values = None
        generate_ids = torch.empty([input_ids.size(0), generation_config.max_length], dtype=torch.long, device=model.device)
        draft_generate_ids = torch.empty([input_ids.size(0), speculate_k + 1], dtype=torch.long, device=model.device)
        # Prefill with main model
        output_ids, past_key_values = prefill(model,
                                              input_ids=input_ids,
                                              past_key_values=past_key_values,
                                              generation_config=generation_config)
        generate_ids[:, 0] = output_ids
        current_input_ids = output_ids.unsqueeze(0)
        
        # Prefill draft model to init kv cache
        _, draft_past_key_values = prefill(draft_model,
                                           input_ids=input_ids,
                                           past_key_values=draft_past_key_values,
                                           generation_config=generation_config)
        n += 1
        input_ids = torch.cat([input_ids, current_input_ids], dim=-1)
        draft_token_total_time = 0
        drafted_total_tokens = 0

        while n < generation_config.max_length:
            # Step 1: auto-regressive decode K tokens from draft model and get final p
            draft_current_input_ids = current_input_ids
            draft_generate_ids[:, 0] = current_input_ids
            draft_prob_list = []
            st = time.perf_counter()
            for step_draft in range(speculate_k):
                draft_output = draft_model(
                    input_ids=draft_current_input_ids,
                    past_key_values=draft_past_key_values,
                    return_dict=True,
                    use_cache=True,
                )
                draft_probs = draft_output['logits'].softmax(-1)
                draft_prob_list.append(draft_probs)
                draft_output_ids = sample(draft_output['logits'],
                                          return_probs=False, do_sample=True,
                                          top_k=generation_config.top_k,
                                          top_p=generation_config.top_p,
                                          temperature=generation_config.temperature)
                draft_generate_ids[:, step_draft + 1] = draft_output_ids
                draft_current_input_ids = draft_output_ids
                draft_past_key_values = draft_output['past_key_values']

                if n + step_draft + 2 >= generation_config.max_new_tokens + input_ids_length:
                    break
        
            drafted_n_tokens = step_draft + 1

            torch.xpu.synchronize()
            draft_token_total_time += (time.perf_counter() - st)
            drafted_total_tokens += drafted_n_tokens
            drafted_input_ids = draft_generate_ids[:, : drafted_n_tokens + 1] # raft input + raft completion
            
            output = model(input_ids=drafted_input_ids,
                           past_key_values=past_key_values,
                           return_dict=True,
                           use_cache=True)
            logits = output['logits']
            target_probs_cpu = logits.softmax(-1).squeeze()
            past_key_values = output['past_key_values']

            max_of_max_matched = logits.size(-2)

            draft_tokens_cpu = drafted_input_ids[:, 1:].squeeze(0)
            draft_probs_cpu = torch.stack(draft_prob_list).squeeze((1,2))

            # q: target prob, p: draft prob
            # q >= p: always accept draft token
            # q < p: q/p prob to accept draft token
            p = draft_probs_cpu[torch.arange(0, drafted_n_tokens, device='cpu'), draft_tokens_cpu]
            q = target_probs_cpu[torch.arange(0, drafted_n_tokens, device='cpu'), draft_tokens_cpu]
            accept_draft_prob = torch.minimum(torch.ones(()), q[:drafted_n_tokens]/ p)
            rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

            if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
                accept_length = drafted_n_tokens + 1
                last_token = multinomial_sample_one_no_sync(target_probs_cpu[-1])
                # fill last token into draft model
                draft_output = draft_model(input_ids=draft_current_input_ids,
                                           past_key_values=draft_past_key_values,
                                           return_dict=True,
                                           use_cache=True)
                draft_past_key_values = draft_output['past_key_values']
                next_tokens = torch.cat([draft_tokens_cpu, last_token])
            else:
                accept_length = rejected_locations[0].item()
                p = draft_probs_cpu[accept_length]
                q = target_probs_cpu[accept_length]
                new = q - p
                new = torch.where(new > 0, new, 0.0)
                new = new / new.sum()
                next_token = multinomial_sample_one_no_sync(new)
                next_tokens = torch.cat([draft_tokens_cpu[:accept_length], next_token])
                # Reset past_key_values
                # Main model
                past_key_values = [
                    (k[:, :, :-(max_of_max_matched - accept_length - 1)], v[:, :, :-(max_of_max_matched - accept_length - 1)]) for k, v in past_key_values
                ]
                # Draft model
                draft_past_key_values = [
                    (k[:, :, :-(max_of_max_matched - accept_length - 2)], v[:, :, :-(max_of_max_matched - accept_length - 2)]) for k, v in draft_past_key_values
                ]

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(generation_config.max_length - n, len(next_tokens))
            n += num_added
            input_ids = torch.cat([input_ids, next_tokens[:num_added].unsqueeze(0)], dim=-1)
            # print(f"n: {n}, num_added: {num_added}, input_ids: {input_ids.size()}, next_tokens: {next_tokens.size()}")
            current_input_ids = input_ids[:, -1:]

        generate_stats = {
            'accept_counts': accept_counts,
            'draft_token_latency': draft_token_total_time / drafted_total_tokens
        }
        print(generate_stats)

        return input_ids, generate_stats
    else:
        return model.sample(
            input_ids,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            synced_gpus=False,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            **model_kwargs,), generate_stats

def encode_tokens(tokenizer, string, bos=True, device='xpu'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, is_draft, use_tp):
    print(f"loading model from {checkpoint_path}")
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    # if "int8" in str(checkpoint_path):
    #     print("Using int8 weight-only quantization!")
    #     from quantize import WeightOnlyInt8QuantHandler
    #     simple_quantizer = WeightOnlyInt8QuantHandler(model)
    #     model = simple_quantizer.convert_for_runtime()

    # if "int4" in str(checkpoint_path):
    #     print("Using int4 quantization!")
    #     path_comps = checkpoint_path.name.split(".")
    #     assert path_comps[-2].startswith("g")
    #     groupsize = int(path_comps[-2][1:])
    #     from quantize import WeightOnlyInt4QuantHandler
    #     simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
    #     model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if is_draft:
        # convert the draft model to bigdl-llm q4_0
        from bigdl.llm import optimize_model
        from bigdl.llm.transformers.embedding import LLMEmbedding
        model = optimize_model(model, optimize_llm=True)
        # module = model.tok_embeddings
        # model.tok_embeddings = LLMEmbedding(
        #     num_embeddings=module.num_embeddings,
        #     embedding_dim=module.embedding_dim,
        #     padding_idx=module.padding_idx,
        #     max_norm=module.max_norm,
        #     norm_type=module.norm_type,
        #     scale_grad_by_freq=module.scale_grad_by_freq,
        #     sparse=module.sparse,
        #     _weight=module.weight.data,
        # )
        model = model.to(device)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    if not is_draft:
        # use ipex float16
        dtype = torch.float16
        print(f"IPEX optimize device: {device}, dtype: {dtype}")
        model = model.eval()
        model = model.to(device)
        model = ipex.optimize(model, dtype=dtype)
        return model
    else:
        return model.eval()


def _load_transformers_model(checkpoint_path, device, is_draft):
    from bigdl.llm.transformers import AutoModelForCausalLM
    if is_draft:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                     load_in_low_bit="sym_int4",
                                     optimize_model=True,
                                     trust_remote_code=True,
                                     use_cache=True).eval()
        model = model.to(device)
    else:
        dtype = torch.float16
        print(f"IPEX optimize device: {device}, dtype: {dtype}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, use_cache=True).eval()
        model = model.half().to(device)
        model = ipex.optimize(model)

    return model

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    # assert checkpoint_path.is_file(), checkpoint_path
    assert checkpoint_path.is_dir(), checkpoint_path

    # tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    tokenizer_path = checkpoint_path / "tokenizer.model"
    # assert tokenizer_path.is_file(), tokenizer_path

    # checkpoint_path = checkpoint_path / "model.pth"

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    # device = 'cpu'
    device = "xpu"
    precision = torch.float16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_transformers_model(checkpoint_path, device, False)

    if is_speculative:
        print("Loading draft model ...")
        draft_model = _load_transformers_model(draft_checkpoint_path, device, True)
    else:
        draft_model = None

    torch.xpu.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    prompt = open(f"/home/wangruonan/yina/BigDL/python/llm/dev/benchmark/all-in-one/prompt/1024.txt", 'r').read()
    # tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    # encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    encoded = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = encoded.size(1)

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if 'xpu' in device else 0


    for i in range(start, num_samples):
        torch.xpu.synchronize()
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            prof = torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True, record_shapes=True)
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens=max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        torch.xpu.synchronize()
        t = time.perf_counter() - t0

        if not interactive:
            # print(tokenizer.decode(y.tolist()))
            print(tokenizer.decode(y[0], skip_special_tokens=True))
        else:
            print()
        tokens_generated = y[0].size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec, tokens_generated:{tokens_generated}")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        print(f"------------\n")
    print("==========")
    if is_speculative:
        print(aggregate_metrics['accept_counts'])
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    # print(f"Memory used: {torch.xpu.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path, args.speculate_k
    )
