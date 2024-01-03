source /opt/intel/oneapi/setvars.sh --force
# source /home/wangruonan/intel/oneapi/setvars.sh --force
export CONDA_PREFIX=/home/wangruonan/miniconda3/envs/yina-llm
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1

export PYTHONPATH=/home/wangruonan/yina/BigDL/python/llm/src

# python3 generate.py --checkpoint_path /home/wangruonan/yina/yina/gpt-fast/scripts/checkpoints/meta-llama/llama2-13b-chat --max_new_tokens 100 --num_samples 10 --temperature 1 --draft_checkpoint_path /dev/shm/hub/Llama-2-7b-chat-hf/model.pth

python3 generate.py \
    --checkpoint_path /dev/shm/Llama-2-7b-chat-hf \
    --max_new_tokens 128 \
    --num_samples 10 \
    --temperature 1 \
    --draft_checkpoint_path /dev/shm/Llama-2-7b-chat-hf
    # /dev/shm/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/Llama-2-13b-chat-hf \