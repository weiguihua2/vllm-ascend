export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_A3_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve /mnt/data/Qwen3.5-35B-A3B   --max-num-seqs 8 --max_model_len 12800   --max-num-batched-tokens 6536   --gpu-memory-utilization 0.9   --tensor-parallel-size 4   --served-model-name qwen3.5   --port 8000    --trust-remote-code   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --decode-context-parallel-size 1   --prefill-context-parallel-size 2   --enable-expert-parallel   --no-enable-prefix-caching --no-async-scheduling --enforce-eager

