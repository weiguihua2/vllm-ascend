export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_A3_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800

vllm serve /mnt/data/Qwen3.5-35B-A3B   --max-num-seqs 8 --max_model_len 12800   --max-num-batched-tokens 6536   --gpu-memory-utilization 0.9   --tensor-parallel-size 4   --served-model-name qwen3.5   --port 8000    --trust-remote-code   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --decode-context-parallel-size 1   --prefill-context-parallel-size 2   --enable-expert-parallel   --no-enable-prefix-caching --no-async-scheduling --enforce-eager




root@node-125:/home/w00504341/bench/benchmark# curl -X POST -s http://1.48.29.125:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "qwen3.5",
    "prompt": ["The capital of France is"],
    "max_tokens": 50,
    "temperature": 0
  }'
{"id":"cmpl-b4c01382d8f9cb7a","object":"text_completion","created":1778830500,"model":"qwen3.5","choices":[{"index":0,"text":" Paris.\n The capital of France is Paris.\n Which sentence is correct?\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:** The user is presenting two sentences and asking which one is correct.\n    *   Sentence","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":55,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# 
root@node-125:/home/w00504341/bench/benchmark# curl -X POST -s http://1.48.29.125:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "qwen3.5",
    "prompt": ["The capital of France is"],
    "max_tokens": 50,
    "temperature": 0
  }'
{"id":"cmpl-bd7c2a43ea7bdf8e","object":"text_completion","created":1778830537,"model":"qwen3.5","choices":[{"index":0,"text":" Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n Paris\n","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":55,"completion_tokens":50,"prompt_tokens_details":null},"kv_transfer_params":null}root@node-125:/home/w00504341/bench/benchmark# 
