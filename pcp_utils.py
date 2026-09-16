export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve /mnt/weight/GLM-5.2-W4A8C8-0713-MTP \
  --host 0.0.0.0 \
  --port 8077 \
  --api-server-count 1 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name glm-52 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --max-num-seqs 16 \
  --max-model-len 13500 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_dsa_cp": false,"enable_sparse_sfa_c8":false,"enable_sparse_li_c8":true,"enable_balance_scheduling":true,"multistream_overlap_shared_expert":true}' \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}'
