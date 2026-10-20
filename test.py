P:
# 清除代理环境变量
unset ftp_proxy FTP_PROXY
unset https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY
source /root/.bashrc
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
unset ftp_proxy
unset https_proxy
unset http_proxy
unset HCCL_INTRA_ROCE_ENABLE

# A5 ub
export PYTHONHASHSEED=0
# export ACL_OP_INIT_MODE=1
export MMC_LOCAL_CONFIG_PATH=/mnt/share/s00899273/mmc/ssd_109_data0/mmc-local.conf
# export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

# 自动获取配置
nic_name=自动获取
local_ip=自动获取

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_ALGO=level0:fullmesh

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_BUFFSIZE=1200

export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TASK_QUEUE_ENABLE=1

export VLLM_USE_FASTOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export VLLM_PP_LAYER_PARTITION="42,36"

vllm serve /mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916 \
  --host 0.0.0.0 \
  --port 8087 \
  --distributed-executor-backend mp \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --master-addr $local_ip \
  --master-port 7060 \
  --nnodes 2 \
  --node-rank 0 \
  --max_model_len 1048576 \
  --max-num-batched-tokens 4096 \
  --served-model-name glm \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-prefix-caching \
  --async-scheduling \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager":true}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/d00843668/profiling_no_pooling", "torch_profiler_with_stack": false}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "kv_connector_extra_config": {
            "connectors":[
            {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 1,
			"tp_size": 8,
                        "pp_size": 2,
                        "pp_layer_partition": "42,36"
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 2
                }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "memcache",
                    "use_layerwise": true
                }
            }
            ]
        }
    }' \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --additional_config '{"enable_cpu_binding": "true", "multistream_overlap_shared_expert": "true", "recompute_scheduler_enable": "true", "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": 1}'



D:
source /root/.bashrc
export PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics && mkdir -p $PROMETHEUS_MULTIPROC_DIR
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export VLLM_ASCEND_DP_ALLREDUCE_ON_NPU=1

unset ftp_proxy
unset https_proxy
unset http_proxy
unset HCCL_INTRA_ROCE_ENABLE

# A5 ub
export PYTHONHASHSEED=0
# export ACL_OP_INIT_MODE=1
export MMC_LOCAL_CONFIG_PATH=/mnt/share/s00899273/mmc/ssd_109_data0/mmc-local.conf
# export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/lib:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH}
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

# 自动获取配置
nic_name=自动获取
local_ip=自动获取

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_ALGO=level0:fullmesh

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_BUFFSIZE=900

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TASK_QUEUE_ENABLE=1

export ASCEND_RT_VISIBLE_DEVICES=$1
export VLLM_USE_FASTOKENS=1

export VLLM_USE_V2_MODEL_RUNNER=0
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

vllm serve /mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916 \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --max_model_len 1048576 \
  --max-num-batched-tokens 60 \
  --served-model-name glm \
  --gpu-memory-utilization 0.95 \
  --enable-expert-parallel \
  --max-num-seqs 10 \
  --trust-remote-code \
  --quantization ascend \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --async-scheduling \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 5,"method": "deepseek_mtp", "enforce_eager":true}' \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/d00843668/profiling_no_pooling", "torch_profiler_with_stack": false}' \
    --kv-transfer-config \
    '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_port": "30300",
        "kv_connector_extra_config": {
            "connectors":[
            {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "30300",
                "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 1,
			"tp_size": 8,
                        "pp_size": 2,
                        "pp_layer_partition": "42,36"
                },
                "decode": {
                        "dp_size": 16,
                        "tp_size": 2
                }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "memcache",
                    "use_layerwise": true
                }
            }
            ]
        }
    }' \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --additional_config '{"enable_cpu_binding": "true","ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false, "enable_super_kernel": false}, "recompute_scheduler_enable": "true", "multistream_overlap_shared_expert": "true","finegrained_tp_config": {"lmhead_tensor_parallel_size":8}, "enable_mlapo": true}'
