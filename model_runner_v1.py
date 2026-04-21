nic_name="enp48s3u1u1" # ifconfig 查看，选和本机 ip 相同的网卡
local_ip=141.61.39.133

export HCCL_IF_IP=$local_ip         # 指定HCCL通信库使用的网卡 IP 地址
export GLOO_SOCKET_IFNAME=$nic_name # 指定使用 Gloo通信库时指定网络接口名称 
export TP_SOCKET_IFNAME=$nic_name   # 指定 TensorParallel使用的网络接口名称
export HCCL_SOCKET_IFNAME=$nic_name # 指定 HCCL 通信库使用的网络接口名称
export OMP_PROC_BIND=false          # 允许操作系统调度线程在多个核心之间迁移
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=10          # 在支持 OpenMP 的程序中，最多使用 100 个 CPU 线程进行并行计算
export HCCL_BUFFSIZE=768           # 每个通信操作的缓冲区大小为 1024 Bytes
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
# export VLLM_ASCEND_ENABLE_MLAPO=1
# export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

# export PYTHONPATH=$PYTHONPATH:/mnt/share/d00933242/vllm_17.0/vllm
# export PYTHONPATH=$PYTHONPATH:/mnt/share/d00933242/vllm_ascend_personal
export VLLM_VERSION=0.18.0

# export ASCEND_RT_VISIBLE_DEVICES=12,13,14,15

# pcp2tp8 64k, batch2, uti 0.9
# dp2tp8 128k, uti 0.95

vllm serve /mnt/weight/DeepSeek-V3.1-Terminus-w8a8-QuaRot-lfs \
    --host 0.0.0.0 \
    --port 8004 \
    --served-model-name "model" \
    --api-server-count 1 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --data-parallel-address 141.61.39.133 \
    --data-parallel-rpc-port 5964  \
    --tensor-parallel-size 8 \
    --pipeline_parallel_size 1 \
    --prefill-context_parallel-size 1 \
    --decode-context_parallel-size 1 \
    --enable-expert-parallel \
    --max-num-seqs 1 \
    --max-model-len 132000 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --quantization ascend \
    --enforce-eager \
    --profiler-config '{
    "profiler": "torch",
        "torch_profiler_dir": "/home/z00911889/profile/test",
        "torch_profiler_with_stack": false,
        "torch_profiler_with_memory":false,
        "torch_profiler_record_shapes":true
    }' \
    2>&1 | tee logs/t.log
