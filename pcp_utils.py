curl -X POST http://80.5.17.112:10086/start_profile
sleep 3
curl -X POST http://80.5.17.112:10086/stop_profile

from torch_npu.profiler.profiler import analyse
analyse("/mnt/share/w00504341/profiling")
