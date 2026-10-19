# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Six-node service configuration extracted from task a0081bff logs.

Source: task_a0081bff-11be-4137-a61c-cb86e2109dec_all_logs_1016,
logs dated 2026-10-15/16. The logged vLLM engine version is 0.30.0;
0_24_0 is the destination Git branch, not the logged runtime version.

This is a configuration snapshot; importing it does not launch services.
Constructor reprs are represented as plain dictionaries and enums as names.
P0 has one API server. P1 is its headless PP worker (no separate API server).
D0-D3 each run four API servers, covering DP ranks 0-15 with TP=2.

Node  Host            Role     Parallel layout           HTTP ports
P0    141.61.133.109  Prefill  TP=8 PP=2, node rank 0     8087
P1    141.61.133.111  Prefill  TP=8 PP=2, node rank 1     headless
D0    141.61.133.114  Decode   TP=2, DP ranks 0-3         8000-8003
D1    141.61.133.113  Decode   TP=2, DP ranks 4-7         8000-8003
D2    141.61.133.115  Decode   TP=2, DP ranks 8-11        8000-8003
D3    141.61.133.116  Decode   TP=2, DP ranks 12-15       8000-8003

For a Decode process, merge DECODE_ARGUMENTS with its argument_overrides.
An omitted port in non-default args is intentionally retained as omitted;
the launcher starts at 8000 and the local process index gives the HTTP port.
Engine IDs are historical runtime-generated identifiers, not launch inputs.
P1's complete command is preserved because it has no non-default-args dump.
Environment values not printed in the logs are not reconstructed.
Requested and resolved engine compilation configurations are kept separately;
Decode requests FULL_DECODE_ONLY but the engine dump reports PIECEWISE.
"""

TASK_ID = "a0081bff-11be-4137-a61c-cb86e2109dec"

PREFILL_ARGUMENTS = {
    "model_tag": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
    "enable_auto_tool_choice": True,
    "tool_call_parser": "glm47",
    "host": "0.0.0.0",
    "port": 8087,
    "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
    "trust_remote_code": True,
    "max_model_len": 1048576,
    "quantization": "ascend",
    "enforce_eager": True,
    "served_model_name": ["glm"],
    "reasoning_parser": "glm45",
    "distributed_executor_backend": "mp",
    "pipeline_parallel_size": 2,
    "master_addr": "172.27.18.109",
    "master_port": 7060,
    "nnodes": 2,
    "tensor_parallel_size": 8,
    "enable_expert_parallel": True,
    "gpu_memory_utilization": 0.95,
    "kv_cache_dtype": "fp8",
    "enable_prefix_caching": True,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 8,
    "async_scheduling": True,
    "speculative_config": {"num_speculative_tokens": 1, "method": "deepseek_mtp", "enforce_eager": True},
    "kv_transfer_config": {
        "kv_connector": "MultiConnector",
        "engine_id": "92891f49-71a1-42ba-967f-654f48cd42cf",
        "kv_buffer_device": "npu",
        "kv_buffer_size": 1000000000.0,
        "kv_role": "kv_producer",
        "kv_rank": None,
        "kv_parallel_size": 1,
        "kv_ip": "127.0.0.1",
        "kv_port": 30000,
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": 8, "pp_size": 2, "pp_layer_partition": "42,36"},
                        "decode": {"dp_size": 16, "tp_size": 2},
                    },
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {"lookup_rpc_port": "0", "backend": "memcache", "use_layerwise": True},
                },
            ]
        },
        "kv_connector_module_path": None,
        "enable_permute_local_kv": False,
        "kv_load_failure_policy": "fail",
    },
    "attention_config": {
        "backend": None,
        "minimax_m3_msa_decode_backend": "triton",
        "backend_per_kind": {},
        "flash_attn_version": None,
        "flash_attn_max_num_splits_for_cuda_graph": 32,
        "tq_max_kv_splits_for_cuda_graph": 32,
        "use_trtllm_attention": None,
        "disable_flashinfer_q_quantization": False,
        "mla_prefill_backend": None,
        "use_prefill_query_quantization": False,
        "indexer_kv_dtype": "fp8",
        "hisparse_config": None,
        "use_non_causal": False,
        "sparse_mla_force_mqa": False,
        "flex_attn_block_m": None,
        "flex_attn_block_n": None,
        "flex_attn_q_block_size": None,
        "flex_attn_kv_block_size": None,
    },
    "additional_config": {
        "enable_cpu_binding": "true",
        "multistream_overlap_shared_expert": "true",
        "recompute_scheduler_enable": "true",
        "enable_dsa_cp": True,
        "enable_flashcomm1": True,
        "enable_fused_mc2": 1,
    },
    "profiler_config": {
        "profiler": "torch",
        "torch_profiler_dir": "/home/d00843668/profiling_no_pooling",
        "proton_profiler_dir": "",
        "proton_context": "shadow",
        "proton_data": "tree",
        "proton_backend": None,
        "proton_mode": None,
        "proton_hook": None,
        "proton_output_format": None,
        "proton_graph_attribution": False,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "capture_torch_profiler": False,
        "detailed_trace_annotation": False,
        "ignore_frontend": False,
        "delay_iterations": 0,
        "max_iterations": 0,
        "warmup_iterations": 0,
        "active_iterations": 5,
        "wait_iterations": 0,
    },
}

DECODE_ARGUMENTS = {
    "model_tag": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
    "enable_auto_tool_choice": True,
    "tool_call_parser": "glm47",
    "host": "0.0.0.0",
    "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
    "trust_remote_code": True,
    "max_model_len": 1048576,
    "quantization": "ascend",
    "served_model_name": ["glm"],
    "reasoning_parser": "glm45",
    "tensor_parallel_size": 2,
    "data_parallel_size": 16,
    "data_parallel_address": "172.27.18.114",
    "data_parallel_rpc_port": 12321,
    "enable_expert_parallel": True,
    "gpu_memory_utilization": 0.95,
    "kv_cache_dtype": "fp8",
    "enable_prefix_caching": True,
    "max_num_batched_tokens": 60,
    "max_num_seqs": 10,
    "async_scheduling": True,
    "speculative_config": {"num_speculative_tokens": 5, "method": "deepseek_mtp", "enforce_eager": True},
    "compilation_config": {
        "mode": None,
        "debug_dump_path": None,
        "cache_dir": "",
        "compile_cache_save_format": "binary",
        "backend": "vllm_ascend.compilation.compiler_interface.AscendCompiler",
        "custom_ops": [],
        "ir_enable_torch_wrap": None,
        "splitting_ops": None,
        "compile_mm_encoder": False,
        "cudagraph_mm_encoder": False,
        "encoder_cudagraph_token_budgets": [],
        "encoder_cudagraph_max_vision_items_per_batch": 0,
        "encoder_cudagraph_max_frames_per_batch": None,
        "compile_sizes": None,
        "compile_ranges_endpoints": None,
        "inductor_compile_config": {
            "enable_auto_functionalized_v2": False,
            "size_asserts": False,
            "alignment_asserts": False,
            "scalar_asserts": False,
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
        },
        "inductor_passes": {},
        "cudagraph_mode": "CUDAGraphMode.FULL_DECODE_ONLY",
        "cudagraph_num_of_warmups": 0,
        "cudagraph_capture_sizes": None,
        "cudagraph_copy_inputs": False,
        "cudagraph_specialize_lora": True,
        "use_inductor_graph_partition": None,
        "pass_config": {},
        "max_cudagraph_capture_size": None,
        "dynamic_shapes_config": {
            "type": "DynamicShapesType.BACKED",
            "evaluate_guards": False,
            "assume_32_bit_indexing": False,
        },
        "local_cache_dir": None,
        "fast_moe_cold_start": None,
        "static_all_moe_layers": [],
    },
    "attention_config": {
        "backend": None,
        "minimax_m3_msa_decode_backend": "triton",
        "backend_per_kind": {},
        "flash_attn_version": None,
        "flash_attn_max_num_splits_for_cuda_graph": 32,
        "tq_max_kv_splits_for_cuda_graph": 32,
        "use_trtllm_attention": None,
        "disable_flashinfer_q_quantization": False,
        "mla_prefill_backend": None,
        "use_prefill_query_quantization": False,
        "indexer_kv_dtype": "fp8",
        "hisparse_config": None,
        "use_non_causal": False,
        "sparse_mla_force_mqa": False,
        "flex_attn_block_m": None,
        "flex_attn_block_n": None,
        "flex_attn_q_block_size": None,
        "flex_attn_kv_block_size": None,
    },
    "additional_config": {
        "enable_cpu_binding": "true",
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": False,
            "enable_super_kernel": False,
        },
        "recompute_scheduler_enable": "true",
        "multistream_overlap_shared_expert": "true",
        "finegrained_tp_config": {"lmhead_tensor_parallel_size": 8},
        "enable_mlapo": True,
    },
    "profiler_config": {
        "profiler": "torch",
        "torch_profiler_dir": "/home/d00843668/profiling_no_pooling",
        "proton_profiler_dir": "",
        "proton_context": "shadow",
        "proton_data": "tree",
        "proton_backend": None,
        "proton_mode": None,
        "proton_hook": None,
        "proton_output_format": None,
        "proton_graph_attribution": False,
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "capture_torch_profiler": False,
        "detailed_trace_annotation": False,
        "ignore_frontend": False,
        "delay_iterations": 0,
        "max_iterations": 0,
        "warmup_iterations": 0,
        "active_iterations": 5,
        "wait_iterations": 0,
    },
}

RUNTIME_PROFILES = {
    "prefill": {
        "source_log": "P0_141.61.133.109-5379/output.log",
        "source_line": 444,
        "resolved_engine_config": {
            "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
            "speculative_config": {
                "method": "mtp",
                "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
                "num_spec_tokens": 1,
            },
            "tokenizer": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
            "skip_tokenizer_init": False,
            "tokenizer_mode": "auto",
            "revision": None,
            "tokenizer_revision": None,
            "trust_remote_code": True,
            "dtype": "torch.bfloat16",
            "max_seq_len": 1048576,
            "download_dir": None,
            "load_format": "auto",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "dcp_comm_backend": "a2a",
            "disable_custom_all_reduce": True,
            "quantization": "ascend",
            "quantization_config": None,
            "enforce_eager": True,
            "enable_return_routed_experts": False,
            "kv_cache_dtype": "fp8",
            "device_config": "npu",
            "structured_outputs_config": {
                "backend": "auto",
                "disable_any_whitespace": False,
                "disable_additional_properties": False,
                "reasoning_parser": "glm45",
                "reasoning_parser_plugin": "",
                "enable_in_reasoning": False,
            },
            "observability_config": {
                "show_hidden_metrics_for_version": None,
                "otlp_traces_endpoint": None,
                "collect_detailed_traces": None,
                "per_request_spec_decode_metrics": "none",
                "kv_cache_metrics": False,
                "kv_cache_metrics_sample": 0.01,
                "cudagraph_metrics": False,
                "enable_layerwise_nvtx_tracing": False,
                "enable_mfu_metrics": False,
                "enable_mm_processor_stats": False,
                "enable_logging_iteration_details": False,
                "jit_monitor_mode": "warn",
                "jit_monitor_verbose": False,
            },
            "seed": 0,
            "served_model_name": "glm",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "pooler_config": None,
            "compilation_config": {
                "mode": "CompilationMode.NONE",
                "debug_dump_path": None,
                "cache_dir": "",
                "compile_cache_save_format": "binary",
                "backend": "vllm_ascend.compilation.compiler_interface.AscendCompiler",
                "custom_ops": ["all"],
                "ir_enable_torch_wrap": False,
                "splitting_ops": [],
                "compile_mm_encoder": False,
                "cudagraph_mm_encoder": False,
                "encoder_cudagraph_token_budgets": [],
                "encoder_cudagraph_max_vision_items_per_batch": 0,
                "encoder_cudagraph_max_frames_per_batch": None,
                "compile_sizes": [],
                "compile_ranges_endpoints": [4096],
                "inductor_compile_config": {
                    "enable_auto_functionalized_v2": False,
                    "size_asserts": False,
                    "alignment_asserts": False,
                    "scalar_asserts": False,
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                },
                "inductor_passes": {},
                "cudagraph_mode": "CUDAGraphMode.NONE",
                "cudagraph_num_of_warmups": 1,
                "cudagraph_capture_sizes": [],
                "cudagraph_copy_inputs": False,
                "cudagraph_specialize_lora": True,
                "use_inductor_graph_partition": False,
                "pass_config": {
                    "fuse_norm_quant": True,
                    "fuse_act_quant": True,
                    "fuse_attn_quant": False,
                    "enable_sp": False,
                    "fuse_gemm_comms": False,
                    "fuse_allreduce_rms": False,
                    "enable_qk_norm_rope_fusion": False,
                    "fuse_rope_kvcache_cat_mla": False,
                    "fuse_act_padding": False,
                    "fuse_qk_norm_rope_kvcache": False,
                },
                "max_cudagraph_capture_size": 0,
                "dynamic_shapes_config": {
                    "type": "DynamicShapesType.BACKED",
                    "evaluate_guards": False,
                    "assume_32_bit_indexing": False,
                },
                "local_cache_dir": None,
                "fast_moe_cold_start": False,
                "static_all_moe_layers": [],
            },
            "kernel_config": {
                "ir_op_priority": {
                    "rms_norm": ["native"],
                    "fused_add_rms_norm": ["native"],
                    "gelu_and_mul_sparse": ["native"],
                },
                "enable_flashinfer_autotune": True,
                "enable_cutedsl_warmup": True,
                "enable_jit_warmup": True,
                "moe_backend": "auto",
                "sparse_indexer_topk_backend": "auto",
                "linear_backend": "auto",
                "linear_backend_per_quant": None,
            },
        },
    },
    "decode": {
        "source_log": "D0_141.61.133.114-5379/output.log",
        "source_line": 1744,
        "resolved_engine_config": {
            "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
            "speculative_config": {
                "method": "mtp",
                "model": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
                "num_spec_tokens": 5,
            },
            "tokenizer": "/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916",
            "skip_tokenizer_init": False,
            "tokenizer_mode": "auto",
            "revision": None,
            "tokenizer_revision": None,
            "trust_remote_code": True,
            "dtype": "torch.bfloat16",
            "max_seq_len": 1048576,
            "download_dir": None,
            "load_format": "auto",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 16,
            "decode_context_parallel_size": 1,
            "dcp_comm_backend": "a2a",
            "disable_custom_all_reduce": True,
            "quantization": "ascend",
            "quantization_config": None,
            "enforce_eager": False,
            "enable_return_routed_experts": False,
            "kv_cache_dtype": "fp8",
            "device_config": "npu",
            "structured_outputs_config": {
                "backend": "auto",
                "disable_any_whitespace": False,
                "disable_additional_properties": False,
                "reasoning_parser": "glm45",
                "reasoning_parser_plugin": "",
                "enable_in_reasoning": False,
            },
            "observability_config": {
                "show_hidden_metrics_for_version": None,
                "otlp_traces_endpoint": None,
                "collect_detailed_traces": None,
                "per_request_spec_decode_metrics": "none",
                "kv_cache_metrics": False,
                "kv_cache_metrics_sample": 0.01,
                "cudagraph_metrics": False,
                "enable_layerwise_nvtx_tracing": False,
                "enable_mfu_metrics": False,
                "enable_mm_processor_stats": False,
                "enable_logging_iteration_details": False,
                "jit_monitor_mode": "warn",
                "jit_monitor_verbose": False,
            },
            "seed": 0,
            "served_model_name": "glm",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "pooler_config": None,
            "compilation_config": {
                "mode": "CompilationMode.VLLM_COMPILE",
                "debug_dump_path": None,
                "cache_dir": "",
                "compile_cache_save_format": "binary",
                "backend": "vllm_ascend.compilation.compiler_interface.AscendCompiler",
                "custom_ops": ["all"],
                "ir_enable_torch_wrap": False,
                "splitting_ops": [
                    "vllm::unified_attention_with_output",
                    "vllm::unified_mla_attention_with_output",
                    "vllm::mamba_mixer2",
                    "vllm::mamba_mixer",
                    "vllm::short_conv",
                    "vllm::qwen4_exp_ple_short_conv",
                    "vllm::qwen4_exp_qsa_with_output",
                    "vllm::linear_attention",
                    "vllm::qwen_gdn_attention_core",
                    "vllm::qwen_gdn_attention_core_fused_norm_packed",
                    "vllm::gdn_attention_core_xpu",
                    "vllm::olmo_hybrid_gdn_full_forward",
                    "vllm::sparse_attn_indexer",
                    "vllm::rocm_aiter_sparse_attn_indexer",
                    "vllm::deepseek_v4_attention",
                    "vllm::hpc_rope_norm_forward",
                    "vllm::unified_kv_cache_update",
                    "vllm::unified_mla_kv_cache_update",
                    "vllm::mla_forward",
                    "vllm::dsa_forward",
                ],
                "compile_mm_encoder": False,
                "cudagraph_mm_encoder": False,
                "encoder_cudagraph_token_budgets": [],
                "encoder_cudagraph_max_vision_items_per_batch": 0,
                "encoder_cudagraph_max_frames_per_batch": None,
                "compile_sizes": [],
                "compile_ranges_endpoints": [60],
                "inductor_compile_config": {
                    "enable_auto_functionalized_v2": False,
                    "size_asserts": False,
                    "alignment_asserts": False,
                    "scalar_asserts": False,
                    "combo_kernels": True,
                    "benchmark_combo_kernel": True,
                },
                "inductor_passes": {},
                "cudagraph_mode": "CUDAGraphMode.PIECEWISE",
                "cudagraph_num_of_warmups": 1,
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 60],
                "cudagraph_copy_inputs": False,
                "cudagraph_specialize_lora": True,
                "use_inductor_graph_partition": False,
                "pass_config": {
                    "fuse_norm_quant": True,
                    "fuse_act_quant": True,
                    "fuse_attn_quant": False,
                    "enable_sp": False,
                    "fuse_gemm_comms": False,
                    "fuse_allreduce_rms": False,
                    "enable_qk_norm_rope_fusion": False,
                    "fuse_rope_kvcache_cat_mla": False,
                    "fuse_act_padding": False,
                    "fuse_qk_norm_rope_kvcache": False,
                },
                "max_cudagraph_capture_size": 60,
                "dynamic_shapes_config": {
                    "type": "DynamicShapesType.BACKED",
                    "evaluate_guards": False,
                    "assume_32_bit_indexing": False,
                },
                "local_cache_dir": None,
                "fast_moe_cold_start": False,
                "static_all_moe_layers": [],
            },
            "kernel_config": {
                "ir_op_priority": {
                    "rms_norm": ["native"],
                    "fused_add_rms_norm": ["native"],
                    "gelu_and_mul_sparse": ["native"],
                },
                "enable_flashinfer_autotune": True,
                "enable_cutedsl_warmup": True,
                "enable_jit_warmup": True,
                "moe_backend": "auto",
                "sparse_indexer_topk_backend": "auto",
                "linear_backend": "auto",
                "linear_backend_per_quant": None,
            },
        },
    },
}

NODE_CONFIGS = {
    "P0": {
        "source_log": "P0_141.61.133.109-5379/output.log",
        "source_sha256": "462d979ef172ec1f2f13da37d28e40a257d6bc574bf6e6e2ead663851f985d13",
        "host": "141.61.133.109",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/p0_0",
        "launch_command": "bash run_dp_template.sh",
        "evidence": {
            "prometheus": {
                "line": 7,
                "text": "[Prometheus] script=run_dp_template.sh pid=88509 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 25,
                "text": "INFO 10-15 23:52:02 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "node_rank": {
                "line": 215,
                "text": "(APIServer pid=91565) INFO 10-15 23:52:09 [arg_utils.py:2286] "
                "Inferred data_parallel_rank 0 from node_rank 0",
            },
            "parallel_layout": {
                "line": 668,
                "text": "(Worker pid=112518) INFO 10-15 23:52:41 "
                "[parallel_state.py:1827] world_size=16 rank=0 local_rank=0 "
                "distributed_init_method=tcp://172.27.18.109:7060 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": (
            "VLLM_ASCEND_ENABLE_FLASHCOMM1",
            "VLLM_ASCEND_ENABLE_PREFETCH_MLP",
            "VLLM_RPC_TIMEOUT",
        ),
        "role": "prefill",
        "arguments_reference": "PREFILL_ARGUMENTS",
        "arguments_source_line": 208,
    },
    "P1": {
        "source_log": "P1_141.61.133.111-5379/output.log",
        "source_sha256": "ec72c30d6563fbcd9f07246fc6a514fb6a757f7943be959cedd9876ec368f049",
        "host": "141.61.133.111",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/p0_1",
        "launch_command": "bash run_dp_template.sh",
        "evidence": {
            "prometheus": {
                "line": 9,
                "text": "[Prometheus] script=run_dp_template.sh pid=60095 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 27,
                "text": "INFO 10-15 23:52:08 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "node_rank": {
                "line": 210,
                "text": "INFO 10-15 23:52:32 [arg_utils.py:2286] Inferred data_parallel_rank 0 from node_rank 1",
            },
            "headless": {
                "line": 235,
                "text": "INFO 10-15 23:53:03 [serve.py:221] Launching vLLM (v0.30.0) "
                "headless multiproc executor, with head node address "
                "172.27.18.109:7060 for torch.distributed process group.",
            },
            "parallel_layout": {
                "line": 458,
                "text": "(Worker pid=86780) INFO 10-15 23:53:15 "
                "[parallel_state.py:1827] world_size=16 rank=8 local_rank=0 "
                "distributed_init_method=tcp://172.27.18.109:7060 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": (
            "VLLM_ASCEND_ENABLE_FLASHCOMM1",
            "VLLM_ASCEND_ENABLE_PREFETCH_MLP",
            "VLLM_RPC_TIMEOUT",
        ),
        "raw_serve_command_lines": (96343, 96377),
        "raw_serve_command": "vllm serve /mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916 --host 0.0.0.0 "
        "--port 8087 --headless --distributed-executor-backend mp "
        "--pipeline-parallel-size 2 --tensor-parallel-size 8 --enable-expert-parallel "
        "--master-addr 172.27.18.109 --master-port 7060 --nnodes 2 --node-rank 1 "
        "--max_model_len 1048576 --max-num-batched-tokens 4096 --served-model-name glm "
        "--gpu-memory-utilization 0.95 --max-num-seqs 8 --trust-remote-code "
        "--enforce-eager --quantization ascend --enable-auto-tool-choice "
        "--tool-call-parser glm47 --reasoning-parser glm45 --enable-prefix-caching "
        '--async-scheduling --speculative-config \'{"num_speculative_tokens": '
        '1,"method": "deepseek_mtp", "enforce_eager":true}\' --profiler-config '
        '\'{"profiler": "torch", "torch_profiler_dir": '
        '"/home/d00843668/profiling_no_pooling", "torch_profiler_with_stack": false}\' '
        "--kv-transfer-config '{\n"
        '  "kv_connector": "MultiConnector",\n'
        '  "kv_role": "kv_producer",\n'
        '  "kv_port": "30000",\n'
        '  "kv_connector_extra_config": {\n'
        '    "connectors": [\n'
        "      {\n"
        '        "kv_connector": "MooncakeConnectorV2",\n'
        '        "kv_role": "kv_producer",\n'
        '        "kv_port": "30000",\n'
        '        "kv_connector_extra_config": {\n'
        '          "prefill": {\n'
        '            "dp_size": 1,\n'
        '            "tp_size": 8,\n'
        '            "pp_size": 2,\n'
        '            "pp_layer_partition": "42,36"\n'
        "          },\n"
        '          "decode": {\n'
        '            "dp_size": 16,\n'
        '            "tp_size": 2\n'
        "          }\n"
        "        }\n"
        "      },\n"
        "      {\n"
        '        "kv_connector": "AscendStoreConnector",\n'
        '        "kv_role": "kv_producer",\n'
        '        "kv_connector_extra_config": {\n'
        '          "lookup_rpc_port": "0",\n'
        '          "backend": "memcache",\n'
        '          "use_layerwise": true\n'
        "        }\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}' --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8 "
        '--additional_config \'{"enable_cpu_binding": "true", '
        '"multistream_overlap_shared_expert": "true", "recompute_scheduler_enable": '
        '"true", "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_fused_mc2": '
        "1}'",
        "role": "prefill",
        "headless": True,
        "node_rank": 1,
        "shared_prefill_arguments_reference": "PREFILL_ARGUMENTS",
        "note": "PREFILL_ARGUMENTS describes the shared model; use the raw command for P1 headless/node-rank "
        "options. No API server on P1.",
    },
    "D0": {
        "source_log": "D0_141.61.133.114-5379/output.log",
        "source_sha256": "82a72b47269eff74cb4b850a7fb0786ee53c8abdd71aef788565624454d42c53",
        "host": "141.61.133.114",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/d0_0",
        "launch_command": "python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 4 "
        "--dp-rank-start 0 --dp-address 172.27.18.114 --dp-rpc-port 12321 "
        "--vllm-start-port 8000",
        "evidence": {
            "prometheus": {
                "line": 7,
                "text": "[Prometheus] script=./run_dp_template.sh pid=67380 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 52,
                "text": "INFO 10-15 23:51:50 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "parallel_layout": {
                "line": 3443,
                "text": "(Worker pid=88940) INFO 10-15 23:53:28 "
                "[parallel_state.py:1827] world_size=32 rank=2 local_rank=0 "
                "distributed_init_method=tcp://172.27.18.114:55793 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": ("VLLM_ASCEND_DP_ALLREDUCE_ON_NPU", "VLLM_RPC_TIMEOUT"),
        "role": "decode",
        "services": (
            {
                "data_parallel_rank": 0,
                "source_line": 803,
                "http_port_from_launcher": 8000,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "data_parallel_rank": 0,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "38e3807f-7038-4e51-bc52-35326f07686d",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30300,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30300",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 1,
                "source_line": 841,
                "http_port_from_launcher": 8001,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8001,
                    "data_parallel_rank": 1,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "ff573c97-74b8-41be-956d-90006c7db36d",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30301,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30301",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 2,
                "source_line": 780,
                "http_port_from_launcher": 8002,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8002,
                    "data_parallel_rank": 2,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "e1e8c117-0291-4325-9351-03d00a06fcae",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30302,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30302",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 3,
                "source_line": 709,
                "http_port_from_launcher": 8003,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8003,
                    "data_parallel_rank": 3,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "07916ea8-e74d-4330-9f0a-754b9497b909",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30303,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30303",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
        ),
        "engine_config_source_lines": (1744, 1747, 1750, 1753),
    },
    "D1": {
        "source_log": "D1_141.61.133.113-5379/output.log",
        "source_sha256": "ead0b1450c79537688570307351e18bb8b9c9612819835dd292ab85184de675a",
        "host": "141.61.133.113",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/d0_1",
        "launch_command": "python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 4 "
        "--dp-rank-start 4 --dp-address 172.27.18.114 --dp-rpc-port 12321 "
        "--vllm-start-port 8000",
        "evidence": {
            "prometheus": {
                "line": 7,
                "text": "[Prometheus] script=./run_dp_template.sh pid=71194 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 68,
                "text": "INFO 10-15 23:52:12 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "parallel_layout": {
                "line": 2631,
                "text": "(Worker pid=95717) INFO 10-15 23:53:42 "
                "[parallel_state.py:1827] world_size=32 rank=14 "
                "local_rank=0 "
                "distributed_init_method=tcp://172.27.18.114:55793 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": ("VLLM_ASCEND_DP_ALLREDUCE_ON_NPU", "VLLM_RPC_TIMEOUT"),
        "role": "decode",
        "services": (
            {
                "data_parallel_rank": 4,
                "source_line": 787,
                "http_port_from_launcher": 8000,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "data_parallel_rank": 4,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "72e98d5a-0e71-40ba-8f5e-fa24b4e015fa",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30300,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30300",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 5,
                "source_line": 740,
                "http_port_from_launcher": 8001,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8001,
                    "data_parallel_rank": 5,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "827d876c-0e35-4673-afc2-147f4aa28987",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30301,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30301",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 6,
                "source_line": 841,
                "http_port_from_launcher": 8002,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8002,
                    "data_parallel_rank": 6,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "ab27570e-3388-4794-9847-ac4e953befec",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30302,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30302",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 7,
                "source_line": 820,
                "http_port_from_launcher": 8003,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8003,
                    "data_parallel_rank": 7,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "b61fdf42-62d5-4383-834d-e0c451e98d96",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30303,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30303",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
        ),
        "engine_config_source_lines": (1733, 1734, 1739, 1740),
    },
    "D2": {
        "source_log": "D2_141.61.133.115-5379/output.log",
        "source_sha256": "90f4093ba002234c25f1bb02f823d9fb31c8e9b74342f1f7ae58dd579debea80",
        "host": "141.61.133.115",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/d0_2",
        "launch_command": "python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 4 "
        "--dp-rank-start 8 --dp-address 172.27.18.114 --dp-rpc-port 12321 "
        "--vllm-start-port 8000",
        "evidence": {
            "prometheus": {
                "line": 7,
                "text": "[Prometheus] script=./run_dp_template.sh pid=69212 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 71,
                "text": "INFO 10-15 23:51:52 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "parallel_layout": {
                "line": 2631,
                "text": "(Worker pid=93055) INFO 10-15 23:53:23 "
                "[parallel_state.py:1827] world_size=32 rank=16 "
                "local_rank=0 "
                "distributed_init_method=tcp://172.27.18.114:55793 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": ("VLLM_ASCEND_DP_ALLREDUCE_ON_NPU", "VLLM_RPC_TIMEOUT"),
        "role": "decode",
        "services": (
            {
                "data_parallel_rank": 8,
                "source_line": 841,
                "http_port_from_launcher": 8000,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "data_parallel_rank": 8,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "fc0a227c-d6e4-434e-a7cf-648554f4df03",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30300,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30300",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 9,
                "source_line": 821,
                "http_port_from_launcher": 8001,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8001,
                    "data_parallel_rank": 9,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "5d7b838a-4ca2-4a0f-9fac-ffe9d85fb81a",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30301,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30301",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 10,
                "source_line": 709,
                "http_port_from_launcher": 8002,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8002,
                    "data_parallel_rank": 10,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "065efa4d-e37e-4036-b426-e75beb1bf3e7",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30302,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30302",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 11,
                "source_line": 787,
                "http_port_from_launcher": 8003,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8003,
                    "data_parallel_rank": 11,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "30fbc053-3f71-4a5a-ba7e-90bb1ef99dc9",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30303,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30303",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
        ),
        "engine_config_source_lines": (1733, 1734, 1738, 1741),
    },
    "D3": {
        "source_log": "D3_141.61.133.116-5379/output.log",
        "source_sha256": "b85df1a2b64240ad6382e56d70eee77b42a9fc114a736745deeaa65214f1454c",
        "host": "141.61.133.116",
        "endpoint_port_from_directory": 5379,
        "working_directory": "/tmp/inference/a0081bff-11be-4137-a61c-cb86e2109dec/d0_3",
        "launch_command": "python launch_online_dp.py --dp-size 16 --tp-size 2 --pp-size 1 --dp-size-local 4 "
        "--dp-rank-start 12 --dp-address 172.27.18.114 --dp-rpc-port 12321 "
        "--vllm-start-port 8000",
        "evidence": {
            "prometheus": {
                "line": 7,
                "text": "[Prometheus] script=./run_dp_template.sh pid=67965 "
                "PROMETHEUS_MULTIPROC_DIR=/dev/shm/vllm_metrics",
            },
            "breakable_cudagraph": {
                "line": 78,
                "text": "INFO 10-15 23:52:06 [platform.py:74] Breakable "
                "CUDAGraph on Ascend is opt-in; using "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=0.",
            },
            "parallel_layout": {
                "line": 3431,
                "text": "(Worker pid=91151) INFO 10-15 23:53:43 "
                "[parallel_state.py:1827] world_size=32 rank=28 "
                "local_rank=0 "
                "distributed_init_method=tcp://172.27.18.114:55793 "
                "backend=hccl",
            },
        },
        "environment_observed": {
            "PROMETHEUS_MULTIPROC_DIR": "/dev/shm/vllm_metrics",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        },
        "environment_names_without_logged_values": ("VLLM_ASCEND_DP_ALLREDUCE_ON_NPU", "VLLM_RPC_TIMEOUT"),
        "role": "decode",
        "services": (
            {
                "data_parallel_rank": 12,
                "source_line": 774,
                "http_port_from_launcher": 8000,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "data_parallel_rank": 12,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "2b7a938b-aa71-4b26-bb80-145c876dcc27",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30300,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30300",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 13,
                "source_line": 771,
                "http_port_from_launcher": 8001,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8001,
                    "data_parallel_rank": 13,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "5549332a-928a-4071-af77-6c08c872caf0",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30301,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30301",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 14,
                "source_line": 841,
                "http_port_from_launcher": 8002,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8002,
                    "data_parallel_rank": 14,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "6fa0c820-63ac-45ad-abe6-5bb1da09f166",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30302,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30302",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
            {
                "data_parallel_rank": 15,
                "source_line": 803,
                "http_port_from_launcher": 8003,
                "arguments_reference": "DECODE_ARGUMENTS",
                "argument_overrides": {
                    "port": 8003,
                    "data_parallel_rank": 15,
                    "kv_transfer_config": {
                        "kv_connector": "MultiConnector",
                        "engine_id": "a1fa6258-e928-4224-8c64-8d46ff3496c5",
                        "kv_buffer_device": "npu",
                        "kv_buffer_size": 1000000000.0,
                        "kv_role": "kv_consumer",
                        "kv_rank": None,
                        "kv_parallel_size": 1,
                        "kv_ip": "127.0.0.1",
                        "kv_port": 30303,
                        "kv_connector_extra_config": {
                            "connectors": [
                                {
                                    "kv_connector": "MooncakeConnectorV2",
                                    "kv_role": "kv_consumer",
                                    "kv_port": "30303",
                                    "kv_connector_extra_config": {
                                        "prefill": {
                                            "dp_size": 1,
                                            "tp_size": 8,
                                            "pp_size": 2,
                                            "pp_layer_partition": "42,36",
                                        },
                                        "decode": {"dp_size": 16, "tp_size": 2},
                                    },
                                },
                                {
                                    "kv_connector": "AscendStoreConnector",
                                    "kv_role": "kv_consumer",
                                    "kv_connector_extra_config": {
                                        "lookup_rpc_port": "0",
                                        "backend": "memcache",
                                        "use_layerwise": True,
                                    },
                                },
                            ]
                        },
                        "kv_connector_module_path": None,
                        "enable_permute_local_kv": False,
                        "kv_load_failure_policy": "fail",
                    },
                },
            },
        ),
        "engine_config_source_lines": (1733, 1734, 1739, 1740),
    },
}
