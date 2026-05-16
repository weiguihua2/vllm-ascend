#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_dcp_group, get_dp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import LazyLoader
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.cp_utils import (
    get_total_cp_world_size,
)
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.utils import AttentionGroup

# yapf: enable
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, using_paged_attention

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    set_draft_graph_params,
    set_graph_params,
    update_full_graph_params,
)
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess
from vllm_ascend.eplb.eplb_updator import EplbUpdator
from vllm_ascend.eplb.utils import model_register
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.patch.worker.patch_module import patch_torch_npu_argsort
from vllm_ascend.quantization.utils import enable_fa_quant
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer
from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm_ascend.utils import (
    calc_split_factor,
    check_gdn_layer,
    enable_sp,
    enable_sp_by_pass,
    get_c_env,
    global_stream,
    kv_cache_spec_uses_sparse_c8,
    lmhead_tp_enable,
    set_weight_prefetch_method,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch
from vllm_ascend.worker.pcp_utils import PCPManager

from vllm_ascend.ascend_forward_context import (  # isort: skip
    MoECommType,
    get_mc2_tokens_capacity,
    select_moe_comm_method,
    set_ascend_forward_context,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")


from vllm.model_executor.layers.attention import Attention, MLAAttention

# if true, allow tensor initialization and casting with internal format (e.g., NZ)
torch.npu.config.allow_internal_format = True

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


def get_tp_context(drafter):
    return getattr(drafter, "tp_group_context", nullcontext())


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: "PerLayerAttnMetadata"
    positions: torch.Tensor
    ec_connector_output: "ECConnectorOutput | None"
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor


class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers, which need to be reverted after
        # the following PR is merged:
        # https://github.com/vllm-project/vllm/pull/28988
        max_pcp_pad_tokens = (
            vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        )
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 2,  # type: ignore[has-type]
            dtype=torch.int32,
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        self._has_gdn = check_gdn_layer(vllm_config)
        if self._has_gdn:
            self.gdn_query_start_loc = self._make_buffer(
                self.max_num_reqs + 1,  # type: ignore[has-type]
                dtype=torch.int32,
            )

        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None
        self.tp_rank = get_tensor_model_parallel_rank()

        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)
        # Dump / PrecisionDebugger configuration now comes from AscendConfig
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger

                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                raise RuntimeError("Dumping/debugging only works in eager mode.")
        # use_hybrid_blocks: if hybrid blocks is used.
        self.use_hybrid_blocks: bool = False
        self.need_accepted_tokens: bool = False

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size
        # Set up Attention
        self.use_sparse = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk"
        )
        if self.use_sparse:
            self.sparse_head_dim = (
                self.model_config.hf_text_config.kv_lora_rank,
                self.model_config.hf_text_config.qk_rope_head_dim,
                self.model_config.hf_text_config.index_head_dim,
            )
        # dsa c8
        self.use_sparse_c8_indexer = self.ascend_config.enable_sparse_c8
        if self.use_sparse_c8_indexer:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

        self.attn_backend = get_attn_backend(
            0,
            self.dtype,
            None,
            use_mla=self.model_config.use_mla,
            use_sparse=self.use_sparse,
            use_mm_prefix=self.model_config is not None and self.model_config.is_mm_prefix_lm,
        )

        try:
            self.dcp_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.pcp_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        except Exception:
            self.dcp_size = 1
            self.dcp_rank = 0
            self.pcp_size = 1
            self.pcp_rank = 0
        if self.pcp_size > 1:
            self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        max_buffer_num_tokens = self.max_num_tokens
        if self.pcp_size * self.dcp_size > 1:
            max_buffer_num_tokens = self.max_num_tokens + self.max_num_reqs * 2 * self.pcp_size
            self.pcp_manager = PCPManager(
                self.pcp_size,
                self.pcp_rank,
                self.dcp_size,
                self.dcp_rank,
                max_buffer_num_tokens,
                self.max_num_reqs,
                self.device,
                self.vllm_config,
                self.use_async_scheduling,
                self.pin_memory,
                self.use_sparse,
            )
            # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
            self.input_ids = self._make_buffer(max_buffer_num_tokens, dtype=torch.int32)
            self.positions = torch.zeros(
                max_buffer_num_tokens, dtype=torch.int64, device=self.device)
            
        # Create a CPU numpy buffer for positions computation when
        # self.positions is a plain tensor (non-CpuGpuBuffer case).
        self._positions_cpu_buf = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._positions_np_buf = self._positions_cpu_buf.numpy()

        self.use_eagle = (
            vllm_config.speculative_config.use_eagle()
            if vllm_config.speculative_config
            else None
        )
        # When True, run update_full_graph_params before self.model (ENPU / graph capture order).
        # Internal / non-public toggle: read C getenv ``ENPU_ENABLE`` from enpu code (not in envs.py).
        _enpu = get_c_env("ENPU_ENABLE")
        self.enable_enpu = _enpu is not None and _enpu.lower() == "true"

        self._set_up_drafter()

        # Event for async GPU→CPU copy of corrected seq_lens in async
        # spec decode mode. Recorded in _prepare_inputs, synchronized
        # in _build_attention_metadata. Created once, reused each iteration.
        # Only backends that consume CPU seq_lens (AscendAttentionBackend,
        # AscendMLABackend) need this; others (SFA, GDN, etc.) do not.
        self._needs_seq_lens_cpu_sync = issubclass(
            self.attn_backend, (AscendAttentionBackend, AscendMLABackend)
        )
        self._seq_lens_cpu_event: torch.npu.Event | None = None
        self._seq_lens_cpu_event_pending = False

        # kv role
        self.is_kv_producer = False
        self.is_kv_consumer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
            self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

        set_cos_and_sin(vllm_config, self.max_num_reqs, self.uniform_decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.uniform_decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        self.decode_threshold = 1 + (self.speculative_config.num_speculative_tokens if self.speculative_config else 0)

        self.use_aclgraph = self._use_aclgraph()

        eplb_config = self.ascend_config.eplb_config
        self.dynamic_eplb = eplb_config.dynamic_eplb
        self.eplb_enable = self.dynamic_eplb or (eplb_config.expert_map_path is not None)
        if self.dynamic_eplb:
            self.is_eplb_warmuped = False
            self.policy_type = eplb_config.eplb_policy_type
            self.eplb_loader = D2DExpertWeightLoader()
            self.manager = Manager()
            self.shared_dict = self.manager.dict({"expert_map": None, "moe_load": None, "expert_maps": None})
            self.eplb_process = EplbProcess(shared_dict=self.shared_dict, policy_type=self.policy_type, enable_d2d=True)
            self.process = self.eplb_process._launch_process()
            self.eplb_updator = EplbUpdator(eplb_config, self.eplb_loader, self.eplb_process, self.process)
        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = NPUInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[[self.cache_config.block_size]],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors,
            ),
            logitsprocs_need_output_token_ids=bool(
                self.vllm_config.model_config.logits_processors
            ),
            is_pooling_model=self.is_pooling_model,
            num_speculative_tokens=(
                self.vllm_config.speculative_config.num_speculative_tokens if self.vllm_config.speculative_config else 0
            ),
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # here we use int32
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        # for cleancode , actually the three attrs is defined in gpu_model_runner
        self.execute_model_state: ExecuteModelState | None = None
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None
        self.reorder_batch_threshold: int | None = None
        self.long_seq_metadata = None
        self.query_lens: torch.Tensor | None = None
        self.cpu_slot_mapping = None
        self.sampling_done_event: torch.npu.Event | None = None

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)
        else:
            self.cudagraph_batch_sizes = []
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_copy_bufs: mamba_utils.MambaCopyBuffers | None = None
        self.enable_hamming_sparse = (self.ascend_config.enable_hamming_sparse is True)
        self.enable_hamming_sparse = self.enable_hamming_sparse and not vllm_config.speculative_config
        if self.enable_hamming_sparse is True:
            from vllm_ascend.worker.kvcomp_utils import initialize_kvcomp_metadata
            self.kvcomp_meta_data = initialize_kvcomp_metadata(max_num_reqs=self.max_num_reqs,
                block_size=self.block_size, device=self.device, vllm_config=self.vllm_config,
                parallel_config=self.parallel_config, dtype=self.dtype)

    @property
    def use_cp(self) -> bool:
        return self.pcp_size * self.dcp_size > 1

    def _init_device_properties(self) -> None:
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _set_up_drafter(self):
        # Set up speculative decoding.
        self.drafter: (
            AscendNgramProposer
            | AscendEagleProposer
            | AscendDraftModelProposer
            | AscendDflashProposer
            | AscendSuffixDecodingProposer
            | AscendMedusaProposer
            | None
        ) = None
        self.actual_seq_lengths_q: list[int] = []
        self.decode_token_per_req = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            assert spec_token_num > 0
            self.decode_token_per_req = 1 + spec_token_num
            if get_pp_group().is_last_rank:
                self.drafter = self._get_drafter()
                if self.speculative_config.method == "eagle3":
                    assert isinstance(self.drafter, AscendEagleProposer)
                    self.use_aux_hidden_state_outputs = self.drafter.eagle3_use_aux_hidden_state
                self.rejection_sampler = RejectionSampler(self.sampler)
        self.discard_request_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_discarded_requests = 0

    def _get_drafter(self):
        return get_spec_decode_method(self.speculative_config.method, self.vllm_config, self.device, self)

    def _use_aclgraph(self) -> bool:
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )

    def _sync_metadata_across_dp(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
        # our case, we still need to sync the other two flags as well. So we need to
        # include them in the all_reduce operation, and more over, we CANNOT skip it
        # even if we are running in eager mode, which harms performance.
        # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
        # immediately once the other two flags are no longer needed.
        if self.dp_size == 1:
            return num_tokens, None, cudagraph_mode

        if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
            num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
            return num_tokens, num_tokens_after_padding, cudagraph_mode

        packed_tensor = torch.zeros(2, self.dp_size, device="cpu", dtype=torch.int32)
        packed_tensor[0][self.dp_rank] = num_tokens
        packed_tensor[1][self.dp_rank] = cudagraph_mode.value
        dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)

        # Unpack the results
        num_tokens_across_dp = packed_tensor[0, :]
        max_tokens_across_dp = int(num_tokens_across_dp.max().item())
        synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor))

        # Create a tensor for num_tokens_after_padding
        if allow_dp_padding or is_draft_model:
            num_tokens_after_padding = torch.tensor(
                [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
            )
        else:
            num_tokens_after_padding = num_tokens_across_dp.cpu()

        return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode

    def get_model(self) -> nn.Module:
        # get raw model out of the aclgraph wrapper.
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.unwrap()
        return self.model

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> int:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if cudagraph_runtime_mode == CUDAGraphMode.FULL and \
            self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.uniform_decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = self.query_start_loc.np[num_reqs]
            self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1] = (
                self.arange_np[1 : num_reqs_padded + 1 - num_reqs] * self.uniform_decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            self.query_start_loc.np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        self.query_start_loc.copy_to_gpu()

        return num_reqs_padded

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """
        :return: tuple[
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # Get the attention state.
        if not scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = num_scheduled_tokens
        else:
            num_valid_tokens = np.array(
                [
                    scheduler_output.num_scheduled_tokens[i]
                    - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for i in self.input_batch.req_ids
                ],
                dtype=np.int32,
            )
        attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)

        # Determine if it's a splitfuse batch
        with_prefill = attn_state not in [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding]
        self.with_prefill = with_prefill

        # Get positions.
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            self.query_pos.np[: cu_num_tokens[-1]],
            out=positions_np,
        )

        # For PCP, compute slot_mapping on GPU using pre-PCP-split positions.
        # Use blocking .to(device) to ensure data lands on GPU before PCP
        # modifies CPU position buffers. PCP and async spec decode are
        # mutually exclusive, so the sync is acceptable.
        if self.pcp_size > 1:
            pre_pcp_positions = torch.from_numpy(
                positions_np[:total_num_scheduled_tokens]
            ).to(self.device)
            pre_pcp_qsl = torch.zeros(
                num_reqs + 1, dtype=torch.int32, device=self.device)
            pre_pcp_qsl[1:num_reqs + 1] = torch.from_numpy(
                cu_num_tokens
            ).to(dtype=torch.int32, device=self.device)
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs, pre_pcp_qsl, pre_pcp_positions)

        if self.use_cp:
            self.pcp_manager.init_batch_info(
                num_scheduled_tokens,
                self.input_batch.num_reqs,
            )

        # for pcp, prefill mtp should use origin scheduleroutput ,
        if self.speculative_config and self.use_cp:
            self.pcp_manager.generate_pcp_mtp_input(
                total_num_scheduled_tokens,
                scheduler_output.num_scheduled_tokens,
                with_prefill,
                self.input_batch,
                self.arange_np,
                req_indices,
                positions_np,
                cu_num_tokens,
                self._draft_token_ids,  # type: ignore[has-type]
                scheduler_output,
                self.num_spec_tokens,
            )

        if self.pcp_size > 1:
            num_scheduled_tokens[:num_reqs], position_pcp = self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs], self.arange_np
            )
            # Re-update after PCP split sequences.
            total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
            positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                position_pcp[:total_num_scheduled_tokens],
                out=positions_np,
            )
        if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
            assert self.pcp_manager.num_scheduled_tokens_padded is not None
            self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
        else:
            self.query_lens = torch.from_numpy(num_scheduled_tokens)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        token_indices_tensor = torch.from_numpy(token_indices)
        # Prepare input_ids.
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids, 0, token_indices_tensor, out=self.is_token_ids.cpu[:total_num_scheduled_tokens]
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds and (self.is_multimodal_model or self.enable_prompt_embeds):
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                if self.pcp_size > 1:
                    # PCP can split one request into non-contiguous token positions.
                    # We must gather prompt embeds by actual scheduled positions.
                    req_positions_np = positions_np[output_idx : output_idx + num_sched]
                    dst_slice = self.inputs_embeds.cpu[output_idx : output_idx + num_sched]
                    self.pcp_manager.fill_prompt_embeds_for_pcp(
                        req_embeds=req_embeds,
                        req_positions_np=req_positions_np,
                        dst_slice=dst_slice,
                    )
                else:
                    start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                    # Skip if trying to read beyond available embeddings
                    if start_pos >= req_embeds.shape[0]:
                        output_idx += num_sched
                        continue

                    # Copy available embeddings
                    end_pos = start_pos + num_sched
                    actual_end = min(end_pos, req_embeds.shape[0])
                    actual_num_sched = actual_end - start_pos

                    if actual_num_sched > 0:
                        self.inputs_embeds.cpu[output_idx : output_idx + actual_num_sched].copy_(
                            req_embeds[start_pos:actual_end]
                        )

                output_idx += num_sched

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.copy_to_gpu()

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
            self.gdn_query_start_loc.copy_to_gpu()


        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # Build prev_positions mapping: current pos -> prev pos (-1 if new).
        # Used for gathering from previous iteration's GPU tensors.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        # Fill unused with -1. Needed for reshape_and_cache in attention_cp
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

        # Copy the tensors to the NPU.
        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)
        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.gpu.copy_(
                self.mrope_positions.cpu,
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)
        base_num_reqs = self.input_batch.num_reqs
        num_reqs = base_num_reqs
        tokens_original = None
        if self.pcp_size > 1:
            # while pcp > 1, we need the original num_scheduled_tokens before split
            # to calculate discard_requests_mask
            tokens_original = [scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids]
            original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[:num_reqs] + np.array(
                tokens_original, dtype=np.int32
            )
            discard_requests_mask = original_seq_lens_np < num_tokens_np
        else:
            discard_requests_mask = self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np

        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = discard_request_indices
        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Sync num_accepted_tokens from CPU (set by
        # _update_states_after_model_execute for hybrid models).
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        # Update num_computed_tokens on GPU. In async spec decode,
        # CPU values are optimistic (all drafts accepted). The kernel
        # corrects on GPU using the previous step's
        # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
        else:
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        # fix prefix cache ci test
        if self.pcp_size > 1:
            # When PCP (Prefill Context Parallel) is enabled, positions use
            # special PCP offsets (position_pcp) that are only computed on CPU.
            # Copy the correctly-computed CPU positions to GPU instead of
            # recomputing on GPU (which would miss the PCP offsets).
            self.positions[:total_num_scheduled_tokens].copy_(
                torch.from_numpy(
                    positions_np[:total_num_scheduled_tokens]
                ).to(self.device),
                non_blocking=True,
            )
        else:
            self.positions[:total_num_scheduled_tokens] = (
                self.num_computed_tokens[req_indices_gpu].to(torch.int64)
                + self.query_pos.gpu[:total_num_scheduled_tokens]
            )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # In async spec decode mode, num_computed_tokens was corrected on GPU
        # by update_num_computed_tokens_for_batch_change, so seq_lens (GPU) is
        # correct but optimistic_seq_lens_cpu is stale (it assumed all drafts
        # were accepted). Sync the corrected values back to CPU so that NPU
        # attention backends (which use _seq_lens_cpu) get the right values.
        # Use non_blocking copy to pinned memory and record an event;
        # _build_attention_metadata will synchronize before reading.
        if (
            self._needs_seq_lens_cpu_sync
            and self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            self.optimistic_seq_lens_cpu[:num_reqs].copy_(
                self.seq_lens[:num_reqs], non_blocking=True
            )
            if self._seq_lens_cpu_event is None:
                self._seq_lens_cpu_event = torch.npu.Event()
            self._seq_lens_cpu_event.record()
            self._seq_lens_cpu_event_pending = True
        else:
            self._seq_lens_cpu_event_pending = False

        # For non-PCP, compute slot_mapping on GPU. PCP slot_mapping was
        # already computed on GPU before PCP split the positions.
        if self.pcp_size <= 1:
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                self.query_start_loc.gpu[: num_reqs + 1],
                self.positions[:total_num_scheduled_tokens],
            )

        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
            num_draft_tokens = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
            if self.use_cp:
                logits_indices = self.pcp_manager.get_logits_indices(cu_num_tokens, num_reqs, tokens_original)
                logits_indices = logits_indices.pin_memory().to(self.device, non_blocking=True)
            else:
                logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                if (self.is_kv_consumer and req_id in new_schedule_reqs) or \
                   (self.input_batch.num_computed_tokens_cpu[req_idx] >= \
                    self.input_batch.num_prompt_tokens[req_idx]):
                    num_decode_draft_tokens[req_idx] = draft_len
                else:
                    num_decode_draft_tokens[req_idx] = -1

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens,
                cu_num_tokens,
                num_pcp_pads=self.pcp_manager.num_pcp_pads_cpu[:num_reqs] if self.pcp_size > 1 else None,
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()
        # save logits_indices for pcp spec decode usage
        self.logits_indices = logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            assert np.sum(num_sampled_tokens) <= self.vllm_config.scheduler_config.max_num_batched_tokens
            self.set_active_loras(self.input_batch, num_scheduled_tokens, num_sampled_tokens)
        if lmhead_tp_enable():
            max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
            logits_indices = nn.functional.pad(logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0]))

        # Cache local scheduled token layout for PCP-aware multimodal preprocess.
        if (
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            self.pcp_manager.cache_local_schedule_layout(
                num_scheduled_tokens=num_scheduled_tokens,
                num_reqs=base_num_reqs,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
            )

        return (
            logits_indices,
            spec_decode_metadata,
            total_num_scheduled_tokens,
        )
