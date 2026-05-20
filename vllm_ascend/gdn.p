#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from typing import Any
import torch
from vllm.logger import logger
import torch_npu
import torch.nn.functional as F
from einops import rearrange
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.utils import maybe_save_kv_layer_to_connector
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_update_npu
from vllm.distributed import get_pcp_group
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.distributed import (
    get_dcp_group,
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
)


def to_int64_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    tensor = tensor.to(torch.int64)
    if tensor.dim() == 0:
        return (tensor.item(),)
    return tuple(tensor.tolist())


def _require_non_spec_prefill_fallback_meta(attn_metadata, field_name: str):
    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    if fallback_meta is None:
        raise RuntimeError(
            f"Expected attn_metadata.non_spec_prefill_fallback_meta.{field_name} for patched GDN non-spec prefill path."
        )
    return fallback_meta


def get_non_spec_causal_conv1d_host_args(attn_metadata) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "causal_conv1d")
    causal_conv1d_meta = fallback_meta.causal_conv1d
    return (
        to_int64_tuple(causal_conv1d_meta.query_start_loc_cpu),
        to_int64_tuple(causal_conv1d_meta.cache_indices_cpu),
        to_int64_tuple(causal_conv1d_meta.has_initial_state_cpu),
    )


def get_non_spec_chunked_prefill_meta(attn_metadata):
    fallback_meta = _require_non_spec_prefill_fallback_meta(attn_metadata, "chunk")
    return fallback_meta.chunk



def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata: Any | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]
    last_width_prefill_x = extract_last_width(x, query_start_loc[num_decodes:], conv_states.shape[-1])

    if get_pcp_group().world_size > 1:
        all_last_width_prefill_x = get_pcp_group().all_gather(last_width_prefill_x.unsqueeze(0).contiguous(), 0)
        pcp_rank = get_pcp_group().rank_in_group
        if pcp_rank > 0:
            conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[pcp_rank - 1, ...]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)],
            )
        )

    if get_pcp_group().world_size > 1:
        conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[-1, ...]
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)
