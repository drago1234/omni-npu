# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
import torch.nn.functional as F
from vllm.attention.backends.utils import PAD_SLOT_ID

def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen) or (dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1 + num_spec) or (dim, width - 1 + num_spec)
    final_states_out: (batch, width - 1 + num_spec, dim)
    out: (batch, dim, seqlen)
    """
    torch.npu.config.allow_internal_format = False
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        unsqueeze = x.dim() ==2
        if unsqueeze:
            x = x.unsqueeze(0)
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
        if unsqueeze:
            out = out.squeeze(0)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
                dtype_in)  # (batch, dim, seqlen + padding_size)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    num_spec_tokens=0,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    num_accepted_tokens: (batch,), dtype int32, optional
        If provided, used to offset where conv_state is read from.
        For spec decoding: conv_state_token_offset = num_accepted_tokens
    num_spec_tokens: int, optional
        Number of speculative tokens. Used for speculative decoding.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    torch.npu.config.allow_internal_format = False
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        if num_accepted_tokens is not None:
            # Create indices for gathering conv_state slices
            batch_indices = torch.clamp(conv_state_indices, 0)  # (batch,)
            conv_state_batch = conv_state[batch_indices]
            conv_state_to_select = [0] * (num_spec_tokens + 1)
            for i in range(num_spec_tokens + 1):
                conv_state_to_select[i] = conv_state_batch[...,
                                                           i:i + width - 1]
            conv_state_selected = torch.zeros_like(conv_state_to_select[0])
            for i in range(num_spec_tokens + 1):
                select_mask = (num_accepted_tokens == i).to(
                    conv_state_selected.dtype)
                conv_state_selected += conv_state_to_select[
                    i] * select_mask.unsqueeze(-1).unsqueeze(-1)
            x_new = torch.cat([conv_state_selected, x],
                              dim=-1).to(weight.dtype)
        else:
            x_new = torch.cat(
                [conv_state[torch.clamp(conv_state_indices, 0)], x],
                dim=-1).to(weight.dtype)
        torch_npu.npu_scatter_nd_update_(conv_state,
                                         conv_state_indices.unsqueeze(1),
                                         x_new[:, :, -state_len:])

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)

def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
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
                final_states_out=conv_states[cache_indices[i]].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]].unsqueeze(0)
                if has_initial_state[i] else None))
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor

def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    # Handle boolean activation to string if needed
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    # Handle unsqueeze/squeeze for single token input
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    # Call the pure PyTorch reference implementation
    # Note: causal_conv1d_update_ref updates conv_state inplace and returns the output.
    # Parameters like intermediate_conv_window, pad_slot_id, metadata
    # are specific to the Triton implementation or debugging and are not directly used
    # by causal_conv1d_update_ref.
    #
    # IMPORTANT CONSIDERATION REGARDING `pad_slot_id`:
    # The original Triton kernel explicitly skips processing for `pad_slot_id` entries
    # within the batch. The `causal_conv1d_update_ref` function, as provided, does NOT
    # have this batch-wise skipping logic. If `conv_state_indices` contains `pad_slot_id`
    # (e.g., -1), and this value is an invalid index for `conv_state`, the `causal_conv1d_update_ref`
    # will likely raise an indexing error.
    # If this skipping behavior is critical, you would need to implement a loop
    # over the batch dimension here to filter out padded entries before calling
    # `causal_conv1d_update_ref` for each valid entry, which would significantly
    # change the function's structure and performance (losing batching).
    # For a direct replacement of the Triton call, we assume `conv_state_indices`
    # will contain valid indices or that `pad_slot_id` skipping is handled upstream.
    out = causal_conv1d_update_ref(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out
