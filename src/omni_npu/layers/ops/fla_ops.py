# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu
import torch.nn.functional as F
from math import log

def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_state = initial_state.contiguous().transpose(-1,-2)
    assert inplace_final_state == True
    eq_len = cu_seqlens is None
    contiguous_states = ssm_state_indices is None
    bs = q.shape[0] if eq_len else len(cu_seqlens) - 1
    T = q.shape[1] // bs
    hv_over_h = v.shape[-2] // q.shape[-2]
    if scale is None:
        scale = k.shape[-1]**-0.5
    #Apply QK L2 norm if enabled teeheehee
    if use_qk_l2norm_in_kernel:
        q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-6)
        k = k / (torch.linalg.norm(k, dim=-1, keepdim=True) + 1e-6)
    q = q * scale
    g = g.exp()
    q = q.repeat(1, 1, 1, hv_over_h).reshape(v.shape)
    k = k.repeat(1, 1, 1, hv_over_h).reshape(v.shape)
    indices_0 = torch.arange(bs, device=q.device) * T
    if not contiguous_states:
        if num_accepted_tokens is None:
            indices = ssm_state_indices.view(-1)[indices_0]
        else:
            indices = ssm_state_indices.view(-1)[indices_0 + num_accepted_tokens]
    else:
        indices = indices_0 if eq_len else cu_seqlens[:-1]
    S = initial_state[indices].to(torch.float32) #[bs, C, D, E]
    #reshape for tensor operation
    A, tbs, C, D = q.shape
    E = v.shape[-1]
    q = q.reshape(A, bs, T, C, D).transpose(1, 2) #[A, T, bs, C, D]
    k = k.reshape(A, bs, T, C, D).transpose(1, 2) #[A, T, bs, C, D]
    v = v.reshape(A, bs, T, C, E).transpose(1, 2) #[A, T, bs, C, E]
    g = g.reshape(A, bs, T, C).transpose(1, 2) #[A, T, bs, C]
    beta = beta.reshape(A, bs, T, C).transpose(1, 2) #[A, T, bs, C]
    o = []
    for t in range(T):
        q_t = q[0][t].to(torch.float32) #[bs, C, D]
        k_t = k[0][t].to(torch.float32) #[bs, C, D]
        v_t = v[0][t].to(torch.float32) #[bs, C, E]
        g_t = g[0][t].to(torch.float32) #[bs, C]
        beta_t = beta[0][t].to(torch.float32) #[bs, C]
        S = g_t.view(bs, C, 1, 1) * S
        x = torch.einsum('abc,abcd->abd', k_t, S) #[bs, C, E]
        y = beta_t.unsqueeze(-1) * (v_t - x) #[bs, C, E]
        S_ = k_t.unsqueeze(-1) * y.unsqueeze(-2) #[bs, C, D, E]
        S = S + S_ #[bs, C, D, E]
        o_t = torch.einsum('abc,abcd->abd', q_t, S) #[bs, C, E]
        if not contiguous_states:
            indices = ssm_state_indices.view(-1)[indices_0 + t]
        else:
            indices = cu_seqlens[:-1] + t if eq_len else indices_0 + t
        torch_npu.npu_scatter_nd_update_(initial_state, indices.unsqueeze(1), S.to(torch.bfloat16))
        o.append(o_t.to(torch.bfloat16))
    o = torch.cat(o, dim = 1)
    o = o.contiguous().reshape(A, tbs, C, E)
    return o, initial_state.contiguous().transpose(-1,-2)

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state

def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    batch_size = initial_state.shape[0]
    core_attn_outs = []
    last_recurrent_states = []
    for b_idx in range(batch_size):
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        cur_q = q[:, start:end, ...]
        cur_k = k[:, start:end, ...]
        cur_v = v[:, start:end, ...]
        cur_g = g[:, start:end, ...]
        cur_b = beta[:, start:end, ...]
        cur_state = initial_state[b_idx].unsqueeze(0)

        chunk_size = 128
        initial_dtype = cur_q.dtype
        cur_q = cur_q.contiguous()
        cur_k = cur_k.contiguous()
        if use_qk_l2norm_in_kernel:
            cur_q = F.normalize(cur_q, p=2, dim=-1)
            cur_k = F.normalize(cur_k, p=2, dim=-1)
        cur_q, cur_k, cur_v, cur_b, cur_g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (cur_q, cur_k, cur_v, cur_b, cur_g)
        ]

        bs, num_qk_heads, sequence_length, k_head_dim = cur_k.shape
        num_v_heads = cur_v.shape[1]
        v_head_dim = cur_v.shape[-1]
        scale = 1 / (k_head_dim**0.5)
        cur_q.mul_(scale)
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        cur_q = F.pad(cur_q, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
        cur_k = F.pad(cur_k, (0, 0, 0, pad_size)).repeat_interleave(num_v_heads // num_qk_heads, dim=1)
        cur_v = F.pad(cur_v, (0, 0, 0, pad_size))
        cur_b = F.pad(cur_b, (0, pad_size)).unsqueeze(-1)
        cur_g = F.pad(cur_g, (0, pad_size))
        sequence_length_padded = sequence_length + pad_size
        v_beta = cur_v * cur_b
        k_beta = cur_k * cur_b
        # reshape to chunks
        cur_q, cur_k, cur_v, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (cur_q, cur_k, cur_v, k_beta, v_beta)
        ]
        cur_g = cur_g.reshape(cur_g.shape[0], cur_g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=cur_q.device),
                        diagonal=0)

        # chunk decay
        cur_g = cur_g.cumsum(dim=-1)
        decay_mask = (cur_g.unsqueeze(-1) -
                    cur_g.unsqueeze(-2)).exp().float()
        attn = -(
            (k_beta @ cur_k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        lg = int(log(chunk_size, 2))

        block_size = 1
        attn_inv = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device).repeat((tuple(attn.shape)[:-2] + (1, 1)))
        attn = attn_inv - attn
        for i in range(lg):
            block_num = chunk_size // block_size
            prod = attn @ attn_inv
            attn_inv_block = attn_inv.view(tuple(attn.shape)[:-2] + (block_num, block_size, block_num, block_size)).transpose(-2, -3)
            prod_block = prod.view(tuple(attn.shape)[:-2] + (block_num, block_size, block_num, block_size)).transpose(-2, -3)
            r0 = torch.arange(block_num // 2, device=attn.device) * 2
            r1 = r0 + 1
            attn_inv_block[:, :, :, r1, r0, :, :] = -attn_inv_block[..., r1, r1, :, :] @ prod_block[..., r1, r0, :, :]
            attn_inv = attn_inv_block.transpose(-2, -3).view(tuple(attn_inv_block.shape)[:-4] + (chunk_size, chunk_size))
            block_size *= 2
        attn = attn_inv

        cur_v = attn @ v_beta
        gexp = cur_g.exp()
        k_cumdecay = attn @ (k_beta * gexp.unsqueeze(-1))

        last_recurrent_state = (torch.zeros(bs, num_v_heads,
                                            k_head_dim, v_head_dim).to(cur_v) if
                                cur_state is None else cur_state.to(cur_v))

        core_attn_out = torch.zeros_like(cur_v)
        mask = torch.triu(torch.ones(chunk_size,
                                    chunk_size,
                                    dtype=torch.bool,
                                    device=cur_q.device),
                        diagonal=1)
        query_view = cur_q.reshape(cur_q.shape[0], cur_q.shape[1], -1, chunk_size, cur_q.shape[-1])
        key_trans = cur_k.reshape(cur_k.shape[0], cur_k.shape[1], -1, chunk_size, cur_k.shape[-1]).transpose(-1, -2)
        qk = query_view @ key_trans
        attn_score = qk * decay_mask.masked_fill_(mask, 0)

        gexp = cur_g[:, :, :, :, None].exp()
        qgexp = cur_q * gexp

        kgexp = (cur_g[:, :, :, -1, None] - cur_g[:, :, :]).exp()[..., None]
        kgexp = cur_k * kgexp

        k_cumdecay_qgexp = torch.cat([k_cumdecay, qgexp], dim=3)
        v_new_out = torch.zeros_like(cur_v)
        attn_inter_out = torch.zeros_like(cur_v)

        # for each chunk
        for i in range(0, sequence_length_padded // chunk_size):
            v_i = cur_v[:, :, i]
            attn = attn_score[:, :, i]
            v_prime_attn_inter = (k_cumdecay_qgexp[:, :, i]) @ last_recurrent_state
            v_prime = v_prime_attn_inter[:, :, :chunk_size]
            attn_inter = v_prime_attn_inter[:, :, chunk_size:]
            v_new = v_i - v_prime
            v_new_out[:, :, i] = v_new
            attn_inter_out[:, :, i] = attn_inter
            last_recurrent_state *= gexp[:, :, i, -1, :, None]
            last_recurrent_state += (kgexp[:, :, i]).transpose(-1, -2) @ v_new
        core_attn_out = attn_inter_out + attn_score @ v_new_out

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0],
                                            core_attn_out.shape[1], -1,
                                            core_attn_out.shape[-1])
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1,
                                                2).contiguous().to(initial_dtype)
        # return core_attn_out, last_recurrent_state.transpose(-1,-2)

        core_attn_outs.append(core_attn_out)
        last_recurrent_states.append(last_recurrent_state.transpose(-1,-2))
    
    tar_dtype = core_attn_outs[0].dtype
    tar_device = core_attn_outs[0].device
    tar_shape = list(core_attn_outs[0].shape)
    tar_shape[1] = cu_seqlens[-1]
    core_attn_out_non_spec = torch.empty(tar_shape,
                                            dtype=tar_dtype,
                                            device=tar_device)
    for b_idx in range(batch_size):
        cur_core_attn_out = core_attn_outs[b_idx]
        start, end = cu_seqlens[b_idx], cu_seqlens[b_idx + 1]
        core_attn_out_non_spec[:, start:end, ...] = cur_core_attn_out
    last_recurrent_states = torch.cat(last_recurrent_states, dim=0)
    return core_attn_out_non_spec, last_recurrent_states
