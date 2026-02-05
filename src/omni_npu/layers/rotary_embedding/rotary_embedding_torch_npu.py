# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch NPU rotary embedding implementation.

This module provides the basic rotary embedding implementation for NPU devices.
It includes optimizations for different rotary_dim values and uses torch_npu
operations when available.
"""
from typing import List, Optional, Tuple, Any, Literal

import torch
import torch_npu

from .common import CachedCosSinMixin, apply_rotary_emb_full_dim
from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT 

NEOX_ROTARY_COEFF = 2

@RotaryEmbedding.register_oot
class NPURotaryEmbedding(CachedCosSinMixin, RotaryEmbedding):
    """Rotary positional embedding for NPU devices.

    This implementation provides optimized rotary embedding application for NPU
    devices. It automatically selects the appropriate implementation based on
    the rotary_dim value:
    - rotary_dim < head_size: Uses native PyTorch implementation (for ChatGLM-style)
    - rotary_dim != 128: Uses small operations implementation
    - rotary_dim == 128: Uses fused torch_npu.npu_apply_rotary_pos_emb operation
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                        is_neox_style, dtype)
        self.max_len = self.max_position_embeddings

        self._set_cos_sin_cache(device=current_platform.device_type, dtype=dtype)

    def _set_cos_sin_cache(self, device, dtype) -> None:
        """Compute the cos and sin cache separately.

        Returns:
            Tuple of (cos, sin) tensors, each of shape [max_position_embeddings, rotary_dim]
        """
        inv_freq = self._compute_inv_freq(self.base).npu()
        t = torch.arange(self.max_len, device=device, dtype=inv_freq.dtype)

        # Adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq) # shape: [max_len, rotary_dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # shape: [max_len, rotary_dim]
        else:
            freqs = torch.outer(t, inv_freq).float() # shape: [max_len, rotary_dim/2]
            emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.shape[0], -1) # shape: [max_len, rotary_dim]

        self.register_buffer("cos_cached", torch.cos(emb).to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb).to(dtype=dtype), persistent=False)

    def _forward_ascend_ops_and_small_ops(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward using ascend ops with small operations.

        Used when rotary_dim >= head_size but rotary_dim != 128.
        """
        assert position_ids.dim() == 1
        cos = torch.index_select(self.cos_cached, dim=0, index=position_ids)
        sin = torch.index_select(self.sin_cached, dim=0, index=position_ids)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
        q_embed = apply_rotary_emb_full_dim(query, cos, sin, self.is_neox_style)
        k_embed = apply_rotary_emb_full_dim(key, cos, sin, self.is_neox_style)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    def _forward_fused_ops(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward using fused torch_npu npu_apply_rotary_pos_emb operations with TND format.
        Used when rotary_dim == 128.
        """
        cos = torch.index_select(
            self.cos_cached, dim=0, index=position_ids.view(-1)).unsqueeze(1)
        sin = torch.index_select(
            self.sin_cached, dim=0, index=position_ids.view(-1)).unsqueeze(1)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()

        # Use npu_apply_rotary_pos_emb
        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, 'TND')

        # Flatten results
        q_embed_flat = q_embed.flatten(1, 2)
        k_embed_flat = k_embed.flatten(1, 2)
        return q_embed_flat, k_embed_flat

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation.

        Automatically dispatches to the appropriate implementation based on rotary_dim.
        """
        # Adapt for ChatGLM: dim = head_size / 2
        if self.rotary_dim < self.head_size:
            if self.cos_sin_cache.device != query.device:
                self.cos_sin_cache = self.cos_sin_cache.npu()
            q_embed, k_embed = super().forward_native(positions, query, key)
        elif self.rotary_dim != 128:
            # torch_npu.npu_apply_rotary_pos_emb last dim must be 128, resort to small ops when not
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(positions, query, key)
        elif self.is_neox_style:
            q_embed, k_embed = self._forward_fused_ops(positions, query, key)
        else:
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(positions, query, key)
        return q_embed, k_embed

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        output_cos_sin: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        def _apply_rotary_emb_torch(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool,
        ) -> torch.Tensor:
            cos = cos.unsqueeze(-2).to(x.dtype)
            sin = sin.unsqueeze(-2).to(x.dtype)
            if is_neox_style:
                x1, x2 = torch.chunk(x, 2, dim=-1)
            else:
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                # 0 1 2 3 4 5
                # x1 = 0 2 4, x2= 1  5
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.cat((o1, o2), dim=-1)
        
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        # patch for kvrmsnormropecache which need cos and sin
        if output_cos_sin:
            cos_cache = torch.cat((cos, cos), dim=1)
            sin_cache = torch.cat((sin, sin), dim=1)
    
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        # patch for q use npu_interleave_rope
        if output_cos_sin:
            query_rot = torch_npu.npu_interleave_rope(query_rot.unsqueeze(2), cos_cache.unsqueeze(1).unsqueeze(1), \
                sin_cache.unsqueeze(1).unsqueeze(1)).squeeze(2)
        else:
            query_rot = _apply_rotary_emb_torch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
     
        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            # patch for k use npu_interleave_rope
            if output_cos_sin:
                key_rot = torch_npu.npu_interleave_rope(key_rot.unsqueeze(2), cos_cache.unsqueeze(1).unsqueeze(1), \
                    sin_cache.unsqueeze(1).unsqueeze(1)).squeeze(2)
            else:
                key_rot = _apply_rotary_emb_torch(key_rot, cos, sin, self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        if output_cos_sin:
            return query, key, cos_cache, sin_cache
        else:
            return query, key, None, None

    #####patch start: for pangu72B-VL
    def get_rope_wrapper(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool = True,
        rope_scaling: Optional[dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
        partial_rotary_factor: float = 1.0,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        num_hidden_layers_cache: int = 1
    ) -> RotaryEmbedding:
        if rope_scaling is not None and rope_scaling.get("mrope_interleaved") == True:
            # adapt Replacing legacy 'type' key with 'rope_type' in 0.6.3
            if dtype is None:
                dtype = torch.get_default_dtype()
            if rope_scaling is not None:
                rope_scaling_tuple = {k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()}
                rope_scaling_args = tuple(rope_scaling_tuple.items())
            else:
                rope_scaling_args = None
            
            if partial_rotary_factor < 1.0:
                rotary_dim = int(rotary_dim * partial_rotary_factor)
            key = (
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                rope_scaling_args,
                None,
                dtype,
                num_hidden_layers_cache
            )
            if key in _ROPE_DICT:
                return _ROPE_DICT[key]

            mrope_section = rope_scaling.get("mrope_section")
            rotary_mode = rope_scaling.get("rotary_mode", "half")
            num_hidden_layers_cache = 1 if get_pp_group().world_size > 1 else num_hidden_layers_cache

            rotary_emb = MRotaryEmbeddingInterleaved(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=mrope_section,
                mrope_interleaved=True,
                rotary_mode=rotary_mode,
                num_hidden_layers_cache=num_hidden_layers_cache
            )

            _ROPE_DICT[key] = rotary_emb
            return rotary_emb

        if rope_scaling is None:
            rope_parameters = {"rope_theta": base, "rope_type": "default"}
        else:
            rope_parameters = rope_scaling.copy()
            if "rope_theta" not in rope_parameters:
                rope_parameters["rope_theta"] = base
            if "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = "default"

        return _orig_get_rope(
            head_size,
            rotary_dim,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
            partial_rotary_factor,
            dual_chunk_attention_config,
        )


@MRotaryEmbedding.register_oot
class MRotaryEmbeddingInterleaved(MRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections and Interleaved Support."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = True,
        rotary_mode: Literal["half", "interleaved"] = "half",
        num_hidden_layers_cache: int = 1
    ) -> None:
        # Enlarge max_position_embeddings for video inputs
        self.cache_max_position_num = max_position_embeddings
        super().__init__(
            head_size,
            rotary_dim,
            self.cache_max_position_num,
            base,
            is_neox_style,
            dtype,
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.rotary_mode = rotary_mode

        if self.mrope_section is None:
            raise ValueError("mrope_section cannot be None.")
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError(
                "Sum of mrope_section must equal rotary_dim // 2.")
        if not self.mrope_interleaved:
            raise ValueError(
                "mrope_interleaved must be True when mrope_section is provided.")

        # Generate interleaved indices
        if len(mrope_section) == 2:
            h_num, w_num = mrope_section[0], mrope_section[1]
            mrope_dim = self.get_mrope_interleaved_id_list(h_num, w_num, 0)
        elif len(mrope_section) == 3:
            t_num, h_num, w_num = mrope_section[0], mrope_section[1], mrope_section[2]
            mrope_dim = self.get_mrope_interleaved_id_list(
                t_num, h_num, w_num, force_last=True)
        else:
            raise AssertionError(
                "Cannot support the length of mrope section is not 2 or 3.")

        mrope_dim = mrope_dim * 2
        self.mrope_dim = mrope_dim

        self.layer_cache = None
        self.layer_counts = 0
        self.num_hidden_layers_cache = num_hidden_layers_cache

    def _rebuild_pos_emb(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave the rotary embedding"""
        cos_sin = self.cos_sin_cache[positions]
        mrope_section_3d = [1] * len(self.mrope_dim)
        mrope_dim = self.mrope_dim
        cos_sin = torch.cat(
            [m[mrope_dim[i]]
                for i, m in enumerate(cos_sin.split(mrope_section_3d, dim=-1))],
            dim=-1,
        )
        cos, sin = cos_sin.chunk(2, dim=-1)
        from einops import rearrange
        if self.rotary_mode == 'half':
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        elif self.rotary_mode == 'interleave':
            cos = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
            sin = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
        else:
            raise ValueError("only support half or interleave")
        cos = cos.reshape(-1, 1, 1, self.rotary_dim)
        sin = sin.reshape(-1, 1, 1, self.rotary_dim)

        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        output_cos_sin: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with interleaved rotary embedding."""
        if self.layer_counts % self.num_hidden_layers_cache == 0:
            cos, sin = self._rebuild_pos_emb(positions, output_cos_sin)
            self.layer_cache = (cos, sin)
            self.layer_counts = 0
        else:
            cos, sin = self.layer_cache
        self.layer_counts += 1

        num_tokens = query.shape[0]

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        if output_cos_sin:
            query_rot = torch_npu.npu_interleave_rope(query_rot.unsqueeze(2), cos, sin).squeeze(2)
        else:
            query_rot = torch_npu.npu_rotary_mul(
                query_rot.unsqueeze(2).contiguous(), cos, sin, rotary_mode=self.rotary_mode
            ).squeeze(2)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(num_tokens, -1)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            if output_cos_sin:
                key_rot = torch_npu.npu_interleave_rope(key_rot.unsqueeze(2), cos, sin).squeeze(2)
            else:
                key_rot = torch_npu.npu_rotary_mul(
                    key_rot.unsqueeze(2).contiguous(), cos, sin, rotary_mode=self.rotary_mode
                ).suqeeze(2)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(num_tokens, -1)

        if output_cos_sin:
            return query, key, cos, sin
        else:
            return query, key, None, None

    @staticmethod
    def get_mrope_interleaved_id_list(a: int, b: int, c: int, force_last: bool = False) -> List[int]:
        """
        Generate an interleaved list of indices for multi-modal rotary embedding.

        Args:
            a: Number of indices for first modality
            b: Number of indices for second modality
            c: Number of indices for third modality
            force_last: Whether to force the last element to be from the first modality

        Returns:
            List of interleaved indices
        """
        if force_last:
            a -= 1

        counts = {0: a, 1: b, 2: c}
        placed = {k: 0 for k in counts}
        rem = counts.copy()
        seq: List[int] = []
        last = None

        total = a + b + c
        for _ in range(total):
            # Candidates: remaining > 0 and ≠ last
            cands = [k for k in rem if rem[k] > 0 and k != last]
            if not cands:
                # If only last remains, relax the condition
                cands = [k for k in rem if rem[k] > 0]

            # Select the rarest candidate
            try:
                best = min(cands, key=lambda k: (placed[k] / counts[k], k))
            except KeyError:
                best = 0

            seq.append(best)
            placed[best] += 1
            rem[best] -= 1
            last = best

        if force_last:
            seq.append(0)

        return seq