# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch, torch_npu
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import apply_rotary_emb_dispatch
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

@MRotaryEmbedding.register_oot
class NPUMRotaryEmbedding(MRotaryEmbedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Separate caches for text rope (1D positions) and multimodal mrope (2D positions)
        self.cached_device = None    # Cached device in decode phase
        self.cached_dtype = None    # Cached dtype in decode phase
        max_len = self.cos_sin_cache.shape[1]

        all_positions = torch.arange(max_len)
        all_cos_sin = self.cos_sin_cache[all_positions]
        self.all_cos, self.all_sin = all_cos_sin.chunk(2, dim=-1)
        
        all_positions_3d = all_positions.unsqueeze(0).expand(3, -1)
        all_cos_sin_3d = self.cos_sin_cache[all_positions_3d]
        all_cos_3d, all_sin_3d = all_cos_sin_3d.chunk(2, dim=-1)
        
        if self.mrope_interleaved:
            self.all_cos_mrope = apply_interleaved_rope(all_cos_3d, self.mrope_section)
            self.all_sin_mrope = apply_interleaved_rope(all_sin_3d, self.mrope_section)
        else:
            self.all_cos_3d = all_cos_3d
            self.all_sin_3d = all_sin_3d

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        is_prefill: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        if is_prefill:
            assert positions.ndim == 1 or positions.ndim == 2
            assert key is not None
            self._match_cos_sin_cache_dtype(query)
            num_tokens = positions.shape[-1]

            # Use position-specific cache: text rope (1D) vs mrope (2D)
            if positions.ndim == 1:
                cos = self.all_cos[positions]
                sin = self.all_sin[positions]
            else:
                if self.mrope_interleaved:
                    pos_indices = positions[0]
                    cos = self.all_cos_mrope[pos_indices]
                    sin = self.all_sin_mrope[pos_indices]
                else:
                    pos_indices = positions[0]
                    cos_3d = self.all_cos_3d[:, pos_indices, :]
                    sin_3d = self.all_sin_3d[:, pos_indices, :]
                    
                    head_size = query.shape[-1] // (query.shape[0] // num_tokens)
                    if head_size > self.rotary_dim:
                        head_size = self.rotary_dim
                    
                    cos, sin = [torch.cat(
                        [x[i][..., self.mrope_section_presum[i]:self.mrope_section_presum[i] + self.mrope_sections[head_size][i]] for i in range(3)],
                        dim=-1
                    ) for x in (cos_3d, sin_3d)]
                

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = query[..., : self.rotary_dim]
            query_pass = query[..., self.rotary_dim :]
            query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin, self.is_neox_style)
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin, self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            return query, key
        else:
            positions = positions[0]
            mrope_section = [0, 0, 0
                            ] if positions.ndim == 1 else self.mrope_section

            # Cache device and dtype transfer to avoid repeated checks per layer
            if self.cached_device is None or self.cached_device != query.device:
                self.cos_sin_cache = self.cos_sin_cache.to(query.device)
                self.cached_device = query.device
            if self.cached_dtype is None or self.cached_dtype != query.dtype:
                self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
                self.cached_dtype = query.dtype

            query, key = torch_npu.npu_mrope(positions.contiguous(),
                                            query.contiguous(),
                                            key.contiguous(),
                                            self.cos_sin_cache.contiguous(),
                                            self.head_size,
                                            mrope_section=mrope_section,
                                            rotary_mode='half')

            return query, key