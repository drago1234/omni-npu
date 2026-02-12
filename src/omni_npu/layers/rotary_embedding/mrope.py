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
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            if positions.ndim == 2:
                assert self.mrope_section
                if self.mrope_interleaved:
                    cos = apply_interleaved_rope(cos, self.mrope_section)
                    sin = apply_interleaved_rope(sin, self.mrope_section)
                else:
                    head_size = query.shape[-1]
                    if head_size > self.rotary_dim:
                        head_size = self.rotary_dim
                    cos, sin = [torch.cat(
                        [x[i][..., self.mrope_section_presum[i]:self.mrope_section_presum[i] + self.mrope_sections[head_size][i]] for i in range(3)],
                        dim=-1
                        ) for x in (cos, sin)]
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

            if self.cos_sin_cache.device != query.device:  # type: ignore
                self.cos_sin_cache = self.cos_sin_cache.to(  # type: ignore
                    query.device)  # type: ignore

            if self.cos_sin_cache.dtype != query.dtype:  # type: ignore
                self.cos_sin_cache = self.cos_sin_cache.to(  # type: ignore
                    query.dtype)  # type: ignore
            query, key = torch_npu.npu_mrope(positions.contiguous(),
                                            query.contiguous(),
                                            key.contiguous(),
                                            self.cos_sin_cache.contiguous(),
                                            self.head_size,
                                            mrope_section=mrope_section,
                                            rotary_mode='half')

            return query, key