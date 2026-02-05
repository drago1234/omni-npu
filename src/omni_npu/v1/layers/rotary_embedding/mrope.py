# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch

from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import apply_rotary_emb_dispatch
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YaRNScalingRotaryEmbedding, yarn_get_mscale

from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope


class MRotaryEmbedding(RotaryEmbeddingBase):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        # YaRN parameters.
        *,
        scaling_factor: float | None = None,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        if self.scaling_factor is not None:
            # Get n-d magnitude scaling corrected for interpolation
            self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        else:
            self.mscale = 1.0

        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
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
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_inv_freq(base)
        return YaRNScalingRotaryEmbedding._compute_inv_freq(self, base)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        if self.scaling_factor is None:
            return super()._compute_cos_sin_cache()
        return YaRNScalingRotaryEmbedding._compute_cos_sin_cache(self)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        is_prefill: bool = False,
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
            import torch_npu
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



    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            assert self.mrope_section

            q, k = triton_mrope(
                query,
                key,
                cos,
                sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.mrope_interleaved,
            )

            return q.reshape(query_shape), k.reshape(key_shape)

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(
                    context_len + mrope_position_delta, seq_len + mrope_position_delta
                )
            )
            for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(
        out: np.ndarray,
        out_offset: int,
        mrope_position_delta: int,
        context_len: int,
        num_new_tokens: int,
    ):
        values = np.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + context_len + num_new_tokens,
            dtype=out.dtype,
        )
        out[:, out_offset : out_offset + num_new_tokens] = values