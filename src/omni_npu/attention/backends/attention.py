"""
NPU Attention backend vendored into omni_npu.attention.backends.

This is a minimized, self-contained version derived from omniinfer sources,
with the following adjustments:
- Replaced omni.* imports with local shims or safe defaults.
- Kept the core torch_npu kernels usage for prefill and decode.
- Removed optional features (best_ep, omni_cache, DSA, SP) for an MVP.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, ClassVar
import math

import torch
import torch_npu

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder as V1AttentionMetadataBuilder,
    CommonAttentionMetadata,
    AttentionCGSupport,
)
from vllm.v1.kv_cache_interface import AttentionSpec


NZ_DIM = 16


@dataclass
class NPUMetadata:
    num_actual_tokens: int
    block_tables: torch.Tensor
    query_cumlens: list[int]
    seq_lens: list[int]
    max_query_len: Optional[int] = None
    slot_mapping: torch.Tensor = None


class NPUAttentionMetadataBuilder(V1AttentionMetadataBuilder[NPUMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> NPUMetadata:

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_cumlens = common_attn_metadata.query_start_loc[1:]
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        max_query_len = common_attn_metadata.max_query_len
        attn_metadata = NPUMetadata(num_actual_tokens=num_actual_tokens,
                                       block_tables=block_table,
                                       query_cumlens=query_cumlens.tolist(),
                                       seq_lens=seq_lens.tolist(),
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping)
        return attn_metadata


class NPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_dtypes() -> list[torch.dtype]:  # type: ignore[override]
        return [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "VLLM_NPU_ATTN"

    @staticmethod
    def get_impl_cls() -> type["NPUAttentionBackendImpl"]:
        return NPUAttentionBackendImpl

    @staticmethod
    def get_metadata_cls():
        return NPUMetadata

    @staticmethod
    def get_builder_cls():
        return NPUAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: Optional[str] = None,
    ) -> tuple[int, ...]:
        # Use TND layout in decode path
        return (num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, ...]:
        raw_tensor = raw_tensor.view(dtype=dtype)
        shape = NPUAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
        sizes = [math.prod(shape), ] * 2
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape) for t in tensors)


class NPUAttentionBackendImpl(AttentionImpl[NPUMetadata]):
    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(
                alibi_slopes, dtype=torch.float32, device=current_platform.device_type
            )
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"Only support decoder models, but {attn_type=}.")

        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError("self.num_heads must be divisible by self.num_kv_heads")

        if NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE is None:
            NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: NPUMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")

        # View q to TND, kv to TH.
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads*self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads*self.head_size).contiguous()
        batch_descriptor = get_forward_context().batch_descriptor

        # update kv cache
        slots = attn_metadata.slot_mapping.view(-1, 1)
        torch_npu.npu_scatter_nd_update_(kv_cache[0].view(-1, key.shape[-1]), slots, key)
        torch_npu.npu_scatter_nd_update_(kv_cache[1].view(-1, value.shape[-1]), slots, value)

        if batch_descriptor is not None and batch_descriptor.uniform:
            attn_output = torch_npu.npu_fused_infer_attention_score(
                query.unsqueeze(1),
                kv_cache[0],
                kv_cache[1],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSND",
                scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=kv_cache[0].shape[1],
                actual_seq_lengths_kv=attn_metadata.seq_lens,
            )[0].squeeze(1)
        else:
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0],
                kv_cache[1],
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=kv_cache[0].shape[1],
                sparse_mode=3,
                atten_mask=NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_cumlens,
                actual_seq_kvlen=attn_metadata.seq_lens,
            )[0]

        output.copy_(attn_output)
        return output
