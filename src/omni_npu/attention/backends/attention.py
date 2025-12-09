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
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
import torch_npu

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.config import get_current_vllm_config, CompilationLevel
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform
from vllm.utils import (
    direct_register_custom_op,
    is_pin_memory_available,
    supports_dynamo,
)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder as V1AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


NZ_DIM = 16


class NPUAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3


def unified_npu_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    self_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = self_layer.kv_cache[forward_context.virtual_engine]
    self_layer.impl.forward(  # type: ignore[attr-defined]
        self_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        trace_flag=False,
    )
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_npu_attention_with_output",
    op_func=unified_npu_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)


@dataclass
class NPUMetadata:
    num_actual_tokens: int
    block_tables: torch.Tensor
    query_cumlens: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: Optional[int] = None
    slot_mapping: torch.Tensor = None
    slot_indices: torch.Tensor = None


class NPUAttentionMetadataBuilder(V1AttentionMetadataBuilder[NPUMetadata]):
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
        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)
        attn_metadata = NPUMetadata(num_actual_tokens=num_actual_tokens,
                                       block_tables=block_table,
                                       query_cumlens=query_cumlens,
                                       seq_lens=seq_lens,
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       slot_indices=slot_indices)
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

        cur_vllm_config = get_current_vllm_config()
        try:
            level = getattr(getattr(cur_vllm_config, "npu_compilation_config", None), "level", None)
            self.enable_graph_mode = bool(
                level is not None and level > getattr(CompilationLevel, "NO_COMPILATION", 0)
            ) and supports_dynamo()
        except Exception:
            self.enable_graph_mode = False

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
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")

        # View q k v to TND.
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()

        # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            block_size = kv_cache[0].shape[-2]
            assert block_size == 128, f"{block_size}"
            cast_key = key.reshape(-1, self.num_kv_heads * self.head_size)
            cast_value = value.reshape(-1, self.num_kv_heads * self.head_size)
            slots = attn_metadata.slot_indices
            torch_npu.npu_scatter_nd_update_(kv_cache[0], slots, cast_key)
            torch_npu.npu_scatter_nd_update_(kv_cache[1], slots, cast_value)

        attn_output = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            kv_cache[0],
            kv_cache[1],
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            softmax_scale=self.scale,
            block_table=attn_metadata.block_tables,
            block_size=block_size,
            sparse_mode=3,
            atten_mask=NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
            actual_seq_qlen=attn_metadata.query_cumlens,
            actual_seq_kvlen=attn_metadata.seq_lens,
        )[0]

        output = output.view_as(attn_output)
        output.copy_(attn_output)

        return output.view(num_tokens, self.hidden_size)
