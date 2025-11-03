"""
Ascend Attention backend vendored into omni_npu.attention.backends.

This is a minimized, self-contained version derived from omniinfer sources,
with the following adjustments:
- Replaced omni.* imports with local shims or safe defaults.
- Kept the core torch_npu kernels usage for prefill and decode.
- Removed optional features (best_ep, omni_cache, DSA, SP) for an MVP.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3


def unified_ascend_attention_with_output(
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
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)


@dataclass
class AscendMetadata:
    num_actual_tokens: int
    block_tables: torch.Tensor
    query_lens: torch.Tensor
    query_lens_list: List
    seq_lens: torch.Tensor
    seq_lens_list: List
    max_query_len: Optional[int] = None
    slot_mapping: torch.Tensor = None
    slot_indices: torch.Tensor = None
    is_only_prefill: bool = False
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    is_pd_seperate_d: bool = False
    kv_index: Optional[torch.Tensor] = None

    @staticmethod
    def advance_step(metadata, positions, block_size, pad_mask, model_layer):
        block_table = metadata.block_tables
        block_indices = block_table.gather(
            dim=1, index=(positions // block_size).reshape(-1, 1)
        ).view(-1)
        block_offsets = positions % block_size
        metadata.slot_mapping[:] = torch.where(
            pad_mask,
            metadata.slot_mapping,
            block_indices * block_size + block_offsets,
        )
        metadata.seq_lens[:] = (positions + 1).to(metadata.seq_lens.dtype)


class AscendAttentionMetadataBuilder(V1AttentionMetadataBuilder[AscendMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        # Derive per-request query lengths from query_start_loc
        qsl = common_attn_metadata.query_start_loc_cpu
        query_lens_cpu = (qsl[1:] - qsl[:-1]).to(torch.long)

        seq_lens = common_attn_metadata.seq_lens
        block_tables = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping.to(self.device, non_blocking=True)
        slot_indices = torch.stack(
            [slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1
        )

        # Heuristic attn state based on max query len
        attn_state = (
            AscendAttentionState.DecodeOnly
            if common_attn_metadata.max_query_len == 1
            else AscendAttentionState.PrefillNoCache
        )

        return AscendMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            block_tables=block_tables,
            query_lens=query_lens_cpu.to(self.device, non_blocking=True),
            query_lens_list=query_lens_cpu.tolist(),
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.to("cpu", non_blocking=True).tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            slot_mapping=slot_mapping,
            slot_indices=slot_indices,
            attn_state=attn_state,
            cos=None,
            sin=None,
            is_pd_seperate_d=False,
            kv_index=None,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> AscendMetadata:
        return self.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)

    def mark_static_for_attn_metadata(self, attn_metadata: AscendMetadata):
        if attn_metadata is not None:
            torch._dynamo.mark_static(attn_metadata.block_tables)
            torch._dynamo.mark_static(attn_metadata.query_lens)
            torch._dynamo.mark_static(attn_metadata.seq_lens)
            if attn_metadata.slot_mapping is not None:
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
            if attn_metadata.slot_indices is not None:
                torch._dynamo.mark_static(attn_metadata.slot_indices)


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_dtypes() -> list[torch.dtype]:  # type: ignore[override]
        return [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "VLLM_ASCEND_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls():
        return AscendMetadata

    @staticmethod
    def get_builder_cls():
        return AscendAttentionMetadataBuilder

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


class AscendAttentionBackendImpl(AttentionImpl[AscendMetadata]):
    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
        kv_stream=None,
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

        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError("self.num_heads must be divisible by self.num_kv_heads")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

        cur_vllm_config = get_current_vllm_config()
        try:
            level = getattr(getattr(cur_vllm_config, "npu_compilation_config", None), "level", None)
            self.enable_graph_mode = bool(
                level is not None and level > getattr(CompilationLevel, "NO_COMPILATION", 0)
            ) and supports_dynamo()
        except Exception:
            self.enable_graph_mode = False

        if AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE is None:
            AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )
        self.kv_stream = kv_stream

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(
                num_tokens,
                self.num_heads,
                self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError(
                "layer._k_scale_float and layer._v_scale_float must both be 1.0"
            )
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            # Graceful bypass for encoder/cross-attention: passthrough output
            return output.view(num_tokens, self.hidden_size)

        # View q k v
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()

        # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            block_size = self.key_cache.shape[1]

            cast_key = key.reshape(-1, 1, self.num_kv_heads * self.head_size)
            cast_value = value.reshape(-1, 1, self.num_kv_heads * self.head_size)

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                stream_for_reshape_and_cache = (
                    self.kv_stream if self.kv_stream is not None else torch.npu.current_stream()
                )
                with torch.npu.stream(stream_for_reshape_and_cache):
                    torch_npu._npu_reshape_and_cache(
                        key,
                        value,
                        self.key_cache.view(
                            self.key_cache.shape[0], block_size, self.num_kv_heads, self.head_size
                        ),
                        self.value_cache.view(
                            self.value_cache.shape[0], block_size, self.num_kv_heads, self.head_size
                        ),
                        attn_metadata.slot_mapping.int(),
                    )
            else:
                torch_npu.scatter_update_(self.key_cache, attn_metadata.slot_indices, cast_key, -2)
                torch_npu.scatter_update_(self.value_cache, attn_metadata.slot_indices, cast_value, -2)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            if attn_metadata is None:
                raise RuntimeError("attn_metadata must not be None")
            if len(attn_metadata.query_lens_list) == 1:
                attn_output = torch_npu.npu_fused_infer_attention_score(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BSND",
                    scale=self.scale,
                    sparse_mode=3,
                    actual_seq_lengths=attn_metadata.query_lens_list,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                )[0].view(-1, self.num_heads, self.head_size)
                output = output.view_as(attn_output)
                output.copy_(attn_output)
            else:
                actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
                actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()
                attn_output = torch_npu.npu_fusion_attention(
                    query[: actual_seq_qlen[-1], :],
                    key[: actual_seq_qlen[-1], :],
                    value[: actual_seq_qlen[-1], :],
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                    sparse_mode=3,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen,
                )[0]
                output[: actual_seq_qlen[-1], :].copy_(attn_output)

        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            block_num, block_size = self.key_cache.shape[0], self.key_cache.shape[1]  # noqa: F841
            num_batch = attn_metadata.seq_lens.shape[0]
            query = query.view(num_batch, -1, self.num_heads * self.head_size)

            # Use TND kernel for decode
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                self.key_cache.view(
                    -1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM
                ),
                self.value_cache.view(
                    -1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM
                ),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=attn_metadata.seq_lens,
            )[0]

            output = output.view_as(attn_output)
            output.copy_(attn_output)

        return output.view(num_tokens, self.hidden_size)
