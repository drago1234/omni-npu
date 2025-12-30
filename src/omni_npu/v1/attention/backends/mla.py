# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
import math

import torch

from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonBaseImpl,
    MLACommonPrefillMetadata
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    CommonAttentionMetadata
)
from vllm.v1.kv_cache_interface import AttentionSpec


logger = init_logger(__name__)


class NPUMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "NPUMLA"

    @staticmethod
    def get_metadata_cls() -> type["NPUMLAMetadata"]:
        return NPUMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["NPUMLAMetadataBuilder"]:
        return NPUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUMLAImpl"]:
        return NPUMLAImpl

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
        shapes = [(num_blocks, block_size, 1, 512), (num_blocks, block_size, 1, 64)]
        sizes = [math.prod(shape) for shape in shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape) for t, shape in zip(tensors, shapes))


@dataclass
class NPUMLAPrefillMetadata(MLACommonPrefillMetadata):
    query_cumlens: torch.Tensor = None
    seq_lens: torch.Tensor = None


@dataclass
class NPUMLADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor


@dataclass
class NPUMLAMetadata(MLACommonMetadata[NPUMLADecodeMetadata]):
    pass


class NPUMLAMetadataBuilder(MLACommonMetadataBuilder[NPUMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_uniform_spec_as_decode: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, NPUMLAMetadata
        )
        self.prefill_metadata_cls = NPUMLAPrefillMetadata
        if self._use_fi_prefill:
            raise ValueError("Flashinfer should not be enabled.")
        if self._use_cudnn_prefill:
            raise ValueError("CUDNN should not be enabled.")
        if self.dcp_world_size > 1:
            raise ValueError("DCP should not be enabled.")
        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> NPUMLADecodeMetadata:
        return NPUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_cumlens=query_start_loc_device[1:],
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> NPUMLAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.prefill is not None:
            metadata.prefill.query_cumlens = metadata.prefill.query_start_loc[1:] - metadata.prefill.query_start_loc[:-1]
            metadata.prefill.seq_lens = metadata.prefill.query_cumlens
            metadata.prefill.query_start_loc = metadata.prefill.query_start_loc.tolist()
        if metadata.prefill is not None and metadata.prefill.chunked_context is not None:
            raise RuntimeError(f"Chunked prefill is not enabled yet.")
        return metadata


class NPUMLAImpl(MLACommonBaseImpl[NPUMLAMetadata]):
    can_return_lse_for_decode: bool = False
    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        self.chunked_prefill_workspace_size = (
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                get_current_vllm_config()
            )
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "NPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "NPUMLAImpl"
            )

        if NPUMLAImpl.SHARE_MASK_TRIL_SPARSE is None:
            NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )
            NPUMLAImpl.DECORE_ATTN_MASK = NPUMLAImpl.SHARE_MASK_TRIL_SPARSE.to(torch.uint8)

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass
