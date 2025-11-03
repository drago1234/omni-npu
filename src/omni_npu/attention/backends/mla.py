"""
Minimal, self-contained Ascend MLA attention backend for omni_npu.

This implementation currently delegates to the standard Ascend attention
backend to remain fully self-contained and avoid external dependencies.
It satisfies vLLM's backend interface so the platform selector can
import and use it. We can iterate later with true MLA specialization.
"""

from typing import List, Tuple, Type
import torch

from vllm.attention.backends.abstract import AttentionBackend

from .attention import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)


class AscendMLABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "VLLM_ASCEND_MLA"

    @staticmethod
    def get_metadata_cls() -> type["AscendMetadata"]:
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
        cache_dtype_str: str | None = None,
    ) -> tuple[int, ...]:
        # Match the standard backend's cache shape for now
        return (num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        # Delegate to the standard Ascend attention implementation
        return AscendAttentionBackendImpl
