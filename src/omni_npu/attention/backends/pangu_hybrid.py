from dataclasses import dataclass
import copy

import torch

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu.attention.backends.utils import register_attention_backend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.dsa import NPUDSABackend
from omni_npu.attention.backends.attention import NPUAttentionBackendImpl


logger = init_logger(__name__)
NPUPanguMLA = "NPUPanguMLA"
NPUPanguDSA = "NPUPanguDSA"
NPUPanguMome = "NPUPanguMome"


def _maybe_padded_raw_tensor_to_strided_caches(
    raw_tensor: torch.Tensor,
    num_blocks: int,
    block_size: int,
    shapes: tuple[tuple[int, ...], ...],
    dtypes: tuple[torch.dtype, ...],
    page_size_bytes: int,
) -> tuple[torch.Tensor, ...]:
    """
    Creates strided views of a raw memory tensor to represent heterogeneous,
    padded KV cache blocks.

    This function maps a flat 1D raw tensor into multiple multi-dimensional
    cache tensors. It assumes the raw tensor is partitioned into `num_blocks`
    pages of `page_size_bytes`. Within each page, the sub-tensors defined by
    `shapes` and `dtypes` are packed sequentially, followed by potential padding
    to fill the rest of the page.

    Memory Layout per Page (Block):
    [ Tensor 0 ] [ Tensor 1 ] ... [ Tensor N ] [ Padding (optional) ]
    |<- bytes ->|<- bytes ->|    |<- bytes ->|
    |<------------------- page_size_bytes ------------------------->|

    Args:
        raw_tensor (torch.Tensor): The underlying 1D memory pool.
        num_blocks (int): The total number of memory blocks (pages) allocated.
        block_size (int): The number of tokens each block can hold.
        shapes (tuple[tuple[int, ...], ...]): A tuple containing the trailing
            dimensions for each sub-tensor (excluding num_blocks and block_size).
        dtypes (tuple[torch.dtype, ...]): The data types corresponding to each shape.
        page_size_bytes (int): The fixed size of each memory block in bytes.

    Returns:
        cache_tensors (tuple[torch.Tensor, ...]): A tuple of strided tensors sharing the same
            underlying storage as `raw_tensor`. Each tensor has the shape:
            (num_blocks, block_size, *shape).

    Raises:
        AssertionError: If shapes and dtypes lengths mismatch, if type sizes don't
            align, if the raw tensor is too small, or if the page size is insufficient.
    """
    assert len(shapes) == len(dtypes), f"Error! {len(shapes)=} while {len(dtypes)=}."

    # Ensure the raw tensor has enough physical memory
    total_required_bytes = num_blocks * page_size_bytes
    actual_bytes = raw_tensor.numel() * raw_tensor.element_size()
    assert actual_bytes >= total_required_bytes, (
        f"Error! Raw tensor has {actual_bytes} bytes, "
        f"but {total_required_bytes} bytes are required."
    )

    cache_tensors = []
    storage_offset_bytes = 0

    for shape, dtype in zip(shapes, dtypes):
        dtype_size = dtype.itemsize
        assert page_size_bytes % dtype_size == 0, (
            f"Error! Page size {page_size_bytes} is not a multiple of dtype size {dtype_size}."
        )
        assert storage_offset_bytes % dtype_size == 0, (
            f"Error! Offset {storage_offset_bytes} is not aligned for dtype size {dtype_size}."
        )

        num_element_per_page = page_size_bytes // dtype_size
        target_shape = (num_blocks, block_size, *shape)

        # Get contiguous strides. stride[0] will equal the total number
        # of elements this specific tensor occupies within a single block.
        stride = torch.empty(target_shape).stride()

        # Override the 0th stride to jump by the full page size
        target_stride = (num_element_per_page, *stride[1:])

        tensor = torch.as_strided(
            raw_tensor.view(dtype),
            size=target_shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        cache_tensors.append(tensor)

        # Advance the byte offset by the size this tensor takes up in one block
        storage_offset_bytes += stride[0] * dtype_size

    # The crucial missing check: Did we exceed the allocated page size?
    assert storage_offset_bytes <= page_size_bytes, (
        f"Error! Sub-tensors require {storage_offset_bytes} bytes per block, "
        f"which exceeds the allocated page_size_bytes of {page_size_bytes}."
    )

    return tuple(cache_tensors)


@register_attention_backend(NPUPanguMLA)
class NPUPanguMLABackend(NPUMLABackend):
    @staticmethod
    def get_name() -> str:
        return NPUPanguMLA

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: "ShareKVSlidingWindowSpec",
    ) -> tuple[torch.Tensor, ...]:
        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.block_size,
            shapes=( (512,), (64,) ),
            dtypes=(kv_cache_spec.dtype,) * 2,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )


@register_attention_backend(NPUPanguDSA)
class NPUPanguDSABackend(NPUDSABackend):
    @staticmethod
    def get_name() -> str:
        return NPUPanguDSA

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: "DSAAttentionSpec",
    ) -> tuple[torch.Tensor, ...]:
        if kv_cache_spec.cache_dtype_str == "fp8_ds_mla":
            shapes = ( (656,), (132,) )
            dtypes = (torch.fp8, torch.int8)  # TODO: replace fp8 with float8_e4m3fn or float8_e5m2
        elif kv_cache_spec.cache_dtype_str == "int8_ds_mla":
            shapes = ( (656,), (130,) )
            dtypes = (torch.int8, torch.int8)
        else:
            shapes = ( (576,), (128,) )
            dtypes = (torch.bfloat16, torch.bfloat16)

        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.block_size,
            shapes=shapes,
            dtypes=dtypes,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )


@register_attention_backend(NPUPanguMome)
class NPUPanguMomeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return NPUPanguMome

    @staticmethod
    def get_builder_cls() -> type["NPUMomeAttentionMetadataBuilder"]:
        return NPUMomeAttentionMetadataBuilder

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: "MomeSpec",
    ) -> tuple[torch.Tensor, ...]:
        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.num_total_tokens,
            shapes=kv_cache_spec.shapes,
            dtypes=kv_cache_spec.dtypes,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )

    @staticmethod
    def get_impl_cls() -> type["NPUAttentionBackendImpl"]:
        return NPUAttentionBackendImpl


@dataclass
class NPUMomeAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # Query and cache management fields
    query_start_loc: torch.Tensor  # shape: [batch + 1,]
    cache_indices: torch.Tensor  # shape: [batch,] or [batch, max_num_blocks]
    max_query_len: int = 1
    pad_slot_id: int = PAD_SLOT_ID
    B_size: int = 1  # Mome block size (kv_cache_spec.block_size)

    # Speculative decoding support
    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # State and token computed tracking
    num_computed_tokens: torch.Tensor | None = None  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor | None = None  # shape: [batch,]
    block_idx_first_scheduled_token: torch.Tensor | None = None  # shape: [batch,]
    block_idx_last_scheduled_token: torch.Tensor | None = None  # shape: [batch,]


class NPUMomeAttentionMetadataBuilder(AttentionMetadataBuilder[NPUMomeAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        from vllm.v1.kv_cache_interface import MomeSpec
        assert isinstance(kv_cache_spec, MomeSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.mome_block_size = kv_cache_spec.block_size
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # Allocate buffers for prefix caching support
        if self.vllm_config.cache_config.enable_prefix_caching:
            max_num_blocks = cdiv(
                self.vllm_config.model_config.max_model_len, self.mome_block_size
            )
            self.cache_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_first_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.cache_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_first_scheduled_token = None
            self.block_idx_last_scheduled_token = None

        # Buffers for cudagraph
        self.num_computed_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mome_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Borrowed from `BaseMambaAttentionMetadataBuilder`. Completely same.
        """
        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # Block index of the last computed token
        block_idx_last_computed_token = cdiv(num_computed_tokens, mome_block_size) - 1
        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = (
            cdiv(num_computed_tokens + 1, mome_block_size) - 1
        )
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = (
            cdiv(common_attn_metadata.seq_lens, mome_block_size) - 1
        )
        # -1 in case it's non-computed and causes later issues with indexing
        block_idx_last_computed_token = torch.clamp(
            block_idx_last_computed_token, min=0
        )
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = torch.clamp(
            block_idx_last_scheduled_token, min=0
        )

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> NPUMomeAttentionMetadata:
        """
        Build Mome attention metadata from common attention metadata.
        """
        num_reqs = common_attn_metadata.num_reqs

        # Split decodes and prefills
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=self.reorder_batch_threshold
            )
        )

        num_computed_tokens = None
        block_idx_last_computed_token = None
        block_idx_first_scheduled_token = None
        block_idx_last_scheduled_token = None

        # Get cache indices
        if self.vllm_config.cache_config.enable_prefix_caching:
            cache_indices = common_attn_metadata.block_table_tensor
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()

            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, self.mome_block_size
            )
        else:
            cache_indices = common_attn_metadata.block_table_tensor[:, 0]

        if num_accepted_tokens is not None:
            self.num_accepted_tokens[:num_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:num_decodes]

        # For cudagraph: copy to persistent buffer if applicable
        if (
            num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.cache_indices_tensor[:num_decodes].copy_(cache_indices, non_blocking=True)
            cache_indices = self.cache_indices_tensor[:num_decodes]  # NOTE: slice to num_decodes instead of num_decode_tokens
            self.cache_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=True
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

                self.block_idx_first_scheduled_token[:num_decodes].copy_(
                    block_idx_first_scheduled_token, non_blocking=True
                )
                block_idx_first_scheduled_token = self.block_idx_first_scheduled_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

                self.num_computed_tokens[:num_decodes].copy_(
                    num_computed_tokens, non_blocking=True
                )
                num_computed_tokens = self.num_computed_tokens[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

        max_query_len = common_attn_metadata.max_query_len

        attn_metadata = NPUMomeAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            cache_indices=cache_indices,
            max_query_len=max_query_len,
            pad_slot_id=PAD_SLOT_ID,
            B_size=self.mome_block_size,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            num_reqs=num_reqs,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"Mome only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)

    def update_block_table(
        self,
        metadata: NPUMomeAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> NPUMomeAttentionMetadata:
        new_metadata = copy.copy(metadata)
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        cache_indices = blk_table if prefix_caching else blk_table[:, 0]
        num_reqs = blk_table.shape[0]

        # For CUDA graphs, copy to persistent buffer
        if (
            metadata.num_prefills == 0
            and num_reqs <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            persistent_cache_indices = self.cache_indices_tensor[:num_reqs]
            persistent_cache_indices.copy_(cache_indices, non_blocking=True)
            cache_indices = persistent_cache_indices

        new_metadata.cache_indices = cache_indices
        return new_metadata
