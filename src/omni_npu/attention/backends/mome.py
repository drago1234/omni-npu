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
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

from omni_npu.attention.backends.utils import register_attention_backend, _maybe_padded_raw_tensor_to_strided_caches
from omni_npu.attention.backends.attention import NPUAttentionBackendImpl


logger = init_logger(__name__)
NPUPanguMome = "NPUPanguMome"


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


class NPUMomeAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
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

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
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
        # TODO: fix block_idx_last_computed_token when APC enabled
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
        num_prompt_tokens: torch.Tensor | None = None,
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

        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        block_idx_last_computed_token = None
        block_idx_first_scheduled_token = None
        block_idx_last_scheduled_token = None

        # For graph capture, num_prompt_tokens is None
        if num_accepted_tokens is not None and num_prompt_tokens is not None:
            num_accepted_tokens = num_accepted_tokens.clone()
            # if the previous schedule is prefill (computed <= prompt), we reset num_accepted_tokens = num_spec + 1
            # so the MoME kernel reads the last few tokens from the cache
            num_accepted_tokens.masked_fill_(num_computed_tokens <= num_prompt_tokens, self.num_spec + 1)

        # Get cache indices
        apc_enabled = self.vllm_config.cache_config.enable_prefix_caching
        if apc_enabled:
            cache_indices = common_attn_metadata.block_table_tensor

            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata, self.mome_block_size
            )
        else:
            cache_indices = common_attn_metadata.block_table_tensor[:, 0]

        seq_lens = torch.diff(common_attn_metadata.query_start_loc, dim=-1)
        idx = torch.nonzero(seq_lens == 0)
        cache_indices[idx] = PAD_SLOT_ID

        # For cudagraph: copy to persistent buffer if applicable
        if (
            num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.cache_indices_tensor[:num_decodes].copy_(
                cache_indices, non_blocking=True
            )
            cache_indices = self.cache_indices_tensor[:num_decodes]  # NOTE: slice to num_decodes instead of num_decode_tokens
            self.cache_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.num_computed_tokens[:num_decodes].copy_(
                num_computed_tokens, non_blocking=True
            )
            num_computed_tokens = self.num_computed_tokens[
                :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
            ]

            if num_accepted_tokens is not None:
                self.num_accepted_tokens[:num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.num_accepted_tokens[:num_decodes]

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

        if num_prefills == 0:
            attn_metadata.prefill = None
        else:
            prefill_query_start_loc = common_attn_metadata.query_start_loc[num_decodes:] \
                - common_attn_metadata.query_start_loc[num_decodes]
            attn_metadata.prefill = NPUMomeAttentionMetadata(
                num_prefills=num_prefills, 
                num_prefill_tokens=num_prefill_tokens, 
                num_decodes=0, 
                num_decode_tokens=0, 
                num_reqs=num_prefills, 
                query_start_loc=prefill_query_start_loc, 
                cache_indices=cache_indices[num_decodes:], 
                max_query_len=max_query_len, 
                pad_slot_id=PAD_SLOT_ID, 
                B_size=self.mome_block_size, 
                num_accepted_tokens=None, 
                num_computed_tokens=num_computed_tokens[num_decodes:],
                block_idx_last_computed_token=block_idx_last_computed_token[num_decodes:] if apc_enabled else None, 
                block_idx_first_scheduled_token=block_idx_first_scheduled_token[num_decodes:] if apc_enabled else None, 
                block_idx_last_scheduled_token=block_idx_last_scheduled_token[num_decodes:] if apc_enabled else None, 
            )
        if num_decodes == 0:
            attn_metadata.decode = None
        else:
            attn_metadata.decode = NPUMomeAttentionMetadata(
                num_prefills=0, 
                num_prefill_tokens=0, 
                num_decodes=num_decodes, 
                num_decode_tokens=num_decode_tokens, 
                num_reqs=num_decodes, 
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes+1], 
                cache_indices=cache_indices[:num_decodes], 
                max_query_len=max_query_len, 
                pad_slot_id=PAD_SLOT_ID, 
                B_size=self.mome_block_size, 
                num_accepted_tokens=num_accepted_tokens[:num_decodes] if num_accepted_tokens is not None else None, 
                num_computed_tokens=num_computed_tokens[:num_decodes],
                block_idx_last_computed_token=block_idx_last_computed_token[:num_decodes] if apc_enabled else None, 
                block_idx_first_scheduled_token=block_idx_first_scheduled_token[:num_decodes] if apc_enabled else None, 
                block_idx_last_scheduled_token=block_idx_last_scheduled_token[:num_decodes] if apc_enabled else None, 
            )

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        num_accepted_tokens: torch.Tensor | None = None,
        num_prompt_tokens: torch.Tensor | None = None,
    ) -> NPUMomeAttentionMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            fast_build=True,
            num_prompt_tokens=num_prompt_tokens,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = None if self.num_spec == 0 else torch.diff(m.query_start_loc)
        return self.build(0, m, num_accepted_tokens=num_accepted_tokens)

    def update_block_table(
        self,
        metadata: NPUMomeAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> NPUMomeAttentionMetadata:
        new_metadata = copy.copy(metadata)
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        cache_indices = blk_table if prefix_caching else blk_table[:, 0]
        num_accepted_tokens = metadata.num_accepted_tokens
        num_computed_tokens = metadata.num_computed_tokens
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

            if num_computed_tokens is not None:
                persistent_num_computed_tokens = self.num_computed_tokens[:num_reqs]
                persistent_num_computed_tokens.copy_(num_computed_tokens, non_blocking=True)
                num_computed_tokens = persistent_num_computed_tokens

            if num_accepted_tokens is not None:
                persistent_num_accepted_tokens = self.num_accepted_tokens[:num_reqs]
                persistent_num_accepted_tokens.copy_(num_accepted_tokens, non_blocking=True)
                num_accepted_tokens = persistent_num_accepted_tokens

        new_metadata.cache_indices = cache_indices
        new_metadata.num_accepted_tokens = num_accepted_tokens
        new_metadata.num_computed_tokens = num_computed_tokens

        if new_metadata.prefill is not None:
            new_metadata.prefill = copy.copy(metadata.prefill)
            new_metadata.prefill.cache_indices = cache_indices[metadata.num_decodes:]
            if num_computed_tokens is not None:
                new_metadata.prefill.num_computed_tokens = num_computed_tokens[metadata.num_decodes:]
            if num_accepted_tokens is not None:
                new_metadata.prefill.num_accepted_tokens = num_accepted_tokens[metadata.num_decodes:]
        if new_metadata.decode is not None:
            new_metadata.decode = copy.copy(metadata.decode)
            new_metadata.decode.cache_indices = cache_indices[:metadata.num_decodes]
            if num_computed_tokens is not None:
                new_metadata.decode.num_computed_tokens = num_computed_tokens[:metadata.num_decodes]
            if num_accepted_tokens is not None:
                new_metadata.decode.num_accepted_tokens = num_accepted_tokens[:metadata.num_decodes]

        return new_metadata
