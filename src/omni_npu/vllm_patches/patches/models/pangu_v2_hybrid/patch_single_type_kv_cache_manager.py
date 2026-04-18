# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Add MomeManager class and update spec_manager_map for Pangu V2 hybrid attention.
"""

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec, SinkFullAttentionSpec
from vllm.v1.core import single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

from .patch_kv_cache_interface import (
    MomeSpec,
    DSAAttentionSpec,
    ShareKVSlidingWindowSpec,
    SinkMLAAttentionSpec
)
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec
)


# MomeManager class
class MomeManager(SingleTypeKVCacheManager):
    """
    Manager for Mome KV cache (similar to Mamba).

    In each block, it always stores representations of k tokens.
    """

    def __init__(
        self,
        kv_cache_spec: MomeSpec,
        block_pool: BlockPool,
        enable_caching: bool = False,
        kv_cache_group_id: int = 0,
        **kwargs
    ) -> None:
        SingleTypeKVCacheManager.__init__(
            self,
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            **kwargs
        )
        self.kernel_size = kv_cache_spec.kernel_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list, ...]:
        """
        For prefix caching, Mome is similar to Mamba. It searches backwards
        until ONE block is hit, and immediately returns that block.
        """

        assert isinstance(kv_cache_spec, MomeSpec), (
            "MomeManager can only be used for mome groups"
        )
        assert dcp_world_size == 1, "DCP not support mome now."
        assert pcp_world_size == 1, "PCP not support mome now."
        computed_blocks: tuple[list, ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Mome computes convolution with respect to previous `kernel_size - 1`
        tokens, so it's very similar to sliding window in KV alloc/free,
        with window_size = kernel_size.
        """
        return max(0, num_computed_tokens - self.kernel_size + 1)

class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        # assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)

class SlidingWindowManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        
        return computed_blocks
    
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0


# Update spec_manager_map and add MomeManager to the module
original_spec_manager_map = dict(single_type_kv_cache_manager.spec_manager_map)
original_spec_manager_map[DSAAttentionSpec] = FullAttentionManager
original_spec_manager_map[ShareKVSlidingWindowSpec] = SlidingWindowManager
original_spec_manager_map[MomeSpec] = MomeManager
original_spec_manager_map[SinkMLAAttentionSpec] = SinkFullAttentionManager


# Create a patch to update spec_manager_map
@register_patch("SingleTypeKVCacheManagerPatch", single_type_kv_cache_manager)
class SingleTypeKVCacheManagerPatch(VLLMPatch):
    """Patch to add MomeManager and update spec_manager_map"""

    _attr_names_to_apply = ["spec_manager_map", "MomeManager"]

    # Patch start
    spec_manager_map: dict = original_spec_manager_map
    MomeManager: type = MomeManager
    # patch end
