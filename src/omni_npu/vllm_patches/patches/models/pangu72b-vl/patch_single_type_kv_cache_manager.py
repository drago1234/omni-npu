from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SinkFullAttentionSpec,
)
from vllm.v1.request import Request
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CrossAttentionManager,
    FullAttentionManager,
    KVCacheSpec,
    MambaManager,
    MLAAttentionSpec,
    SlidingWindowManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.core import single_type_kv_cache_manager

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


#####patch start: for pangu72B-VL
class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec, block_pool, kv_cache_group_id, dcp_world_size
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: list[KVCacheBlock],
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.

        Returns:
            The number of blocks.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = (
            num_required_blocks
            - len(new_computed_blocks)
            - len(self.req_to_blocks[request_id])
        )
        if len(self.req_to_blocks[request_id]) > 0:
            num_new_blocks = num_new_blocks + len(self.sink_blocks)
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it will be changed from a free block
        # to a computed block when the request is allocated, so we also count
        # it as needed to be allocated.
        num_evictable_computed_blocks = sum(
            blk.ref_cnt == 0 and not blk.is_null for blk in new_computed_blocks
        )
        return num_new_blocks + num_evictable_computed_blocks

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: list[KVCacheBlock]
    ) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
        """
        if request_id not in self.num_cached_block:
            # A new request.
            req_blocks = self.req_to_blocks[request_id]
            assert len(req_blocks) == 0
            req_blocks.extend(self.sink_blocks)
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:
            # A running request. Should not have new computed blocks.
            assert len(new_computed_blocks) == 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if len(req_blocks) > 0:
            num_new_blocks = num_new_blocks + len(self.sink_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            if len(req_blocks) == 0:
                req_blocks.extend(self.sink_blocks + new_blocks)
            else:
                req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id][len(self.sink_blocks) :],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        if len(req_blocks) > 0:
            req_blocks = req_blocks[len(self.sink_blocks) :]

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)
#####patch end



@register_patch("single_type_kv_cache_managerPatch", single_type_kv_cache_manager)
class single_type_kv_cache_managerPatch(VLLMPatch):
    _attr_names_to_apply = ['spec_manager_map']


    #####patch start: for pangu72B-VL
    spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
        FullAttentionSpec: FullAttentionManager,
        MLAAttentionSpec: FullAttentionManager,
        SlidingWindowSpec: SlidingWindowManager,
        ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
        MambaSpec: MambaManager,
        CrossAttentionSpec: CrossAttentionManager,
        SinkFullAttentionSpec: SinkFullAttentionManager,
    }
    #####patch end
