from vllm.v1.core import single_type_kv_cache_manager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CrossAttentionManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkMLAAttentionSpec,
    SlidingWindowSpec,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


# patch start
class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkMLAAttentionSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)
# patch end


@register_patch("single_type_kv_cache_managerPatch", single_type_kv_cache_manager)
class single_type_kv_cache_managerPatch(VLLMPatch):
    _attr_names_to_apply = ['spec_manager_map']

    # patch start
    spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
        FullAttentionSpec: FullAttentionManager,
        MLAAttentionSpec: FullAttentionManager,
        SlidingWindowSpec: SlidingWindowManager,
        ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
        MambaSpec: MambaManager,
        CrossAttentionSpec: CrossAttentionManager,
        SinkMLAAttentionSpec: SinkFullAttentionManager,
    }
    # patch end