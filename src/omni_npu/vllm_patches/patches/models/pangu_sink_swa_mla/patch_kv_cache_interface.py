from dataclasses import dataclass

from typing_extensions import Self

from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("SinkMLAAttentionSpecPatch", kv_cache_interface)
class SinkMLAAttentionSpecPatch(VLLMPatch):
    _attr_names_to_apply = ['SinkMLAAttentionSpec']

    # patch start
    @dataclass(frozen=True)
    class SinkMLAAttentionSpec(MLAAttentionSpec):
        sink_len: int = 0
        @classmethod
        def merge(cls, specs: list[Self]) -> Self:
            assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
                "All attention layers in the same KV cache group must be MLAAttentionSpec."
            )
            cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
            assert len(cache_dtype_str_set) == 1, (
                "All attention layers in the same KV cache group must use the same "
                "quantization method."
            )
            return cls(
                block_size=specs[0].block_size,
                num_kv_heads=specs[0].num_kv_heads,
                head_size=specs[0].head_size,
                dtype=specs[0].dtype,
                cache_dtype_str=cache_dtype_str_set.pop(),
                sink_len=specs[0].sink_len,
            )
    # patch end
    
    kv_cache_interface.SinkMLAAttentionSpec = SinkMLAAttentionSpec