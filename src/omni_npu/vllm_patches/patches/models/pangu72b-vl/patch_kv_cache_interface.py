from dataclasses import dataclass, fields

from typing_extensions import Self

from vllm.utils.torch_utils import get_dtype_size

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.v1 import kv_cache_interface
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, AttentionSpec

@register_patch("FullAttentionSpecPatch", FullAttentionSpec)
class FullAttentionSpecPatch(VLLMPatch):
    _attr_names_to_apply = ['__post_init__', 'merge', 'head_size_v', 'set_head_size_v', 'page_size_bytes']
    head_size_v: int | None = None

    def set_head_size_v(self, head_size_v : int):
        object.__setattr__(self, "head_size_v", head_size_v)

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
    
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        merged_spec.set_head_size_v(specs[0].head_size_v)
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    @property
    def page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

@register_patch("SinkFullAttentionSpecPatch", kv_cache_interface)
class SinkFullAttentionSpecPatch(VLLMPatch):
    _attr_names_to_apply = ['SinkFullAttentionSpec']

    @dataclass(frozen=True)
    class SinkFullAttentionSpec(FullAttentionSpec):
        sink_len: int | None = None

        @classmethod
        def merge(cls, specs: list[Self]) -> Self:
            """
            Merge a list of FullAttentionSpec objects into a single
            FullAttentionSpec object.
            """
            assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
                "All attention layers in the same KV cache group must be FullAttentionSpec."
            )

            sliding_window = set(
                spec.sliding_window for spec in specs if spec.sliding_window is not None
            )
            attention_chunk_size = set(
                spec.attention_chunk_size
                for spec in specs
                if spec.attention_chunk_size is not None
            )
            assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
                "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
            )
            merged_spec = cls(
                block_size=specs[0].block_size,
                num_kv_heads=specs[0].num_kv_heads,
                head_size=specs[0].head_size,
                sink_len=specs[0].sink_len,
                dtype=specs[0].dtype,
                sliding_window=cls.merge_window_sizes(sliding_window),
                attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            )
            merged_spec.set_head_size_v(specs[0].head_size_v)
            for spec in specs:
                for f in fields(AttentionSpec):
                    assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                        "All attention layers in the same KV cache group must have "
                        "the same attention spec."
                    )
            assert (merged_spec.sliding_window is not None) + (
                merged_spec.attention_chunk_size is not None
            ) <= 1, (
                "Model with both sliding window layers and chunked local attention "
                "layers is not supported."
            )
            return merged_spec
    kv_cache_interface.SinkFullAttentionSpec = SinkFullAttentionSpec
