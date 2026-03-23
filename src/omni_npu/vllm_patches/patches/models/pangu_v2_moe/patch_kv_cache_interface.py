# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@register_patch("PanguV2MoeMLAAttentionSpecPatch", MLAAttentionSpec)
class PanguV2MoeMLAAttentionSpecPatch(VLLMPatch):
    """
    Patch for MLAAttentionSpec to add head_size consistency check.
    Required for pangu_v2_moe with different attention layers.
    """

    _attr_names_to_apply = ["merge"]

    ##### patch start: for pangu_v2_moe head_size check
    @classmethod
    def merge(cls, specs: list[MLAAttentionSpec]) -> MLAAttentionSpec:
        """
        Merge a list of MLAAttentionSpec objects into a single MLAAttentionSpec object.
        Added head_size consistency check for pangu_v2_moe compatibility.
        """

        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "quantization method."
        )
        head_size_set = set(spec.head_size for spec in specs)
        assert len(head_size_set) == 1, (
            "All attention layers in the same KV cache group must use the same "
            "head size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            cache_dtype_str=cache_dtype_str_set.pop(),
        )
    ##### patch end