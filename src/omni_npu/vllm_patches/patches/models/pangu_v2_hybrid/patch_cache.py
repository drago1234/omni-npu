from typing import Literal

from vllm.config import cache

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("CachePatch", cache)
class CachePatch(VLLMPatch):
    _attr_names_to_apply = ['BlockSize']

    BlockSize = Literal[1, 8, 16, 32, 48, 64, 128, 256]