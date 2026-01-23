import torch
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.model_executor.models.openpangu_mtp import  OpenPanguMTP

@register_patch("OpenPanguMTPPatch", OpenPanguMTP)
class OpenPanguMTPPatch(VLLMPatch):
    _attr_names_to_apply = ['embed_input_ids']

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)