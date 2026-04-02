# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    MambaModelConfig,
    VerifyAndUpdateConfig,
)
import vllm.model_executor.models.config as models_config_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class PanguV2MoEForCausalLMConfig(MambaModelConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:

        # set mamba_block_size
        # TODO: align pagesizes, particularly for dsa c8
        super().verify_and_update_config(vllm_config)

@register_patch("PanguV2MoEModelsConfigMapPatch", models_config_module)
class PanguV2MoEModelsConfigMapPatch(VLLMPatch):
    """Add PanguV2MoEForCausalLM to MODELS_CONFIG_MAP."""

    _attr_names_to_apply = ["MODELS_CONFIG_MAP"]
    MODELS_CONFIG_MAP = {
        **MODELS_CONFIG_MAP,
        "PanguV2MoEForCausalLM": PanguV2MoEForCausalLMConfig,
    }
