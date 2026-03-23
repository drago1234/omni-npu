# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Literal

from pydantic.dataclasses import dataclass

from vllm.logger import init_logger
from vllm.config.utils import config
from vllm.config.speculative import SpeculativeConfig
from vllm.config import speculative
from vllm.utils.import_utils import LazyLoader
if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


_origin_hf_config_override = SpeculativeConfig.hf_config_override

@register_patch("PanguV2MoeSpeculativeConfigPatch", SpeculativeConfig)
@config
@dataclass
class PanguV2MoeSpeculativeConfigPatch(VLLMPatch):
    """Patch for vLLM's SpeculativeConfig to support pangu_v2_moe and pangu_v2_moe_mtp model_type.
    """

    _attr_names_to_apply = ['hf_config_override']

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:

        #####patch start: for pangu_v2_moe_mtp
        if hf_config.model_type == "pangu_v2_moe":
            hf_config.model_type = "mtp"
        if hf_config.model_type == "mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["PanguV2MTPModel"]}
            )
            return hf_config
        #####patch end

        return _origin_hf_config_override(hf_config)
