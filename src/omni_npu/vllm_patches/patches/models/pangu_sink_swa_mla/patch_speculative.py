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


@register_patch("SpeculativePatch", speculative)
class SpeculativePatch(VLLMPatch):
    _attr_names_to_apply = ['MTPModelTypes']

    #####patch start: for openpangu_mtp
    MTPModelTypes = Literal[
        "deepseek_mtp",
        "mimo_mtp",
        "glm4_moe_mtp",
        "ernie_mtp",
        "qwen3_next_mtp",
        "longcat_flash_mtp",
        "mtp",
        "openpangu_mtp",
    ]
    #####patch end


@register_patch("SpeculativeConfigPatch", SpeculativeConfig)
@config
@dataclass
class SpeculativeConfigPatch(VLLMPatch):
    """Patch for vLLM's SpeculativeConfig to support openpangu_v2 and openpangu_mtp model_type.
    """

    _attr_names_to_apply = ['hf_config_override']

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe", "openpangu_v2"):
            hf_config.model_type = "openpangu_mtp"
        if hf_config.model_type == "openpangu_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        return hf_config
