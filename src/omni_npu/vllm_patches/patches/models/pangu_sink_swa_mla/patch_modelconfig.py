from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger
from vllm.config.utils import config
from vllm.config.model import ModelConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("ModelConfigPatch", ModelConfig)
@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfigPatch(VLLMPatch):
    """Patch for vLLM's ModelConfig to support openpangu_v2 and openpangu_mtp model_type.
    """

    _attr_names_to_apply = ['is_deepseek_mla', 'get_total_num_hidden_layers']

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "openpangu_mtp",
            "openpangu_v2",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def get_total_num_hidden_layers(self) -> int:
        if (
            self.hf_text_config.model_type == "deepseek_mtp"
            or self.hf_config.model_type == "mimo_mtp"
            or self.hf_config.model_type == "glm4_moe_mtp"
            or self.hf_config.model_type == "ernie_mtp"
            or self.hf_config.model_type == "qwen3_next_mtp"
            or self.hf_config.model_type == "openpangu_mtp"
        ):
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 0
            )
        elif self.hf_config.model_type == "longcat_flash_mtp":
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 1
            )
        else:
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_hidden_layers", 0
            )
        return total_num_hidden_layers