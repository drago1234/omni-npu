from typing import List, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig, CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    is_activation_quantization_format,
)

from omni_npu.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import NPUCompressedTensorsW8A8Int8
from omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe import NPUCompressedTensorsW8A8Int8MoEMethod


NPU_COMPRESSED_TENSORS = "npu-compressed-tensors"


@register_quantization_config(NPU_COMPRESSED_TENSORS)
class NPUCompressedTensorsConfig(CompressedTensorsConfig):
    """Config class for NPU

    This class is a general class that parse quantization configs
    that are supported on NPU hardware.
    """

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "NPU hardware dose not support \"get_min_capability\" feature.")

    @staticmethod
    def _is_dynamic_token_w8a8(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value)
        is_token = (weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_8_bits and is_token and weight_quant.symmetric and is_dynamic

    @staticmethod
    def _is_dynamic_token_w4a8_int(
        weight_quant: QuantizationArgs, input_quant: QuantizationArgs
    ) -> bool:
        is_weight_4_bits = weight_quant.num_bits == 4
        is_activation_8_bits = input_quant.num_bits == 8
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        is_token = (weight_strategy and input_quant.strategy == QuantizationStrategy.TOKEN.value)
        is_dynamic = not weight_quant.dynamic and input_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_weight_4_bits and is_activation_8_bits and is_token and weight_quant.symmetric and is_dynamic

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        format: Optional[str] = None,
        layer_name: Optional[str] = None,
    ) -> "CompressedTensorsScheme":
        # use the per-layer format if defined, otherwise, use global format
        format = format if format is not None else self.quant_format

        act_quant_format = is_activation_quantization_format(format)
        if act_quant_format:
            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return NPUCompressedTensorsW8A8Int8(
                    strategy=weight_quant.strategy,
                    is_static_input_scheme=False,
                    input_symmetric=input_quant.symmetric,
                )

            if self._is_dynamic_token_w4a8_int(weight_quant, input_quant):
                raise NotImplementedError

        raise NotImplementedError("No compressed-tensors compatible scheme was found.")

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg['quant_method']
        if torch.npu.is_available() and quant_method == 'compressed-tensors':
            return NPU_COMPRESSED_TENSORS
        return None

    def get_moe_method(self, layer: FusedMoE) -> Optional[QuantizeMethodBase]:
        if "Linear" in self.target_scheme_map:
            matched_target = "Linear"
        else:
            # May have instead defined the linear layers in the fused model

            fused_layers = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
            current_scheme = None
            for fused_layer in fused_layers:
                # Check if one of the fused layers are defined in quant_config
                matched_target = find_matched_target(
                    layer_name=fused_layer,
                    module=layer,
                    targets=self.target_scheme_map.keys(),
                    fused_mapping=self.packed_modules_mapping,
                )

                # Only valid if down_proj, gate_proj, and up_proj
                # are mapped to the same quant scheme in the quant_config
                if current_scheme is None:
                    current_scheme = self.target_scheme_map.get(matched_target)
                else:
                    assert current_scheme == self.target_scheme_map.get(
                        matched_target
                    )

        weight_quant = self.target_scheme_map[matched_target].get("weights")
        input_quant = self.target_scheme_map[matched_target].get(
            "input_activations"
        )

        if self._is_dynamic_token_w8a8(weight_quant, input_quant):
            return NPUCompressedTensorsW8A8Int8MoEMethod(self, layer)
        elif self._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            raise NotImplementedError
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)
            return quant_method
        if isinstance(layer, FusedMoE):
            return self.get_moe_method(layer)
        return None

