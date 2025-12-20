# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch is used for enable_eplb fix in ParallelConfig and FusedMoE
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" OMNI_NPU_VLLM_PATCHES="EPLBParallelConfig,EPLBFusedMoE" before vllm serve
from vllm.config.parallel import ParallelConfig
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

import os
import torch
from typing import Callable, TYPE_CHECKING, Any, Literal, Optional, Union
from typing_extensions import Self
from pydantic import model_validator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.distributed import get_tensor_model_parallel_world_size, get_dp_group
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEMethodBase, UnquantizedFusedMoEMethod, maybe_roundup_hidden_size, determine_expert_map, get_compressed_expert_map
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]

@config
@dataclass
@register_patch("EPLBParallelConfig", ParallelConfig)
class ParallelConfigPatch(VLLMPatch):
    """
    ParallelConfig for EPLB on NPU devices (when eplb enabled).
    """

    _attr_names_to_apply = ['_validate_parallel_config']

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}")

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})")

        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if self.enable_eplb:
            if not self.enable_expert_parallel:
                raise ValueError(
                    "enable_expert_parallel must be True to use EPLB.")
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts.")

        if self.prefill_context_parallel_size > 1:
            raise ValueError(
                "Prefill context parallelism is not fully supported. "
                "Please set prefill_context_parallel_size to 1.")
        return self
