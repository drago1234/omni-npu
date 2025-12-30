# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch is used for enable_eplb fix in ParallelConfig and FusedMoE
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" OMNI_NPU_VLLM_PATCHES="EPLBEngineConfig,EPLBSharedFusedMoE" before vllm serve
from vllm import EngineArgs
_Orig_Create_Engine_Config = EngineArgs.create_engine_config
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

import os
import torch
from typing import Callable, TYPE_CHECKING, Any, Literal, Optional, Union
from typing_extensions import Self

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.config import VllmConfig
if TYPE_CHECKING:
    from vllm.usage.usage_lib import UsageContext
else:
    UsageContext = Any

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]

@register_patch("EPLBEngineConfig", EngineArgs)
class EngineConfigPatch(VLLMPatch):
    """
    EngineConfig for EPLB on NPU devices (when eplb enabled).
    """
    _attr_names_to_apply = ['create_engine_config']

    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:

        from vllm.platforms import current_platform

        _orig_is_cuda_alike = current_platform.is_cuda_alike

        def _npu_temp_cuda_alike_true():
            if getattr(current_platform, "device_type", None) == "npu":
                return True
            return _orig_is_cuda_alike()

        current_platform.is_cuda_alike = _npu_temp_cuda_alike_true
        try:
            return _Orig_Create_Engine_Config(self, usage_context, headless)
        finally:
            current_platform.is_cuda_alike = _orig_is_cuda_alike

@register_patch("EPLBSharedFusedMoE", SharedFusedMoE)
class SharedFusedMoEPatch(VLLMPatch):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """
    _attr_names_to_apply = ['__init__']

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super(SharedFusedMoE, self).__init__(**kwargs)
        self._shared_experts = shared_experts
        self.use_overlapped = (
            use_overlapped
            and not (
                # TODO(wentao): find the root cause and remove this condition
                False
                or (self.moe_config.use_flashinfer_cutlass_kernels and self.dp_size > 1)
            )
            and self._shared_experts is not None
        )
        self._gate = gate