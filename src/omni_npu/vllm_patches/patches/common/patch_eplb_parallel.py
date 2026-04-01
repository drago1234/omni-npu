# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch only contains EPLB-specific SharedFusedMoE behavior.
# Engine-config wrapping is handled by patch_args_utils.py.

import torch

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

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
