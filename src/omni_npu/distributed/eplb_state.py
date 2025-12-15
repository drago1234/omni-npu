# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from dataclasses import dataclass
import torch
from typing import Optional
from vllm.config import ParallelConfig
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config

logger = init_logger(__name__)

try:
    from omni_placement.omni_planner import OmniPlanner
except ImportError:
    logger.warning("OmniPlanner is not installed, please install it when using vllm --enable_eplb True")
    OmniPlanner = None

@dataclass
class EplbState:
    planner: Optional[OmniPlanner] = None
    start_step: bool = False

    @classmethod
    def build(cls,
              model: MixtureOfExperts,
              device: torch.device,
              parallel_config: ParallelConfig,
              global_expert_load: Optional[torch.Tensor] = None,
              old_global_expert_indices: Optional[torch.Tensor] = None,
              rank_mapping: Optional[dict[int, int]] = None,
              ) -> "EplbState":
        if OmniPlanner is not None:
            param_dict = dict(model.named_parameters())
            planner = OmniPlanner()
            planner.init_dram_weights(param_dict, first_k_dense_replace=3)
            return cls(planner)
        else:
            planner = None

    def step(
        self,
        model: MixtureOfExperts,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False) -> None:
        if OmniPlanner is not None:
            if not self.start_step:
                self.planner.start_dynamic_optimize_expert_load_balance()
                self.start_step = True
            self.planner.place_experts()
        else:
            pass

    @staticmethod
    def recv_state() -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def rearrange(
        self,
        model: MixtureOfExperts,
        is_profile: bool = False,
        execute_shuffle: bool = True,
        global_expert_load: Optional[torch.Tensor] = None,
        rank_mapping: Optional[dict[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        pass
