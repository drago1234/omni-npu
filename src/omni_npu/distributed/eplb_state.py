# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from dataclasses import dataclass
import torch
from typing import Optional, Sequence
from vllm.config import VllmConfig
from vllm.config import ModelConfig, ParallelConfig
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

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.planner = None
        self.start_step = None
        self.is_async = False

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        global_expert_load: torch.Tensor | None = None,
        old_global_expert_indices: torch.Tensor | None = None,
        rank_mapping: dict[int, int] | None = None,
    ):
        if OmniPlanner is not None and model_config.runner != "draft":
            param_dict = dict(model.named_parameters())
            planner = OmniPlanner()
            planner.init_dram_weights(param_dict, first_k_dense_replace=3)
            self.planner = planner
        else:
            self.planner = None

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        if OmniPlanner is not None:
            if not self.start_step:
                self.planner.start_dynamic_optimize_expert_load_balance()
                self.start_step = True
            self.planner.place_experts()
        else:
            pass

    def rearrange(
        self,
        is_profile: bool = False,
        execute_shuffle: bool = True,
        global_expert_loads: list[torch.Tensor] | None = None,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        pass

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        pass

    @staticmethod
    def recv_state() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pass

    @classmethod
    def get_eep_state(
        cls, parallel_config: ParallelConfig
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        dict[int, int] | None,
    ]:
        global_expert_loads = None
        old_global_expert_indices_per_model = None
        rank_mapping = None
        return (
            global_expert_loads, 
            old_global_expert_indices_per_model, 
            rank_mapping,
        ) 
def _init_omni_eplb_configs(vllm_config: VllmConfig, local_rank: int)-> None:
    if vllm_config.additional_config is None or "omni_placement_config" not in vllm_config.additional_config:
        return
    if not vllm_config.parallel_config.enable_eplb:
        return
    else:            
        if local_rank == 0:
            logger.info("Enable omni eplb for vLLM NPUWorker in local rank 0")
            from omni_placement.utils import apply_omni_eplb_attributes
            apply_omni_eplb_attributes(additional_config=vllm_config.additional_config)
        else:
            return       
