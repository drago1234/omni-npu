# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch is used for enable_eplb fix in ParallelConfig and FusedMoE
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" OMNI_NPU_VLLM_PATCHES="EPLBParallelConfig,EPLBFusedMoE" before vllm serve
from vllm.config.parallel import ParallelConfig
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

import os
import torch
from typing import Callable, TYPE_CHECKING, Any, Literal, Optional, Union
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cuda_device_count_stateless, get_open_ports_list
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

    _attr_names_to_apply = ['__post_init__']

    def __post_init__(self) -> None:
        # Forward deprecated fields to their new location
        if self.num_redundant_experts is not None:
            self.eplb_config.num_redundant_experts = self.num_redundant_experts
            logger.warning_once(
                "num_redundant_experts is deprecated and has been replaced "
                "with eplb_config.num_redundant_experts. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect."
            )
        if self.eplb_window_size is not None:
            self.eplb_config.window_size = self.eplb_window_size
            logger.warning_once(
                "eplb_window_size is deprecated and has been replaced "
                "with eplb_config.window_size. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect."
            )
        if self.eplb_step_interval is not None:
            self.eplb_config.step_interval = self.eplb_step_interval
            logger.warning_once(
                "eplb_step_interval is deprecated and has been replaced "
                "with eplb_config.step_interval. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect."
            )
        if self.eplb_log_balancedness is not None:
            self.eplb_config.log_balancedness = self.eplb_log_balancedness
            logger.warning_once(
                "eplb_log_balancedness is deprecated and has been replaced "
                "with eplb_config.log_balancedness. This will be removed "
                "in v0.12.0. Changing this field after initialization will "
                "have no effect."
            )
        # Continue with the rest of the initialization
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self._data_parallel_master_port_list:
                self._data_parallel_master_port_list = get_open_ports_list(5)
            self.data_parallel_master_port = self._data_parallel_master_port_list.pop()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_external_lb:
                raise ValueError(
                    "data_parallel_external_lb can only "
                    "be set when data_parallel_size > 1"
                )

        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.enable_eplb:
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
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
                    "num_redundant_experts."
                )
        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif (
                current_platform.is_cuda()
                and cuda_device_count_stateless() < self.world_size
            ):
                if not ray_found:
                    raise ValueError(
                        "Unable to load Ray: "
                        f"{ray_utils.ray_import_err}. Ray is "
                        "required for multi-node inference, "
                        "please install Ray with `pip install "
                        "ray`."
                    )
                backend = "ray"
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        if not -1 <= self._api_process_rank < self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

@register_patch("EPLBFusedMoE", FusedMoE)
class FusedMoEPatch(VLLMPatch):
    """
    FusedMoEPatch for EPLB on NPU devices (when eplb enabled).
    """

    _attr_names_to_apply = ['__init__']

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
        expert_mapping: Optional[list[tuple[str, str, int, str]]] = None,
    ):
        super(FusedMoE, self).__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size, moe_in_dtype, quant_config, self.moe_parallel_config
        )

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            expert_placement_strategy = (
                vllm_config.parallel_config.expert_placement_strategy
            )
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = (
                    (num_expert_group is not None and num_expert_group > 1)
                    and num_redundant_experts == 0
                    and not self.enable_eplb
                )

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement."
                    )
                    expert_placement_strategy = "linear"

            self.expert_map: Optional[torch.Tensor]
            local_num_experts, expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts, None)

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
        )
        self.moe_config = moe
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None
        quant_method = (
            UnquantizedFusedMoEMethod(moe)
            if quant_config is None
            else quant_config.get_quant_method(self, prefix)
        )
        if quant_method is None:
            quant_method = UnquantizedFusedMoEMethod(moe)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

        # TODO(bnell): flashinfer uses non-batched format.
        # Does it really need a batched buffer?
        if (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_config.use_flashinfer_cutlass_kernels
        ):
            if vllm_config.parallel_config.enable_dbo:
                self.batched_hidden_states = torch.zeros(
                    (2, moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device(),
                )

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (2, moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device(),
                )
            else:
                self.batched_hidden_states = torch.zeros(
                    (moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device(),
                )

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device(),
                )
