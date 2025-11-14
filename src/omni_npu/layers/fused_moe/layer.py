# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable
import torch, torch_npu
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.distributed import get_world_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoeWeightScaleSupported,
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.custom_op import CustomOp


logger = init_logger("vllm.omni_npu.layers.fused_moe.layer")


@UnquantizedFusedMoEMethod.register_oot
class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.moe_infer_fusion(layer,
                                    x,
                                    topk_ids,
                                    topk_weights,
                                    layer.w13_weight,
                                    layer.w2_weight)
        return out

    def moe_infer_fusion(self, layer, x, topk_ids, topk_weight, w1, w2):
        hidden_states = x.view(-1, x.shape[-1])
        topk_weight = topk_weight.to(x.dtype)
        max_num_deployed_expert = self.moe.num_experts
        topk_ids = topk_ids.int()

        expert_range = [0, max_num_deployed_expert]
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            scale=None,
            expert_num=max_num_deployed_expert,
            active_expert_range=expert_range,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=-1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group,
                               tokens_per_expert)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)

        # combine tensors, do reduceSum and D2H toghter
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E//EP
        # sum: EP, the number of tokens each rank receives from other cards
        ep_size = get_ep_group().world_size
        combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, the total number of tokens routed from the current rank to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, the number of tokens each rank receives from other cards
        output_splits = combine_tokens_cpu[0]
        # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
        gathered_tokens = expanded_x.new_empty(
            all_tokens.item(), expanded_x.shape[1]
        )
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits)
        # reroute
        # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
        hidden_states_sorted_by_experts, _, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            tokens_per_expert_group.view(ep_size, -1),
            per_token_scales=None
        )
        group_list = tokens_per_local_expert.to(torch.int64)
        mm1_mm3 = torch_npu.npu_grouped_matmul([hidden_states_sorted_by_experts], [w1],
                                               group_list=group_list, split_item=3, group_type=0,
                                               group_list_type=1)[0]
        intermediate_h = torch_npu.npu_swiglu(mm1_mm3)
        # gmm2: down
        hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
            [intermediate_h],
            [w2],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=1
        )[0]
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0,
                                   gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32))
        gathered_tokens = new_x.new_empty(*expanded_x.shape)

        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)
        return hidden_states, gathered_tokens, topk_weight, expanded_row_idx

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = (
            get_ep_group().rank_in_group * self.n_routed_experts
        )
        self.local_expert_indices = [
            self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
        ]
        self.initialized = True


class AscendFusedMoE(FusedMoE):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: Optional[torch.Tensor] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            # profile run, force load balance
            ep_rank = get_ep_group().rank
            global_num_experts = router_logits.shape[1]
            num_tokens = router_logits.shape[0]
            topk_ids = torch.arange(
                ep_rank * num_tokens * top_k,
                (ep_rank+1) * num_tokens * top_k,
                dtype=torch.int32,
                device=router_logits.device,
            ).view(num_tokens, top_k) % global_num_experts
            topk_weights = torch.rand_like(topk_ids)
            row_idx = torch.arange(topk_ids.numel(),
                                   device=topk_ids.device,
                                   dtype=torch.int32) \
                            .view(-1, num_tokens) \
                            .transpose(0, 1)
            return topk_weights, topk_ids, row_idx

        if use_grouped_topk and num_expert_group != 1:
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,
                bias=e_score_correction_bias,
                k_group=topk_group,
                group_count=num_expert_group,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=routed_scaling_factor,
                eps=1e-20,
            )
            row_idx = torch.arange(topk_ids.numel(),
                                   device=current_platform.device_type,
                                   dtype=torch.int32) \
                            .view(-1, router_logits.shape[0]) \
                            .transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(router_logits, k=top_k)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.tensor,
                                       tp_rank: int):
        # adapt loaded_weight shape
        loaded_weight = loaded_weight.squeeze(-1)
        # adapt end
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """With NPU all-to-all, there is no need to perform all-reduce on final hidden states.
        """
        return final_hidden_states

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,):
                # topk_weights: torch.Tensor,
                # topk_ids: torch.Tensor,
                # pertoken_scale: torch.Tensor,
                # comm_group: Optional[GroupCoordinator] = None):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        shared_output = self.shared_experts(hidden_states)
        # NOTE(runze): `shared_experts` MLP has been created with
        # reduce_results=False, so it's necessary to reduce here.
        shared_output = tensor_model_parallel_all_reduce(shared_output)

        topk_weights, topk_ids, row_idx = AscendFusedMoE.select_experts(
            hidden_states,
            router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias
        )

        # Matrix multiply.
        final_hidden_states_list = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        final_hidden_states = final_hidden_states_list[0]
        gathered_tokens = final_hidden_states_list[1]
        expanded_row_idx = final_hidden_states_list[3]

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2
        )

        if self.reduce_results and self.tp_size > 1:
            # NOTE(runze): this branch should not be triggered because self.reduce_results is False
            logger.error(f"Reduce should not be performed. Something may be wrong! {self.reduce_results=}.")
            final_hidden_states = get_tp_group().all_reduce(final_hidden_states)

        return shared_output, final_hidden_states

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> None:
        if self.ep_size:
            ep_rank = self.ep_rank
            if expert_id < ep_rank * self.local_num_experts or expert_id >= (ep_rank + 1) * self.local_num_experts:
                return False if return_success else None
            tp_rank = 0
            expert_id -= ep_rank * self.local_num_experts
        else:
            tp_rank = self.tp_rank
        # compressed-tensors checkpoints with packed weights are stored flipped
        loaded_weight = loaded_weight.t().contiguous() if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod") else loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = ~shard_dim

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=tp_rank)
            return True if return_success else None

        # Case weight scales and zero_points
        if ("scale" in weight_name or "zero" in weight_name or "offset" in weight_name or "bias" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                if "int4_scale" in weight_name:
                    shard_dim = 1
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                shard_dim = 1
                if "bias" in weight_name:
                    shard_dim = 0
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None
        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank)
            return True if return_success else None


class AscendNonOverlapSharedFusedMoE(AscendFusedMoE):

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return self._shared_experts if self.use_overlapped else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            shared_out = self._shared_experts(hidden_states)

            # Reduce outputs if necessary, since the MLP should
            # have been created with reduce_results=False.
            if (
                self.reduce_results
                and self.tp_size > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return shared_out, fused_out


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
):
    gating_output = gating_output.float()
    # scores = torch.softmax(gating_output, dim=-1)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx
