import torch
from vllm.forward_context import get_forward_context
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
)
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from contextlib import nullcontext
from vllm.utils.torch_utils import current_stream


from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@register_patch("FusedMoEPatch", FusedMoE)
class FusedMoEPatch(VLLMPatch):
    _attr_names_to_apply = ['forward_native', 'forward_impl']

    
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        if self.shared_experts is None:
            if current_platform.is_tpu() or current_platform.is_out_of_tree():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                return (reduce_output(fused_output) + zero_expert_result)[
                    ..., :og_hidden_states
                ]
            else:
                return reduce_output(fused_output)[..., :og_hidden_states]
        else:
            if current_platform.is_tpu() or current_platform.is_out_of_tree():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            return (
                reduce_output(shared_output)[..., :og_hidden_states],
                reduce_output(fused_output)[..., :og_hidden_states],
            )

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self.ensure_moe_quant_config_init()
        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not isinstance(self.quant_method, FusedMoEModularMethod)
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared_experts, use_chunked_impl
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

        do_naive_dispatch_combine: bool = self.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states_combined, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )
            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_output = self.shared_experts(hidden_states)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states_combined
                if do_naive_dispatch_combine
                else hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not self.rocm_aiter_fmoe_enabled
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                if isinstance(final_hidden_states, tuple):
                    return (
                        final_hidden_states[0],
                        combine_output(final_hidden_states[1]),
                    )
                else:
                    shared_output = self.shared_experts(hidden_states)
                    return (
                        shared_output,
                        combine_output(final_hidden_states),
                    )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return (combine_output(final_hidden_states), zero_expert_result)
            else:
                return combine_output(final_hidden_states)
