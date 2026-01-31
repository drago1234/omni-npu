# This patch is used for qwen3-next model
# Qwen3NextPatch and FusedRecurrentPatch are used to replace the community triton operators.
# KVCacheUtilsPatch's get_kv_cache_config_from_groups is designed to resolve the lack of support for shared kv_cache.
# KVCacheUtilsPatch's unify_hybrid_kv_cache_specs is used to fix the issue where disable_hybrid_kv_cache_manager is always true in the PD separation scenario of vLLM 0.12.0.
# SchedulerPatch supports retrieving block IDs for multiple groups.
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" 
# OMNI_NPU_VLLM_PATCHES="Qwen3NextPatch,FusedRecurrentPatch,KVCacheUtilsPatch,SchedulerPatch" before vllm serve

from math import log
import torch
from einops import rearrange
from dataclasses import replace
import vllm.model_executor.models.qwen3_next as qwen3_next
import vllm.model_executor.layers.fla.ops.fused_recurrent as fused_recurrent
import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    AttentionSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.core.sched.scheduler import Scheduler
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.layers.ops.fla_ops import chunk_gated_delta_rule as chunk_gated_delta_rule_npu
from omni_npu.layers.ops.fla_ops import fused_recurrent_gated_delta_rule_fwd \
    as fused_recurrent_gated_delta_rule_fwd_npu
from omni_npu.layers.ops.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
from omni_npu.layers.ops.causal_conv1d import causal_conv1d_update as causal_conv1d_update_npu

@register_patch("Qwen3NextPatch", qwen3_next)
class Qwen3NextPatch(VLLMPatch):
    """
    Patch qwen3-next custom op for compatibility with vLLM.
    """
    _attr_names_to_apply = [
        'fused_gdn_gating', 'causal_conv1d_fn', 'causal_conv1d_update',
        'chunk_gated_delta_rule'
    ]

    def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = b.sigmoid()
        g = -A_log.float().exp() * (1 + (a.float() + dt_bias).exp()).log()
        g, beta = map(lambda x: rearrange(x, 'l d -> 1 l d'), (g, beta))
        return g, beta
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
    chunk_gated_delta_rule = chunk_gated_delta_rule_npu

@register_patch("FusedRecurrentPatch", fused_recurrent)
class FusedRecurrentPatch(VLLMPatch):
    """
    Patch qwen3-next custom op for compatibility with vLLM.
    """
    _attr_names_to_apply = ['fused_recurrent_gated_delta_rule_fwd']
    fused_recurrent_gated_delta_rule_fwd = fused_recurrent_gated_delta_rule_fwd_npu

@register_patch("KVCacheUtilsPatch", kv_cache_utils)
class KVCacheUtilsPatch(VLLMPatch):
    """
    Patch kv_cache_utils for compatibility with vLLM.
    For get_kv_cache_config_from_groups, the added lines are between the patch begin and patch end comments.
    For unify_hybrid_kv_cache_specs, skip this method.
    """
    _attr_names_to_apply = ['get_kv_cache_config_from_groups', 'unify_hybrid_kv_cache_specs']

    def get_kv_cache_config_from_groups(
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        """
        Generate the KV cache configuration from the KV cache groups and spec
        of each layer.

        Args:
            vllm_config: The global VllmConfig
            kv_cache_groups: The KV cache groups
            available_memory: Memory available for KV cache in bytes
        Returns:
            The generated KVCacheConfig
        """
        if len(kv_cache_groups) == 0:
            # Attention free models do not have KV cache.
            # Return num_blocks=1 as BlockPool always needs a null_block.
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
            )

        # Determine how model runners should initialize the KV cache tensors.
        if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            # Special case: all layers have the same type of KV cache but with
            # different hidden size. Allocate different amount of memory for each
            # layer based on its hidden size.
            num_blocks = (
                available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
            )
            num_blocks = kv_cache_utils.may_override_num_blocks(vllm_config, num_blocks)
            per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
            kv_cache_tensors = [
                KVCacheTensor(
                    size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in kv_cache_groups[0].layer_names
            ]
        else:
            # General case:
            # We will have group_size memory pools, each is shared by one layer from
            # each group. As layers of different groups have different block table,
            # they will use different parts of the shared Tensor.
            # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
            # (sw.1, padding) will be: (group_size = 2)
            # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
            # full.1, sw.2: share another Tensor with size=available_memory//2
            group_size = max(len(group.layer_names) for group in kv_cache_groups)

            page_size = kv_cache_utils.get_uniform_page_size(
                [group.kv_cache_spec for group in kv_cache_groups]
            )
            assert group_size > 0, "group_size must be greater than 0"
            num_blocks = kv_cache_utils.get_num_blocks(
                vllm_config, group_size, available_memory, page_size
            )
            # patch begin
            # Community impl shares kv_cache across groups, but llmdatadist doesn't support it — reworked for non-sharing.
            num_blocks //= len(kv_cache_groups)
            kv_cache_tensors = []
            for i in range(group_size):
                for j in range(len(kv_cache_groups)):
                    kv_cache_spec = kv_cache_groups[j].kv_cache_spec
                    # kv_cache not shared, MambaSpec no padding; set page_size_padded to None for its actual page_size_bytes (reduces memory waste),
                    # replace method is safe and won't modify the original object.
                    if isinstance(kv_cache_spec, MambaSpec):
                        kv_cache_spec = replace(kv_cache_spec, page_size_padded=None)
                    page_size = kv_cache_spec.page_size_bytes
                    kv_cache_tensors.append(
                        KVCacheTensor(size=page_size * num_blocks, shared_by=[kv_cache_groups[j].layer_names[i]])
                    )
            # patch end

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )

    def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
        """
        This function tries to convert the KV cache specs to one type if the model
        is a hybrid model with multiple type of KV cache. It will convert all
        SlidingWindowSpec to FullAttentionSpec if both types are present.

        Args:
            kv_cache_spec: The kv cache spec of each attention layer in the model
        """
        pass

@register_patch("SchedulerPatch", Scheduler)
class SchedulerPatch(VLLMPatch):
    """
    Patch scheduler for compatibility with vLLM.
    The changes lines are between the patch begin and patch end comments.
    origin (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
    now get AttentionSpec's block_ids
    """
    _attr_names_to_apply = ['_update_waiting_for_remote_kv']

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # pacth begin
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            if len(block_ids) == 1:
                block_ids = block_ids[0]
            else:
                # find AttentionSpec
                for idx, group in enumerate(self.kv_cache_manager.kv_cache_config.kv_cache_groups):
                    if isinstance(group.kv_cache_spec, AttentionSpec):
                        block_ids = block_ids[idx]
                        break
            # patch end
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True
