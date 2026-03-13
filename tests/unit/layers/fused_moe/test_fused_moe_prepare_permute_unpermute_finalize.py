# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


pytestmark = pytest.mark.unit


@pytest.fixture
def prepare_module(monkeypatch):
    torch_npu = MagicMock()
    torch_npu.npu_moe_init_routing_v2 = MagicMock()
    torch_npu.npu_dynamic_quant = MagicMock()
    torch_npu.npu_moe_re_routing = MagicMock()
    torch_npu.npu_moe_finalize_routing = MagicMock()
    torch_npu.npu_moe_distribute_dispatch_v2 = MagicMock()
    torch_npu.npu_moe_distribute_combine_v2 = MagicMock()
    torch_npu.npu = SimpleNamespace(get_device_name=lambda _: "Ascend910C")

    class DummyBackend:
        def get_hccl_comm_name(self, rank_in_group):
            return f"hccl_{rank_in_group}"

    class DummyDeviceGroup:
        def _get_backend(self, device):
            return DummyBackend()

    ep_group = SimpleNamespace(
        world_size=2,
        rank=1,
        rank_in_group=1,
        device_group=DummyDeviceGroup(),
        all_gather=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim),
    )
    context_holder = SimpleNamespace(attn_metadata=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: ep_group
    distributed_module.get_dp_group = lambda: SimpleNamespace(
        world_size=1,
        all_gather=lambda tensor, dim=0: tensor,
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_reduce=lambda x: x)
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)

    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: context_holder
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    math_utils_module = types.ModuleType("vllm.utils.math_utils")
    math_utils_module.cdiv = lambda a, b: (a + b - 1) // b
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", math_utils_module)

    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda output, input, *args, **kwargs: output.copy_(input),
        raising=False,
    )

    module_name = "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    importlib.reload(module)

    stubs = SimpleNamespace(
        torch_npu=torch_npu,
        ep_group=ep_group,
        context_holder=context_holder,
    )
    return module, stubs


def test_all2all_prepare_permute_no_quant(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_method=SimpleNamespace(
            moe_quant_config=None,
            num_of_redundant_experts=0,
        ),
        quant_config=None,
        w13_weight=torch.zeros(2, 1, 1),
    )
    expanded_x = torch.arange(6, dtype=torch.float32).view(2, 3)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    tokens_per_expert = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        tokens_per_expert,
        None,
    )
    sorted_states = torch.full((2, 3), 2.0)
    gathered_idxs_unsort = torch.tensor([1, 0], dtype=torch.int32)
    tokens_per_local_expert = torch.tensor([1, 1], dtype=torch.int32)
    stubs.torch_npu.npu_moe_re_routing.return_value = (
        sorted_states,
        None,
        gathered_idxs_unsort,
        tokens_per_local_expert,
    )

    handler = module.All2AllPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
    )

    assert result.hidden_states_sorted_by_experts is sorted_states
    assert result.dynamic_scale is None
    assert result.input_splits == [1, 1]
    assert result.output_splits == [1, 1]
    assert result.expert_tokens.equal(tokens_per_local_expert)
    assert stubs.torch_npu.npu_moe_init_routing_v2.call_args.kwargs["quant_mode"] == -1


def test_agrs_prepare_permute_with_quant(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=MagicMock(),
        w13_weight=torch.zeros(2, 1, 1),  # shape: (local_num_experts, ...)
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    x_int8 = x.to(torch.int8)
    x_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)

    stubs.torch_npu.npu_dynamic_quant.return_value = (x_int8, x_scale)

    expanded_x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    expanded_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)

    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        expert_tokens,
        expanded_scale,
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )
    
    stubs.torch_npu.npu_moe_init_routing_v2.assert_called_once()
    assert torch.equal(result.hidden_states_sorted_by_experts, expanded_x)
    assert torch.equal(result.expert_tokens, expert_tokens)
    assert result.dtype == x.dtype

def test_all2all_unpermute_finalize_reorders_and_finalizes(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_method=SimpleNamespace(
            moe_quant_config=None,
            num_of_redundant_experts=0,
        ),
    )
    handler = module.All2AllPrepPmtAndUnpmtFinal(layer)
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((2, 2), 5.0)

    prepare_result = module.All2AllPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=None,
        gathered_idxs_unsort=torch.tensor([1, 0], dtype=torch.int32),
        expanded_x=torch.zeros(2, 2),
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        input_splits=[1, 1],
        output_splits=[1, 1],
    )
    hidden_states = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    topk_weights = torch.ones(2, 1, dtype=torch.float32)

    output = handler.unpermute_finalize(
        layer=layer,
        hidden_states=hidden_states,
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=topk_weights,
        all2all_prepare_permute_result=prepare_result,
    )

    assert torch.equal(output, torch.full((2, 2), 5.0))
    stubs.torch_npu.npu_moe_finalize_routing.assert_called_once()
    assert (
        stubs.torch_npu.npu_moe_finalize_routing.call_args.kwargs["scales"].dtype
        == hidden_states.dtype
    )


def test_dispatch_prepare_permute_passes_quant_mode_and_mask(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    stubs.context_holder.attn_metadata = {
        0: SimpleNamespace(decode=SimpleNamespace(mc2_mask=torch.tensor([1, 0, 1], dtype=torch.bool)))
    }
    dispatch_out = (
        torch.ones(2, 3),
        torch.ones(2),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    stubs.torch_npu.npu_moe_distribute_dispatch_v2.return_value = dispatch_out

    handler = module.DispatchCombinePrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert result.expert_tokens.dtype == torch.int64
    kwargs = stubs.torch_npu.npu_moe_distribute_dispatch_v2.call_args.kwargs
    assert kwargs["quant_mode"] == 2
    assert kwargs["group_ep"] == "hccl_1"
    assert torch.equal(kwargs["x_active_mask"], torch.tensor([1, 0], dtype=torch.bool))


def test_dispatch_unpermute_finalize_passes_counts_and_mask(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=None,
    )
    stubs.context_holder.attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(mc2_mask=torch.tensor([1, 1], dtype=torch.bool))
    )
    stubs.torch_npu.npu_moe_distribute_combine_v2.return_value = torch.full((2, 2), 7.0)
    handler = module.DispatchCombinePrepPmtAndUnpmtFinal(layer)
    prepare_result = module.DispatchCombinePreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        tp_recv_counts=torch.tensor([1, 1], dtype=torch.int32),
        ep_recv_counts=torch.tensor([1, 1], dtype=torch.int32),
        expand_idx=torch.tensor([0, 1], dtype=torch.int32),
    )

    output = handler.unpermute_finalize(
        layer=layer,
        hidden_states=torch.ones(2, 2),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        dispatch_combine_prepare_permute_result=prepare_result,
    )

    assert torch.equal(output, torch.full((2, 2), 7.0))
    kwargs = stubs.torch_npu.npu_moe_distribute_combine_v2.call_args.kwargs
    assert kwargs["ep_send_counts"] is prepare_result.ep_recv_counts
    assert kwargs["tp_send_counts"] is prepare_result.tp_recv_counts
    assert torch.equal(kwargs["x_active_mask"], torch.tensor([1, 1], dtype=torch.bool))


def test_agrs_unpermute_finalize_quant_masks_out_of_range_experts(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.ones(2, 2, dtype=torch.float32)
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        expert_range=[2, 4],
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0], [3]], dtype=torch.int32),
        dtype=torch.float16,
    )
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    hidden_states = torch.ones(2, 2, dtype=torch.float32)

    y = handler.unpermute_finalize(
        layer=layer,
        hidden_states=hidden_states,
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=topk_weights,
        agrs_prepare_permute_result=prepare_result,
    )

    assert y.dtype == torch.float16
    scales = stubs.torch_npu.npu_moe_finalize_routing.call_args.kwargs["scales"]
    assert torch.equal(scales, torch.tensor([[0.0], [1.0]], dtype=torch.float32))


def test_strategy_selector_a2_tp_dp_branches(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=4)}
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy_small, _ = selector.select_communication_strategy(num_tokens=4)
    strategy_large, _ = selector.select_communication_strategy(num_tokens=8)

    assert strategy_small == "agrs"
    assert strategy_large == "all2all"


def test_strategy_selector_a2_tp_only_always_agrs(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=0)}
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=999)

    assert strategy == "agrs"


def test_strategy_selector_returns_dispatch_for_small_token_on_non_a2(prepare_module):
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=4)
    assert strategy == "dispatch_combine"


def test_strategy_selector_non_a2_tp_dp_branches(prepare_module):
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=16)}
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy_small, _ = selector.select_communication_strategy(num_tokens=8)
    strategy_large, _ = selector.select_communication_strategy(num_tokens=200)

    assert strategy_small == "agrs"
    assert strategy_large == "all2all"