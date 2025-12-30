# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def prepare_module(monkeypatch):
    torch_npu = MagicMock()
    torch_npu.npu_moe_distribute_dispatch_v2 = MagicMock()
    torch_npu.npu_moe_distribute_combine_v2 = MagicMock()

    distributed_module = types.ModuleType("vllm.distributed")
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "hccl-comm")
    device_group = SimpleNamespace(_get_backend=lambda device: backend)
    distributed_module.get_ep_group = lambda: SimpleNamespace(
        device_group=device_group,
        rank_in_group=0,
    )

    fused_moe_module = types.ModuleType("vllm.model_executor.layers.fused_moe")

    class DummyFusedMoEConfig:
        def __init__(self, ep_size=2, ep_rank=0, num_experts=4):
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.num_experts = num_experts

    fused_moe_module.FusedMoEConfig = DummyFusedMoEConfig

    config_module = types.ModuleType("vllm.model_executor.layers.fused_moe.config")

    class DummyFusedMoEQuantConfig:
        def __init__(self, use_int8_w8a8=False):
            self.use_int8_w8a8 = use_int8_w8a8

    config_module.FusedMoEQuantConfig = DummyFusedMoEQuantConfig

    modular_kernel_module = types.ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")

    class DummyExpertTokensMetadata:
        def __init__(self, expert_num_tokens, expert_num_tokens_cpu):
            self.expert_num_tokens = expert_num_tokens
            self.expert_num_tokens_cpu = expert_num_tokens_cpu

    class DummyFusedMoEPrepareAndFinalize:
        pass

    class DummyActivationFormat:
        Standard = "standard"

    modular_kernel_module.ExpertTokensMetadata = DummyExpertTokensMetadata
    modular_kernel_module.FusedMoEActivationFormat = DummyActivationFormat
    modular_kernel_module.FusedMoEPrepareAndFinalize = DummyFusedMoEPrepareAndFinalize
    modular_kernel_module.PrepareResultType = tuple
    modular_kernel_module.TopKWeightAndReduce = object

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.modular_kernel", modular_kernel_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    sys.modules.pop("omni_npu.layers.fused_moe.npu_moe_prepare_finalize", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.npu_moe_prepare_finalize")
    importlib.reload(module)

    stubs = SimpleNamespace(
        torch_npu=torch_npu,
        moe_config_cls=DummyFusedMoEConfig,
        quant_config_cls=DummyFusedMoEQuantConfig,
        expert_meta_cls=DummyExpertTokensMetadata,
    )
    return module, stubs


@pytest.mark.unit
def test_init_sets_group_name_and_state(prepare_module):
    module, stubs = prepare_module
    moe = stubs.moe_config_cls(ep_size=2, ep_rank=1, num_experts=8)

    prepare = module.NpuMoEPrepareAndFinalize(moe)

    assert prepare.moe_all_to_all_group_name == "hccl-comm"
    assert prepare.expand_idx is None
    assert prepare.ep_recv_counts is None
    assert prepare.tp_recv_counts is None


@pytest.mark.unit
def test_prepare_sets_metadata_and_calls_dispatch(prepare_module):
    module, stubs = prepare_module
    moe = stubs.moe_config_cls(ep_size=2, ep_rank=0, num_experts=4)
    prepare = module.NpuMoEPrepareAndFinalize(moe)

    expand_x = torch.ones(2, 3)
    dynamic_scale = torch.ones(2, 1)
    expand_idx = torch.tensor([0, 1])
    expert_token_nums = torch.tensor([1, 1])
    ep_recv_counts = torch.tensor([2])
    tp_recv_counts = torch.tensor([2])
    stubs.torch_npu.npu_moe_distribute_dispatch_v2.return_value = (
        expand_x,
        dynamic_scale,
        expand_idx,
        expert_token_nums,
        ep_recv_counts,
        tp_recv_counts,
        "unused",
    )

    quant = stubs.quant_config_cls(use_int8_w8a8=True)
    result = prepare.prepare(
        a1=torch.ones(2, 3),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant,
    )

    assert result[0] is expand_x
    assert result[1] is dynamic_scale
    assert isinstance(result[2], stubs.expert_meta_cls)
    assert prepare.expand_idx is expand_idx
    assert prepare.ep_recv_counts is ep_recv_counts
    assert prepare.tp_recv_counts is tp_recv_counts
    stubs.torch_npu.npu_moe_distribute_dispatch_v2.assert_called_once()


@pytest.mark.unit
def test_prepare_uses_non_quant_mode_for_bf16(prepare_module):
    module, stubs = prepare_module
    moe = stubs.moe_config_cls(ep_size=1, ep_rank=0, num_experts=2)
    prepare = module.NpuMoEPrepareAndFinalize(moe)

    stubs.torch_npu.npu_moe_distribute_dispatch_v2.return_value = (
        torch.ones(1, 1),
        None,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([1]),
    )

    quant = stubs.quant_config_cls(use_int8_w8a8=False)
    prepare.prepare(
        a1=torch.ones(1, 1),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        num_experts=2,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=quant,
    )

    call_kwargs = stubs.torch_npu.npu_moe_distribute_dispatch_v2.call_args.kwargs
    assert call_kwargs["quant_mode"] == 0


@pytest.mark.unit
def test_finalize_calls_combine_and_copies_output(prepare_module):
    module, stubs = prepare_module
    moe = stubs.moe_config_cls(ep_size=2, ep_rank=0, num_experts=4)
    prepare = module.NpuMoEPrepareAndFinalize(moe)
    prepare.expand_idx = torch.tensor([0, 1])
    prepare.ep_recv_counts = torch.tensor([2])
    prepare.tp_recv_counts = torch.tensor([2])

    combined = torch.full((2, 2), 7.0)
    stubs.torch_npu.npu_moe_distribute_combine_v2.return_value = combined

    output = torch.zeros(2, 2)
    prepare.finalize(
        output=output,
        fused_expert_output=torch.ones(2, 2),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=None,
    )

    assert torch.equal(output, combined)
    stubs.torch_npu.npu_moe_distribute_combine_v2.assert_called_once()


@pytest.mark.unit
def test_properties_return_expected_values(prepare_module):
    module, stubs = prepare_module
    moe = stubs.moe_config_cls()
    prepare = module.NpuMoEPrepareAndFinalize(moe)

    assert prepare.activation_format == module.FusedMoEActivationFormat.Standard
    assert prepare.topk_indices_dtype() == torch.int32
    assert prepare.max_num_tokens_per_rank() is None
    assert prepare.num_dispatchers() == 1
    assert prepare.output_is_reduced() is False
