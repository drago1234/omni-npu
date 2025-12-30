# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def permute_module(monkeypatch):
    torch_npu = MagicMock()
    torch_npu.npu_swiglu.side_effect = lambda tensor: tensor
    torch_npu.npu_dequant_swiglu_quant.side_effect = (
        lambda gate_up_proj, **kwargs: (gate_up_proj, torch.ones(gate_up_proj.shape[0]))
    )

    def _grouped_matmul(inputs, weights, **kwargs):
        return [inputs[0]]

    torch_npu.npu_grouped_matmul.side_effect = _grouped_matmul

    config_module = types.ModuleType("vllm.model_executor.layers.fused_moe.config")

    class DummyFusedMoEQuantConfig:
        def __init__(self, use_int8_w8a8=False, w1_scale=None, w2_scale=None):
            self.use_int8_w8a8 = use_int8_w8a8
            self.w1_scale = w1_scale
            self.w2_scale = w2_scale

    config_module.FusedMoEQuantConfig = DummyFusedMoEQuantConfig

    modular_kernel_module = types.ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")

    class DummyExpertTokensMetadata:
        def __init__(self, expert_num_tokens):
            self.expert_num_tokens = expert_num_tokens

    class DummyActivationFormat:
        Standard = "standard"

    class DummyPermuteExpertsUnpermute:
        def __init__(self, quant_config):
            self.quant_config = quant_config

    modular_kernel_module.ExpertTokensMetadata = DummyExpertTokensMetadata
    modular_kernel_module.FusedMoEActivationFormat = DummyActivationFormat
    modular_kernel_module.FusedMoEPermuteExpertsUnpermute = DummyPermuteExpertsUnpermute
    modular_kernel_module.TopKWeightAndReduce = object

    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: SimpleNamespace(world_size=2)

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.modular_kernel", modular_kernel_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    sys.modules.pop("omni_npu.layers.fused_moe.npu_moe_permute_unpermute", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.npu_moe_permute_unpermute")
    importlib.reload(module)

    stubs = SimpleNamespace(
        torch_npu=torch_npu,
        quant_config_cls=DummyFusedMoEQuantConfig,
        expert_meta_cls=DummyExpertTokensMetadata,
        distributed=distributed_module,
    )
    return module, stubs


@pytest.mark.unit
def test_init_sets_scale_2_for_int8(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(
        w13_weight=[torch.zeros(1), torch.zeros(1), torch.zeros(1)],
        w13_weight_scale=torch.zeros(1, 4),
    )
    quant = stubs.quant_config_cls(use_int8_w8a8=True, w1_scale=torch.ones(1), w2_scale=torch.ones(1))

    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    assert permuter.local_expert_num == 3
    assert permuter.scale_2 is not None
    assert permuter.scale_2.shape == (3, 2)


@pytest.mark.unit
def test_init_sets_scale_2_none_for_bf16(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1)])
    quant = stubs.quant_config_cls(use_int8_w8a8=False)

    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    assert permuter.local_expert_num == 1
    assert permuter.scale_2 is None


@pytest.mark.unit
def test_activation_formats_and_supports(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1)])
    quant = stubs.quant_config_cls()
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    formats = permuter.activation_formats
    assert formats == (module.FusedMoEActivationFormat.Standard, module.FusedMoEActivationFormat.Standard)
    assert permuter.supports_chunking() is True
    assert permuter.supports_expert_map() is True


@pytest.mark.unit
def test_moe_problem_size(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1)])
    quant = stubs.quant_config_cls()
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    w1 = torch.zeros(2, 4, 8)
    w2 = torch.zeros(2, 4, 8)
    a1 = torch.zeros(3, 8)
    topk_ids = torch.zeros(3, 5, dtype=torch.int32)

    assert permuter.moe_problem_size(a1, w1, w2, topk_ids) == (2, 3, 4, 8, 5)


@pytest.mark.unit
def test_workspace_shapes_uses_world_size_and_min_topk(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1), torch.zeros(1)])
    quant = stubs.quant_config_cls()
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    shapes = permuter.workspace_shapes(
        M=4,
        N=6,
        K=8,
        topk=5,
        global_num_experts=8,
        local_num_experts=2,
        expert_tokens_meta=None,
    )

    expected_factor = 4 * min(5, 2) * 2
    assert shapes[0] == (expected_factor, 6)
    assert shapes[1] == (expected_factor, 3)
    assert shapes[2] == (expected_factor, 8)


@pytest.mark.unit
def test_apply_requires_expert_tokens_meta(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1)])
    quant = stubs.quant_config_cls()
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)

    with pytest.raises(ValueError, match="expert_tokens_meta"):
        permuter.apply(
            output=torch.zeros(1, 1),
            hidden_states=torch.zeros(1, 1),
            w1=torch.zeros(1, 1, 1),
            w2=torch.zeros(1, 1, 1),
            topk_weights=torch.zeros(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            activation="silu",
            global_num_experts=1,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.zeros(1, 1),
            workspace2=torch.zeros(1, 1),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


@pytest.mark.unit
def test_apply_int8_path_calls_npu_kernels(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(
        w13_weight=[torch.zeros(1)],
        w13_weight_scale=torch.zeros(1, 2),
    )
    quant = stubs.quant_config_cls(
        use_int8_w8a8=True,
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)
    expert_tokens_meta = stubs.expert_meta_cls(expert_num_tokens=torch.tensor([1]))

    expected = torch.full((2, 2), 7.0)
    stubs.torch_npu.npu_grouped_matmul.side_effect = [
        [torch.ones(2, 2, dtype=torch.int32)],
        [expected],
    ]

    output = torch.zeros(2, 2)
    permuter.apply(
        output=output,
        hidden_states=torch.ones(2, 2),
        w1=torch.ones(1, 2, 2),
        w2=torch.ones(1, 2, 2),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=torch.ones(1),
        a2_scale=None,
        workspace13=torch.zeros(1, 1),
        workspace2=torch.zeros(1, 1),
        expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=False,
    )

    assert torch.equal(output, expected)
    stubs.torch_npu.npu_dequant_swiglu_quant.assert_called_once()
    stubs.torch_npu.npu_swiglu.assert_not_called()


@pytest.mark.unit
def test_apply_bf16_path_calls_swiglu(permute_module):
    module, stubs = permute_module
    layer = SimpleNamespace(w13_weight=[torch.zeros(1)])
    quant = stubs.quant_config_cls(use_int8_w8a8=False)
    permuter = module.NPUFusedMoEPermuteExpertsUnpermute(quant, layer)
    expert_tokens_meta = stubs.expert_meta_cls(expert_num_tokens=torch.tensor([1]))

    expected = torch.full((2, 2), 3.0)
    stubs.torch_npu.npu_grouped_matmul.side_effect = [
        [torch.ones(2, 2)],
        [expected],
    ]

    output = torch.zeros(2, 2)
    permuter.apply(
        output=output,
        hidden_states=torch.ones(2, 2),
        w1=torch.ones(1, 2, 2),
        w2=torch.ones(1, 2, 2),
        topk_weights=torch.ones(1, 1),
        topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        activation="silu",
        global_num_experts=1,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.zeros(1, 1),
        workspace2=torch.zeros(1, 1),
        expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=False,
    )

    assert torch.equal(output, expected)
    stubs.torch_npu.npu_swiglu.assert_called_once()
    stubs.torch_npu.npu_dequant_swiglu_quant.assert_not_called()
