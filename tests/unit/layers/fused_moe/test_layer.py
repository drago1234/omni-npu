# SPDX-License-Identifier: MIT
import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def layer_module(monkeypatch):
    torch_npu = sys.modules["torch_npu"]
    torch_npu.npu_format_cast = MagicMock(side_effect=lambda tensor, _: tensor)
    torch_npu.npu_grouped_matmul = MagicMock(
        side_effect=lambda inputs, _weights, **kwargs: [inputs[0]]
    )
    torch_npu.npu_swiglu = MagicMock(side_effect=lambda tensor: tensor)
    torch_npu.npu_moe_gating_top_k = MagicMock(
        return_value=(
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    torch_npu.npu = SimpleNamespace(get_device_name=lambda _: "Ascend910C")

    context_holder = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_dp_group",
        lambda: SimpleNamespace(
            world_size=1,
            all_gather=lambda tensor, dim=0: tensor,
            reduce_scatter=lambda tensor, dim=0: tensor,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_tp_group",
        lambda: SimpleNamespace(all_reduce=lambda x: x),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm.forward_context"],
        "get_forward_context",
        lambda: context_holder,
        raising=False,
    )

    sys.modules.pop("omni_npu.layers.fused_moe.layer", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.layer")
    importlib.reload(module)
    return module, torch_npu, context_holder


class _DummyStrategy:
    def prepare_permute(self, layer, x, topk_ids):
        return SimpleNamespace(
            hidden_states_sorted_by_experts=x,
            expert_tokens=torch.tensor([x.shape[0]], dtype=torch.int64),
            dynamic_scale=None,
        )

    def unpermute_finalize(self, layer, hidden_states, topk_ids, topk_weights, result):
        return hidden_states


class _DummyStream:
    def wait_stream(self, _other):
        return None


@contextmanager
def _stream_ctx(_stream):
    yield


@pytest.mark.unit
def test_select_experts_profile_mode(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = None
    module.get_ep_group = MagicMock(return_value=SimpleNamespace(rank=1))

    topk_weights, topk_ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(2, 4),
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
    )

    assert topk_weights.shape == (2, 2)
    assert topk_ids.shape == (2, 2)
    assert torch.all(topk_ids < 4)


@pytest.mark.unit
def test_select_experts_grouped_topk_requires_group_args(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}

    with pytest.raises(ValueError, match="topk_group is None"):
        module.NPUFusedMoE.select_experts(
            router_logits=torch.zeros(1, 4),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=None,
            num_expert_group=2,
        )


@pytest.mark.unit
def test_apply_experts_uses_grouped_matmul_twice(layer_module):
    module, torch_npu, _ = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        w13_weight=torch.ones(2, 4, 4),
        w2_weight=torch.ones(2, 4, 4),
    )
    prepare_result = module.PreparePermuteResult(
        hidden_states_sorted_by_experts=torch.ones(3, 4),
        expert_tokens=torch.tensor([2, 1], dtype=torch.int64),
        dynamic_scale=None,
    )

    out = method.apply_experts(layer, prepare_result)
    assert out.shape == (3, 4)
    assert torch_npu.npu_grouped_matmul.call_count == 2


@pytest.mark.unit
def test_maybe_all_reduce_tensor_model_parallel(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.moe_parallel_config = SimpleNamespace(use_ep=True)
    module.tensor_model_parallel_all_reduce = MagicMock(return_value=torch.tensor([9.0]))

    x = torch.tensor([1.0])
    assert torch.equal(fused.maybe_all_reduce_tensor_model_parallel(x), x)
    module.tensor_model_parallel_all_reduce.assert_not_called()

    fused.moe_parallel_config.use_ep = False
    y = fused.maybe_all_reduce_tensor_model_parallel(x)
    module.tensor_model_parallel_all_reduce.assert_called_once_with(x)
    assert torch.equal(y, torch.tensor([9.0]))


@pytest.mark.unit
def test_select_experts_custom_routing_function(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    custom_fn = MagicMock(
        return_value=(
            torch.full((2, 1), 0.5, dtype=torch.float32),
            torch.ones(2, 1, dtype=torch.int32),
        )
    )

    topk_weights, topk_ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(2, 4),
        top_k=1,
        use_grouped_topk=False,
        renormalize=True,
        custom_routing_function=custom_fn,
    )

    custom_fn.assert_called_once()
    assert torch.equal(topk_weights, torch.full((2, 1), 0.5, dtype=torch.float32))
    assert torch.equal(topk_ids, torch.ones(2, 1, dtype=torch.int32))


@pytest.mark.unit
def test_apply_slices_and_gathers_on_all2all(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 3.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 4.0))
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=None,
    )
    output = method.apply(
        layer=layer,
        hidden_states=torch.arange(12, dtype=torch.float32).view(3, 4),
        router_logits=torch.zeros(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert output.shape == (3, 4)
    assert method.apply_prepare_permute.call_args.args[2].shape == (2, 4)


@pytest.mark.unit
def test_apply_with_gate_and_shared_experts_adds_plugin_output(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.select_communication_strategy = MagicMock(return_value=("agrs", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(module, "named_stream", lambda _name: _DummyStream())
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: _DummyStream(), raising=False)
    monkeypatch.setattr(module.torch.npu, "stream", _stream_ctx, raising=False)
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    def _gate(x):
        return torch.zeros(x.shape[0], 2, dtype=torch.float32), None

    shared = lambda x: torch.full_like(x, 5.0)
    shared.gate_up_proj = SimpleNamespace(tp_size=1)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=_gate,
        shared_experts=shared,
    )
    output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4, dtype=torch.float32),
        router_logits=None,
        top_k=1,
        renormalize=False,
    )
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert torch.equal(shared_output, torch.full((2, 4), 5.0))
    assert torch.equal(routed_output, torch.full((2, 4), 8.0))


@pytest.mark.unit
def test_apply_shared_experts_reduce_branch(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 1.0))
    monkeypatch.setattr(module, "named_stream", lambda _name: _DummyStream())
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: _DummyStream(), raising=False)
    monkeypatch.setattr(module.torch.npu, "stream", _stream_ctx, raising=False)
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    all_reduce_mock = MagicMock(return_value=torch.full((2, 4), 6.0))
    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", all_reduce_mock)
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    shared = MagicMock(return_value=torch.full((3, 4), 6.0))
    shared.gate_up_proj = SimpleNamespace(tp_size=2)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=shared,
    )
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=torch.zeros(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert isinstance(output, tuple)
    all_reduce_mock.assert_called_once()
    assert shared.call_count == 1
    assert torch.equal(shared.call_args.args[0], hidden_states)


@pytest.mark.unit
def test_weight_loader_handles_transposed_weights(layer_module, monkeypatch):
    module, _, _ = layer_module
    super_weight_loader = MagicMock(return_value=True)
    monkeypatch.setattr(module.FusedMoE, "weight_loader", super_weight_loader, raising=False)
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)

    param = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).view(2, 4, 4))
    setattr(param, "is_weight_transposed", True)
    original = param.data.clone()

    result = fused.weight_loader(
        param=param,
        loaded_weight=torch.zeros(2, 3, 4),
        weight_name="w13_weight",
        shard_id="0",
        expert_id=0,
        return_success=True,
    )

    assert result is True
    assert param.shape == original.shape
    assert super_weight_loader.call_count == 1


@pytest.mark.unit
def test_forward_uses_expert_mask_when_rocm_enabled(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.quant_method = SimpleNamespace(apply=MagicMock(return_value=torch.ones(1, 2)))
    fused.top_k = 2
    fused.renormalize = False
    fused.use_grouped_topk = False
    fused.global_num_experts = 4
    fused.expert_map = torch.tensor([0, 1])
    fused.expert_mask = torch.tensor([1, 0])
    fused.rocm_aiter_fmoe_enabled = True
    fused.topk_group = None
    fused.num_expert_group = None
    fused.custom_routing_function = None
    fused.scoring_func = "softmax"
    fused.routed_scaling_factor = 1.0
    fused.e_score_correction_bias = None
    fused.activation = "silu"
    fused.apply_router_weight_on_input = False
    fused.enable_eplb = False

    out = fused.forward(
        hidden_states=torch.ones(1, 2, dtype=torch.float32),
        router_logits=torch.zeros(1, 4, dtype=torch.float32),
    )

    assert torch.equal(out, torch.ones(1, 2))
    kwargs = fused.quant_method.apply.call_args.kwargs
    assert torch.equal(kwargs["expert_map"], fused.expert_mask)


@pytest.mark.unit
def test_unquantized_method_init_sets_tp_info(layer_module, monkeypatch):
    module, _, _ = layer_module
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod,
        "__init__",
        lambda self, moe: setattr(self, "_moe", moe),
        raising=False,
    )
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_tensor_model_parallel_rank = lambda: 2

    method = module.NPUUnquantizedFusedMoEMethod(SimpleNamespace())
    assert method.tp_size == 4
    assert method.tp_rank == 2


@pytest.mark.unit
def test_apply_slice_path_without_padding_slices_router_logits(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    select_mock = MagicMock(
        return_value=(
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(module.NPUFusedMoE, "select_experts", select_mock)

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=None,
    )
    router_logits = torch.arange(16, dtype=torch.float32).view(4, 4)
    out = method.apply(
        layer=layer,
        hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )

    assert out.shape == (4, 4)
    assert select_mock.call_args.kwargs["router_logits"].shape == (2, 4)


@pytest.mark.unit
def test_process_weights_after_loading_transposes_and_marks(layer_module, monkeypatch):
    module, torch_npu, _ = layer_module
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=torch.nn.Parameter(torch.randn(2, 5, 6)),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 3)
    assert layer.w2_weight.shape == (2, 6, 5)
    assert getattr(layer.w13_weight, "is_weight_transposed", False) is True
    assert getattr(layer.w2_weight, "is_weight_transposed", False) is True
    assert torch_npu.npu_format_cast.call_count == 2


@pytest.mark.unit
def test_fused_moe_init_sets_gate_and_strategy_selector(layer_module, monkeypatch):
    module, _, _ = layer_module
    selector_mock = MagicMock()

    def _fake_super_init(self, *args, **kwargs):
        self.quant_method = SimpleNamespace(make_communication_strategy_selector=selector_mock)

    monkeypatch.setattr(module.FusedMoE, "__init__", _fake_super_init, raising=False)
    gate_obj = object()
    fused = module.NPUFusedMoE(gate=gate_obj)

    assert fused.gate is gate_obj
    assert fused.is_internal_router is True
    selector_mock.assert_called_once_with(fused)


@pytest.mark.unit
def test_weight_loader_handles_non_full_load_transposed_branch(layer_module, monkeypatch):
    module, _, _ = layer_module
    super_weight_loader = MagicMock(return_value=True)
    monkeypatch.setattr(module.FusedMoE, "weight_loader", super_weight_loader, raising=False)
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)

    param = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).view(2, 4, 4))
    setattr(param, "is_weight_transposed", True)
    result = fused.weight_loader(
        param=param,
        loaded_weight=torch.zeros(3, 4),
        weight_name="w2_weight",
        shard_id="0",
        expert_id=1,
        return_success=True,
    )

    assert result is True
    assert param.shape == (2, 4, 4)
    assert super_weight_loader.call_count == 1


@pytest.mark.unit
def test_maybe_init_modular_kernel_returns_none(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    assert fused.maybe_init_modular_kernel() is None


@pytest.mark.unit
def test_select_experts_grouped_topk_requires_num_expert_group(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}

    with pytest.raises(ValueError, match="num_expert_group is None"):
        module.NPUFusedMoE.select_experts(
            router_logits=torch.zeros(1, 4),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=1,
            num_expert_group=None,
        )


@pytest.mark.unit
def test_select_experts_default_path_with_renormalize(layer_module):
    module, torch_npu, context_holder = layer_module
    context_holder.attn_metadata = {}
    torch_npu.npu_moe_gating_top_k.return_value = (
        torch.tensor([[2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )

    weights, ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(1, 4),
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
    )

    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    assert ids.shape == (1, 2)
