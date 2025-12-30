# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def layer_module(monkeypatch):
    """Provide the fused MoE layer module with all heavy dependencies mocked."""
    torch_npu = MagicMock()
    torch_npu.npu_format_cast.side_effect = lambda tensor, fmt: tensor
    torch_npu.npu_swiglu.side_effect = lambda tensor: tensor

    def _grouped_matmul(inputs, weights, **kwargs):
        return [inputs[0]]

    torch_npu.npu_grouped_matmul.side_effect = _grouped_matmul
    torch_npu.npu_moe_gating_top_k.side_effect = (
        lambda logits, k, **kwargs: (
            torch.ones(logits.shape[:-1] + (k,), device=logits.device),
            torch.zeros(logits.shape[:-1] + (k,), dtype=torch.int32, device=logits.device),
            torch.arange(logits.shape[0] * k, device=logits.device, dtype=torch.int32).view(logits.shape[0], k),
        )
    )
    torch_npu.npu_moe_gating_top_k_softmax.side_effect = (
        lambda logits, k, **kwargs: (
            torch.ones(logits.shape[:-1] + (k,), device=logits.device),
            torch.zeros(logits.shape[:-1] + (k,), dtype=torch.int32, device=logits.device),
            torch.arange(logits.shape[0] * k, device=logits.device, dtype=torch.int32).view(logits.shape[0], k),
        )
    )

    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(config=SimpleNamespace()))
    elif not hasattr(torch.npu, "config"):
        monkeypatch.setattr(torch.npu, "config", SimpleNamespace())
    torch.npu.config.allow_internal_format = False

    vllm_module = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: MagicMock()

    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: SimpleNamespace(rank=0)
    distributed_module.tensor_model_parallel_all_reduce = MagicMock(side_effect=lambda tensor: tensor)
    distributed_module.tensor_model_parallel_all_gather = MagicMock(
        side_effect=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim)
    )
    distributed_module.get_tensor_model_parallel_world_size = MagicMock(return_value=1)
    distributed_module.get_tensor_model_parallel_rank = MagicMock(return_value=0)

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    context_holder = SimpleNamespace(attn_metadata=None)
    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: context_holder

    model_executor_module = types.ModuleType("vllm.model_executor")
    model_executor_module.__path__ = []
    layers_module = types.ModuleType("vllm.model_executor.layers")
    layers_module.__path__ = []
    fused_moe_pkg = types.ModuleType("vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []

    fused_layer_base = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")

    class DummyFusedMoE:
        register_oot = classmethod(lambda cls, sub: sub)

        def __init__(self):
            self.moe_parallel_config = SimpleNamespace(use_ep=False)
            self.quant_method = MagicMock()

        def ensure_moe_quant_config_init(self):
            return None

        def _maybe_init_expert_routing_tables(self):
            return "routing"

    class DummyUnquantizedFusedMoEMethod:
        register_oot = classmethod(lambda cls, sub: sub)

        def __init__(self):
            self.moe = MagicMock(moe_parallel_config=SimpleNamespace(use_ep=True))
            self.moe_quant_config = MagicMock()
            self.fused_experts = MagicMock()

        def process_weights_after_loading(self, layer):
            self.processed = True

    fused_layer_base.FusedMoE = DummyFusedMoE
    fused_layer_base.UnquantizedFusedMoEMethod = DummyUnquantizedFusedMoEMethod

    modular_kernel_module = types.ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")

    class DummyPrepareFinalize:
        pass

    class DummyPermuteExpertsUnpermute:
        pass

    modular_kernel_module.FusedMoEPrepareAndFinalize = DummyPrepareFinalize
    modular_kernel_module.FusedMoEPermuteExpertsUnpermute = DummyPermuteExpertsUnpermute

    fused_moe_modular_method_module = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe_modular_method")

    class DummyFusedMoEModularMethod:
        register_oot = classmethod(lambda cls, sub: sub)

        def __init__(self, old_quant_method=None, experts=None):
            self.old_quant_method = old_quant_method
            self.experts = experts
            self.fused_experts = getattr(old_quant_method, "fused_experts", MagicMock())

        @classmethod
        def make(cls, fused_moe, old_quant_method, prepare_finalize, unused):
            inst = cls(old_quant_method, None)
            inst.prepare_finalize = prepare_finalize
            inst.fused_moe = fused_moe
            return inst

    fused_moe_modular_method_module.FusedMoEModularMethod = DummyFusedMoEModularMethod

    shared_fused_moe_module = types.ModuleType("vllm.model_executor.layers.fused_moe.shared_fused_moe")

    class DummySharedFusedMoE:
        register_oot = classmethod(lambda cls, sub: sub)

    shared_fused_moe_module.SharedFusedMoE = DummySharedFusedMoE

    fused_moe_module = types.ModuleType("omni_npu.layers.fused_moe.fused_moe")
    fused_experts_tp = MagicMock(return_value=torch.tensor([42.0]))
    moe_infer_fusion = MagicMock(return_value=torch.tensor([24.0]))
    fused_moe_module.fused_experts_tp = fused_experts_tp
    fused_moe_module.moe_infer_fusion = moe_infer_fusion

    npu_prepare_module = types.ModuleType("omni_npu.layers.fused_moe.npu_moe_prepare_finalize")

    class DummyNpuPrepareAndFinalize:
        def __init__(self, moe):
            self.moe = moe

    npu_prepare_module.NpuMoEPrepareAndFinalize = DummyNpuPrepareAndFinalize

    npu_permute_module = types.ModuleType("omni_npu.layers.fused_moe.npu_moe_permute_unpermute")

    class DummyNpuPermuteExpertsUnpermute:
        def __init__(self, moe_quant_config, layer):
            self.moe_quant_config = moe_quant_config
            self.layer = layer

    npu_permute_module.NPUFusedMoEPermuteExpertsUnpermute = DummyNpuPermuteExpertsUnpermute

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.layer", fused_layer_base)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.modular_kernel", modular_kernel_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.fused_moe_modular_method", fused_moe_modular_method_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.shared_fused_moe", shared_fused_moe_module)
    monkeypatch.setitem(sys.modules, "omni_npu.layers.fused_moe.fused_moe", fused_moe_module)
    monkeypatch.setitem(sys.modules, "omni_npu.layers.fused_moe.npu_moe_prepare_finalize", npu_prepare_module)
    monkeypatch.setitem(sys.modules, "omni_npu.layers.fused_moe.npu_moe_permute_unpermute", npu_permute_module)

    for parent, child in (
        (vllm_module, logger_module),
        (vllm_module, platforms_module),
        (vllm_module, distributed_module),
        (vllm_module, forward_context_module),
        (model_executor_module, layers_module),
        (layers_module, fused_moe_pkg),
        (fused_moe_pkg, fused_layer_base),
        (fused_moe_pkg, modular_kernel_module),
        (fused_moe_pkg, fused_moe_modular_method_module),
        (fused_moe_pkg, shared_fused_moe_module),
    ):
        setattr(parent, child.__name__.split(".")[-1], child)

    sys.modules.pop("omni_npu.layers.fused_moe.layer", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.layer")
    importlib.reload(module)

    stubs = SimpleNamespace(
        torch_npu=torch_npu,
        fused_experts_tp=fused_experts_tp,
        moe_infer_fusion=moe_infer_fusion,
        distributed=distributed_module,
        context_holder=context_holder,
    )
    return module, stubs


@pytest.mark.unit
def test_forward_oot_uses_all2all_path(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.moe = MagicMock(moe_parallel_config=SimpleNamespace(use_ep=True))
    method.moe_quant_config = MagicMock()
    method.fused_experts = MagicMock()
    layer = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True), shared_experts=None)

    module.get_tensor_model_parallel_world_size = MagicMock(return_value=2)
    module.get_tensor_model_parallel_rank = MagicMock(return_value=0)
    module.tensor_model_parallel_all_gather = MagicMock(
        side_effect=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim)
    )
    module.NPUFusedMoE.select_experts = MagicMock(
        return_value=(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None)
    )
    module.moe_infer_fusion = MagicMock(return_value=torch.ones(1, 3))
    stubs.context_holder.attn_metadata = None

    hidden = torch.ones(2, 3)
    logits = torch.ones(2, 2)

    output = method.forward_oot(layer, hidden, logits, top_k=1, renormalize=False)

    assert output.shape == (2, 3)
    module.moe_infer_fusion.assert_called_once()
    module.tensor_model_parallel_all_gather.assert_called_once()


@pytest.mark.unit
def test_forward_oot_returns_shared_output_and_reduces(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.moe = MagicMock(moe_parallel_config=SimpleNamespace(use_ep=True))
    method.moe_quant_config = MagicMock()
    method.fused_experts = MagicMock(return_value=torch.ones(1, 2))

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        shared_experts=MagicMock(return_value=torch.full((1, 2), 5.0)),
        w13_weight=None,
        w2_weight=None,
    )

    module.get_tensor_model_parallel_world_size = MagicMock(return_value=2)
    module.get_tensor_model_parallel_rank = MagicMock(return_value=0)
    module.tensor_model_parallel_all_gather = MagicMock(
        side_effect=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim)
    )
    module.tensor_model_parallel_all_reduce = MagicMock(side_effect=lambda tensor: tensor + 1)
    module.NPUFusedMoE.select_experts = MagicMock(
        return_value=(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None)
    )
    stubs.context_holder.attn_metadata = {0: SimpleNamespace(num_prefills=0)}

    hidden = torch.ones(2, 2)
    logits = torch.ones(2, 2)
    share_output, expert_output = method.forward_oot(layer, hidden, logits, top_k=1, renormalize=True)

    assert torch.equal(share_output, torch.full((1, 2), 6.0))
    assert expert_output.shape == (2, 2)
    module.tensor_model_parallel_all_reduce.assert_called_once()
    method.fused_experts.assert_called_once()


@pytest.mark.unit
def test_forward_oot_uses_fused_experts_tp_when_not_ep(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.moe = MagicMock(moe_parallel_config=SimpleNamespace(use_ep=False))
    method.moe_quant_config = MagicMock()

    layer = SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=False))
    module.NPUFusedMoE.select_experts = MagicMock(
        return_value=(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), None)
    )
    stubs.context_holder.attn_metadata = {}
    stubs.fused_experts_tp.reset_mock()

    result = method.forward_oot(layer, torch.ones(1, 2), torch.ones(1, 2), top_k=1, renormalize=False)

    assert torch.equal(result, stubs.fused_experts_tp.return_value)
    stubs.fused_experts_tp.assert_called_once()


@pytest.mark.unit
def test_process_weights_after_loading_transposes_and_casts(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.moe_quant_config = MagicMock()
    method.moe = MagicMock(moe_parallel_config=SimpleNamespace(use_ep=True))

    layer = SimpleNamespace()
    layer.w13_weight = torch.nn.Parameter(torch.ones(2, 4, 3), requires_grad=True)
    layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 3), requires_grad=True)

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 3, 4)
    assert layer.w2_weight.shape == (2, 3, 4)
    assert layer.w13_weight.requires_grad is False
    assert layer.w2_weight.requires_grad is False
    assert stubs.torch_npu.npu_format_cast.call_count == 2


@pytest.mark.unit
def test_gmm_expert_invokes_grouped_matmul_and_swiglu(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        w13_weight=torch.ones(1, 1, 1),
        w2_weight=torch.ones(1, 1, 1),
    )

    stubs.torch_npu.npu_grouped_matmul.reset_mock()
    stubs.torch_npu.npu_swiglu.reset_mock()
    output = method.gmm_expert(layer, torch.ones(1, 1), expert_tokens=[0])

    assert isinstance(output, torch.Tensor)
    stubs.torch_npu.npu_grouped_matmul.assert_called()
    stubs.torch_npu.npu_swiglu.assert_called_once()


@pytest.mark.unit
def test_maybe_all_reduce_tensor_model_parallel(layer_module):
    module, stubs = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.moe_parallel_config = SimpleNamespace(use_ep=True)
    module.tensor_model_parallel_all_reduce = MagicMock()
    tensor = torch.tensor([1.0])

    result_no_reduce = fused.maybe_all_reduce_tensor_model_parallel(tensor)
    assert torch.equal(result_no_reduce, tensor)
    module.tensor_model_parallel_all_reduce.assert_not_called()

    fused.moe_parallel_config.use_ep = False
    module.tensor_model_parallel_all_reduce = MagicMock(return_value=torch.tensor([2.0]))
    reduced = fused.maybe_all_reduce_tensor_model_parallel(tensor)
    module.tensor_model_parallel_all_reduce.assert_called_once_with(tensor)
    assert torch.equal(reduced, torch.tensor([2.0]))


@pytest.mark.unit
def test_maybe_init_modular_kernel_wraps_quant_method(layer_module):
    module, stubs = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.ensure_moe_quant_config_init = MagicMock()
    fused._maybe_init_expert_routing_tables = MagicMock(return_value="routing")
    original_quant = MagicMock(maybe_make_prepare_finalize=MagicMock(return_value="prepare"))
    fused.quant_method = original_quant
    module.FusedMoEModularMethod.make = MagicMock(return_value="wrapped")

    fused.maybe_init_modular_kernel()

    module.FusedMoEModularMethod.make.assert_called_once_with(fused, original_quant, "prepare", None)
    assert fused.quant_method == "wrapped"


@pytest.mark.unit
def test_select_experts_profile_mode(layer_module):
    module, stubs = layer_module
    stubs.context_holder.attn_metadata = None
    module.get_ep_group = MagicMock(return_value=SimpleNamespace(rank=1))
    logits = torch.zeros(2, 4, dtype=torch.float32)

    topk_weights, topk_ids, row_idx = module.NPUFusedMoE.select_experts(
        router_logits=logits,
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
    )

    assert topk_weights.shape == (2, 2)
    assert topk_ids.shape == (2, 2)
    assert row_idx.shape == (2, 2)
    assert torch.all(topk_ids < logits.shape[1])


@pytest.mark.unit
def test_select_experts_grouped_topk(layer_module):
    module, stubs = layer_module
    stubs.context_holder.attn_metadata = {}
    stubs.torch_npu.npu_moe_gating_top_k.reset_mock()
    logits = torch.zeros(1, 3)

    module.NPUFusedMoE.select_experts(
        router_logits=logits,
        top_k=1,
        use_grouped_topk=True,
        renormalize=False,
        topk_group=2,
        num_expert_group=2,
    )

    stubs.torch_npu.npu_moe_gating_top_k.assert_called_once()


@pytest.mark.unit
def test_select_experts_custom_routing(layer_module):
    module, stubs = layer_module
    stubs.context_holder.attn_metadata = {}
    logits = torch.zeros(1, 2)

    def custom_route(gating_output, topk, renormalize):
        return torch.full((1, topk), 0.5), torch.ones((1, topk), dtype=torch.int32), torch.zeros((1, topk), dtype=torch.int32)

    topk_weights, topk_ids, row_idx = module.NPUFusedMoE.select_experts(
        router_logits=logits,
        top_k=1,
        use_grouped_topk=False,
        renormalize=True,
        custom_routing_function=custom_route,
    )

    assert torch.equal(topk_weights, torch.full((1, 1), 0.5))
    assert torch.equal(topk_ids, torch.ones((1, 1), dtype=torch.int32))
    assert torch.equal(row_idx, torch.zeros((1, 1), dtype=torch.int32))


@pytest.mark.unit
def test_maybe_make_prepare_finalize_and_select_gemm_impl(layer_module):
    module, stubs = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.moe = MagicMock()
    method.moe_quant_config = MagicMock()
    prepare = method.maybe_make_prepare_finalize(routing_tables=None)
    assert isinstance(prepare, module.NpuMoEPrepareAndFinalize)

    layer = SimpleNamespace()
    gemm_impl = method.select_gemm_impl(prepare_finalize=prepare, layer=layer)
    assert isinstance(gemm_impl, module.NPUFusedMoEPermuteExpertsUnpermute)


@pytest.mark.unit
def test_npu_fused_moe_modular_method_delegates(layer_module):
    module, stubs = layer_module
    old_quant = MagicMock()
    old_quant.apply = MagicMock(return_value="applied")
    old_quant.gmm_expert = MagicMock(return_value="gmm")

    method = module.NPUFusedMoEModularMethod(old_quant, experts="experts")

    assert method.apply("input") == "applied"
    old_quant.apply.assert_called_once_with("input")
    assert method.gmm_expert("a", "b") == "gmm"
    old_quant.gmm_expert.assert_called_once_with("a", "b")
    assert old_quant.fused_experts is method.fused_experts
