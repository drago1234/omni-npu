import importlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def layer_v1_module(monkeypatch):
    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.tensor_model_parallel_all_gather = MagicMock(
        side_effect=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim)
    )
    distributed_module.get_tensor_model_parallel_world_size = MagicMock(return_value=1)
    distributed_module.get_tensor_model_parallel_rank = MagicMock(return_value=0)

    shared_layer_module = types.ModuleType("omni_npu.layers.fused_moe.layer")

    class DummyNPUSharedFusedMoE:
        def __init__(self, *args, **kwargs):
            self.quant_method = MagicMock()
            self.shared_experts = None
            self.top_k = 1
            self.renormalize = False
            self.use_grouped_topk = False
            self.global_num_experts = 0
            self.expert_map = None
            self.rocm_aiter_fmoe_enabled = False
            self.expert_mask = None
            self.topk_group = None
            self.num_expert_group = None
            self.custom_routing_function = None
            self.scoring_func = "softmax"
            self.routed_scaling_factor = 1.0
            self.e_score_correction_bias = None
            self.activation = "silu"
            self.apply_router_weight_on_input = False
            self.enable_eplb = False
            self.expert_load_view = None
            self.logical_to_physical_map = None
            self.logical_replica_count = None

        def ensure_moe_quant_config_init(self):
            return None

    shared_layer_module.NPUSharedFusedMoE = DummyNPUSharedFusedMoE

    prefetch_module = types.ModuleType("omni_npu.v1.layers.prefetch")

    class DummyPrefetcherBase:
        def __init__(self, *args, **kwargs):
            return None

        def prefetch_moe(self, *args, **kwargs):
            return None

        def prefetch_attention(self, *args, **kwargs):
            return None

    prefetch_module.PrefetcherBase = DummyPrefetcherBase

    loader_module = types.ModuleType("omni_npu.v1.models.config_loader.loader")
    loader_module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(enable_prefetch=True)
    )

    prepare_module = types.ModuleType(
        "omni_npu.v1.layers.fused_moe.fused_moe_prepare_permute_unpermute_finalize"
    )

    class DummyAll2All:
        def __init__(self, layer):
            self.layer = layer

    class DummyDispatchCombine:
        def __init__(self, layer):
            self.layer = layer

    prepare_module.All2AllPrepPmtAndUnpmtFinal = DummyAll2All
    prepare_module.DispatchCombinePrepPmtAndUnpmtFinal = DummyDispatchCombine

    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "omni_npu.layers.fused_moe.layer", shared_layer_module)
    monkeypatch.setitem(sys.modules, "omni_npu.v1.layers.prefetch", prefetch_module)
    monkeypatch.setitem(sys.modules, "omni_npu.v1.models.config_loader.loader", loader_module)
    monkeypatch.setitem(
        sys.modules,
        "omni_npu.v1.layers.fused_moe.fused_moe_prepare_permute_unpermute_finalize",
        prepare_module,
    )

    module_name = "omni_npu.v1.layers.fused_moe.layer"
    sys.modules.pop(module_name, None)
    module_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omni_npu"
        / "v1"
        / "layers"
        / "fused_moe"
        / "layer.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    stubs = SimpleNamespace(
        distributed=distributed_module,
        loader=loader_module,
        prepare_module=prepare_module,
    )
    return module, stubs


@pytest.mark.unit
def test_make_prepare_permute_and_unpermute_finalize_sets_handlers(layer_v1_module):
    module, stubs = layer_v1_module
    layer = module.NPUFusedMoEV1.__new__(module.NPUFusedMoEV1)
    layer.quant_method = MagicMock()

    layer.make_prepare_permute_and_unpermute_finalize()

    layer.quant_method.set_prepare_permute_and_unpermute_finalize.assert_called_once()
    kwargs = layer.quant_method.set_prepare_permute_and_unpermute_finalize.call_args.kwargs
    assert isinstance(
        kwargs["prefill_prepare_permute_and_unpermute_finalize"],
        stubs.prepare_module.All2AllPrepPmtAndUnpmtFinal,
    )
    assert isinstance(
        kwargs["decode_prepare_permute_and_unpermute_finalize"],
        stubs.prepare_module.DispatchCombinePrepPmtAndUnpmtFinal,
    )


@pytest.mark.unit
def test_forward_prefetch_and_shared_experts(layer_v1_module):
    module, stubs = layer_v1_module
    stubs.distributed.get_tensor_model_parallel_world_size.return_value = 1

    layer = module.NPUFusedMoEV1.__new__(module.NPUFusedMoEV1)
    layer.top_k = 1
    layer.renormalize = False
    layer.use_grouped_topk = False
    layer.global_num_experts = 2
    layer.expert_map = None
    layer.rocm_aiter_fmoe_enabled = False
    layer.expert_mask = None
    layer.topk_group = None
    layer.num_expert_group = None
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.routed_scaling_factor = 1.0
    layer.e_score_correction_bias = None
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.enable_eplb = False
    layer.expert_load_view = None
    layer.logical_to_physical_map = None
    layer.logical_replica_count = None

    hidden_states = torch.ones(2, 3)
    router_logits = torch.full((2, 2), 4.0)
    layer.gate = MagicMock(return_value=(router_logits, None))
    layer.shared_experts = MagicMock(return_value=torch.full((2, 3), 5.0))
    layer.quant_method = MagicMock()
    layer.quant_method.apply = MagicMock(return_value=torch.full((2, 3), 6.0))
    layer.prefetch_moe = MagicMock()
    layer.prefetch_attention = MagicMock()

    share_output, expert_output = layer.forward(hidden_states, router_logits)

    assert torch.equal(share_output, torch.full((2, 3), 5.0))
    assert torch.equal(expert_output, torch.full((2, 3), 6.0))
    layer.prefetch_moe.assert_any_call(
        trigger=hidden_states,
        prefetch_experts=False,
        prefetch_shared_experts=True,
    )
    layer.prefetch_moe.assert_any_call(
        trigger=hidden_states,
        prefetch_experts=True,
        prefetch_shared_experts=False,
    )
    layer.prefetch_attention.assert_called_once_with(trigger=hidden_states)
    stubs.distributed.tensor_model_parallel_all_gather.assert_not_called()


@pytest.mark.unit
def test_forward_tp_padding_and_all_gather(layer_v1_module):
    module, stubs = layer_v1_module
    stubs.distributed.get_tensor_model_parallel_world_size.return_value = 2
    stubs.distributed.get_tensor_model_parallel_rank.return_value = 1

    layer = module.NPUFusedMoEV1.__new__(module.NPUFusedMoEV1)
    layer.top_k = 1
    layer.renormalize = False
    layer.use_grouped_topk = False
    layer.global_num_experts = 2
    layer.expert_map = None
    layer.rocm_aiter_fmoe_enabled = False
    layer.expert_mask = None
    layer.topk_group = None
    layer.num_expert_group = None
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.routed_scaling_factor = 1.0
    layer.e_score_correction_bias = None
    layer.activation = "silu"
    layer.apply_router_weight_on_input = False
    layer.enable_eplb = False
    layer.expert_load_view = None
    layer.logical_to_physical_map = None
    layer.logical_replica_count = None

    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    router_logits = torch.arange(6, dtype=torch.float32).view(3, 2)
    layer.gate = MagicMock(return_value=(router_logits, None))
    layer.shared_experts = None
    layer.quant_method = MagicMock()
    layer.quant_method.apply = MagicMock(return_value=torch.ones(2, 2))
    layer.prefetch_moe = MagicMock()
    layer.prefetch_attention = MagicMock()

    share_output, expert_output = layer.forward(hidden_states, router_logits)

    expected_padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, 1), value=0)
    expected_local = expected_padded[2:4]
    passed_x = layer.quant_method.apply.call_args.kwargs["x"]
    passed_router = layer.quant_method.apply.call_args.kwargs["router_logits"]
    assert torch.equal(passed_x, expected_local)
    assert torch.equal(passed_router, expected_padded[2:4])
    assert share_output is None
    assert expert_output.shape == (3, 2)
    stubs.distributed.tensor_model_parallel_all_gather.assert_called_once()

