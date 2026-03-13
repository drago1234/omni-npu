# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest
import torch

def _make_dummy_method_cls():
    from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase

    class _DummyMethod(NPUFusedMoEMethodBase):
        def apply_experts(self, layer, prepare_permute_result, activation="silu"):
            return prepare_permute_result.hidden_states_sorted_by_experts

    return _DummyMethod


class _DummyStrategy:
    def __init__(self):
        self.called = False

    def prepare_permute(self, layer, x, topk_ids):
        self.called = True
        return SimpleNamespace(
            hidden_states_sorted_by_experts=x,
            expert_tokens=torch.tensor([x.shape[0]], dtype=torch.int64),
            dynamic_scale=None,
        )

    def unpermute_finalize(self, layer, hidden_states, topk_ids, topk_weights, result):
        self.called = True
        return hidden_states + 1


@pytest.mark.unit
def test_method_base_delegates_prepare_and_finalize():
    method = _make_dummy_method_cls()()
    strategy = _DummyStrategy()
    x = torch.ones(2, 3)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)

    prepare_result = method.apply_prepare_permute(strategy, SimpleNamespace(), x, topk_ids)
    out = method.apply_unpermute_finalize(
        strategy,
        SimpleNamespace(),
        prepare_result.hidden_states_sorted_by_experts,
        topk_ids,
        topk_weights,
        prepare_result,
    )

    assert strategy.called is True
    assert torch.equal(out, torch.full((2, 3), 2.0))


@pytest.mark.unit
def test_make_communication_strategy_selector_sets_selector(monkeypatch):
    from omni_npu.layers.fused_moe import fused_moe_method_base as base_module

    method = _make_dummy_method_cls()()
    fake_selector = SimpleNamespace(select_communication_strategy=lambda n: ("agrs", object()))
    monkeypatch.setattr(base_module, "CommunicationStrategySelector", lambda _moe: fake_selector)

    method.make_communication_strategy_selector(SimpleNamespace())
    strategy, _ = method.select_communication_strategy(4)

    assert method.communication_strategy_selector is fake_selector
    assert strategy == "agrs"
