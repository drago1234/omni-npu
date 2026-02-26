import importlib
import importlib.machinery
import sys
import types

import pytest
import torch


class _DummyStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyDefaultStream:
    def __init__(self):
        self.waited_streams = []

    def wait_stream(self, stream):
        self.waited_streams.append(stream)


@pytest.fixture
def topk_mod(monkeypatch):
    fake_default_stream = _DummyDefaultStream()
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_aiter_ops = types.ModuleType("vllm._aiter_ops")
    fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    fake_aiter_ops.__spec__ = importlib.machinery.ModuleSpec("vllm._aiter_ops", loader=None)
    def _fallback_attr(name):
        if name.startswith("npu_"):
            return lambda *args, **kwargs: None
        raise AttributeError(f"module 'torch_npu' has no attribute {name}")
    setattr(
        fake_torch_npu,
        "npu",
        types.SimpleNamespace(
            stream=lambda _stream: _DummyStreamCtx(),
            Stream=lambda: "fake_npu_stream",
        ),
    )
    setattr(fake_torch_npu, "__getattr__", _fallback_attr)
    setattr(fake_torch_npu, "npu_top_k_top_p_sample", lambda *args, **kwargs: None)
    setattr(fake_torch_npu, "npu_fusion_attention", lambda *args, **kwargs: None)
    setattr(fake_aiter_ops, "rocm_aiter_ops", None)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)

    import omni_npu.sample.ops.topk_topp_sampler as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(
        mod.torch,
        "npu",
        types.SimpleNamespace(default_stream=lambda: fake_default_stream),
        raising=False,
    )
    return mod, fake_torch_npu, fake_default_stream


def test_apply_top_k_top_p_npu_passthrough_when_k_p_none(topk_mod):
    mod, _, _ = topk_mod
    logits = torch.randn(2, 5, dtype=torch.float32)

    out = mod.apply_top_k_top_p_npu(logits, None, None)

    assert out is logits


def test_apply_top_k_top_p_npu_default_inputs_and_dtype_conversion(topk_mod):
    mod, fake_torch_npu, _ = topk_mod
    called = {}

    def fake_sample(logits, k, p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["k"] = k
        called["p"] = p
        called["q"] = q
        called["is_need_logits"] = is_need_logits
        return torch.zeros((logits.shape[0],), dtype=torch.int32), logits + 1

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    logits = torch.randn(3, 7, dtype=torch.float32)
    top_p = torch.tensor([0.8, 0.9, 1.0], dtype=torch.float32)

    out = mod.apply_top_k_top_p_npu(logits, k=None, p=top_p)

    assert out.dtype == torch.bfloat16
    assert called["logits_dtype"] == torch.bfloat16
    assert called["k"].dtype == torch.int32
    assert torch.equal(called["k"], torch.tensor([7, 7, 7], dtype=torch.int32))
    assert called["p"].dtype == torch.bfloat16
    assert torch.allclose(called["p"].float(), top_p, atol=2e-3, rtol=0)
    assert called["q"] is None
    assert called["is_need_logits"] is True


def test_generate_coins_waits_stream_and_supports_seeded_generators(topk_mod):
    mod, _, fake_default_stream = topk_mod
    probs = torch.zeros((2, 4), dtype=torch.bfloat16)
    stream = object()

    g0 = torch.Generator(device=probs.device.type).manual_seed(11)
    g1 = torch.Generator(device=probs.device.type).manual_seed(22)
    q1 = mod.generate_coins(probs, {0: g0, 1: g1}, stream)

    g0 = torch.Generator(device=probs.device.type).manual_seed(11)
    g1 = torch.Generator(device=probs.device.type).manual_seed(22)
    q2 = mod.generate_coins(probs, {0: g0, 1: g1}, stream)

    assert q1.dtype == torch.float32
    assert q1.shape == probs.shape
    assert torch.allclose(q1, q2)
    assert fake_default_stream.waited_streams[-1] is stream


@pytest.mark.parametrize(
    "logprobs_mode,expect_none",
    [
        ("raw_logprobs", True),
        ("processed_logits", False),
        ("processed_logprobs", False),
    ],
)
def test_forward_npu_handles_logprobs_modes(topk_mod, monkeypatch, logprobs_mode, expect_none):
    mod, fake_torch_npu, _ = topk_mod

    fake_logits_out = torch.tensor([[2.0, 1.0], [0.5, 0.5]], dtype=torch.bfloat16)
    fake_token_ids = torch.tensor([0, 1], dtype=torch.int32)
    fake_torch_npu.npu_top_k_top_p_sample = (
        lambda logits, k, p, q, is_need_logits: (fake_token_ids, fake_logits_out)
    )
    monkeypatch.setattr(
        mod,
        "generate_coins",
        lambda probs, generators, stream: torch.ones_like(probs, dtype=torch.float32),
    )

    sampler = mod.NPUTopKTopPSampler.__new__(mod.NPUTopKTopPSampler)
    sampler.logprobs_mode = logprobs_mode
    sampler.dsa_stream = object()

    logits = torch.randn(2, 2, dtype=torch.float32)
    token_ids, logits_to_return = sampler.forward_npu(logits, generators={}, k=None, p=None)

    assert torch.equal(token_ids, fake_token_ids)
    if expect_none:
        assert logits_to_return is None
    elif logprobs_mode == "processed_logits":
        assert torch.equal(logits_to_return, fake_logits_out)
    else:
        expected = fake_logits_out.log_softmax(dim=-1, dtype=torch.float32)
        assert torch.allclose(logits_to_return, expected)


def test_sampler_init_sets_npu_specific_attributes(topk_mod, monkeypatch):
    mod, fake_torch_npu, _ = topk_mod
    parent_called = {}

    def fake_parent_init(self, logprobs_mode):
        parent_called["logprobs_mode"] = logprobs_mode

    monkeypatch.setattr(mod.V1TopKTopPSampler, "__init__", fake_parent_init)
    fake_torch_npu.npu.Stream = lambda: "stream_from_npu"

    sampler = mod.NPUTopKTopPSampler(logprobs_mode="processed_logits")

    assert parent_called["logprobs_mode"] == "processed_logits"
    assert sampler.apply_top_k_top_p is mod.apply_top_k_top_p_npu
    assert sampler.forward == sampler.forward_npu
    assert sampler.dsa_stream == "stream_from_npu"

