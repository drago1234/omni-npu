import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

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
def rejection_mod(monkeypatch):
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
    setattr(
        fake_torch_npu,
        "_C",
        types.SimpleNamespace(_NPUTaskGroupHandle=object),
    )
    setattr(fake_aiter_ops, "rocm_aiter_ops", None)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            default_stream=lambda: fake_default_stream,
            Stream=object,
            NPUGraph=object,
            ExternalEvent=object,
        ),
        raising=False,
    )

    import omni_npu.sample.rejection_sampler as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(
        mod.torch,
        "npu",
        types.SimpleNamespace(default_stream=lambda: fake_default_stream),
        raising=False,
    )
    return mod, fake_torch_npu, fake_default_stream


def test_expand_batch_to_tokens_with_replacement(rejection_mod):
    mod, _, _ = rejection_mod
    x = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    cu = torch.tensor([2, 5, 6], dtype=torch.int64)

    out = mod.expand_batch_to_tokens(x, cu, num_tokens=6, replace_from=0, replace_to=9)

    expected = torch.tensor([9.0, 9.0, 2.0, 2.0, 2.0, 9.0], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_compute_probs_returns_logits_for_all_greedy(rejection_mod):
    mod, _, _ = rejection_mod
    logits = torch.randn(3, 4)
    metadata = SimpleNamespace(all_greedy=True, temperature=None, top_k=None, top_p=None)

    out = mod.compute_probs(logits, cu_num_draft_tokens=torch.tensor([1, 3]), sampling_metadata=metadata)

    assert out is logits


def test_compute_probs_applies_temperature_and_topk_topp(rejection_mod, monkeypatch):
    mod, _, _ = rejection_mod
    captured = {}

    def fake_apply(logits, k, p):
        captured["logits"] = logits.clone()
        captured["k"] = k.clone()
        captured["p"] = p.clone()
        return logits + 3

    monkeypatch.setattr(mod, "apply_top_k_top_p_npu", fake_apply)

    logits = torch.tensor([[4.0, 2.0], [10.0, 6.0], [8.0, 4.0]], dtype=torch.float32)
    metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([mod.GREEDY_TEMPERATURE, 2.0], dtype=torch.float32),
        top_k=torch.tensor([1, 2], dtype=torch.int64),
        top_p=torch.tensor([0.7, 0.9], dtype=torch.float32),
    )
    cu = torch.tensor([1, 3], dtype=torch.int64)

    out = mod.compute_probs(logits, cu_num_draft_tokens=cu, sampling_metadata=metadata)

    expected_after_div = torch.tensor([[4.0, 2.0], [5.0, 3.0], [4.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(captured["logits"], expected_after_div)
    assert torch.equal(captured["k"], torch.tensor([1, 2, 2], dtype=torch.int64))
    assert torch.allclose(captured["p"], torch.tensor([0.7, 0.9, 0.9], dtype=torch.float32))
    assert torch.allclose(out, expected_after_div + 3)


def test_sample_recovered_tokens_native_without_draft_probs(rejection_mod):
    mod, _, _ = rejection_mod
    recovered = torch.empty((2,), dtype=torch.int32)
    cu = torch.tensor([2], dtype=torch.int64)
    draft_ids = torch.tensor([0, 1], dtype=torch.int64)
    target_probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.6, 0.0]], dtype=torch.float32)
    q = torch.ones((1, 3), dtype=torch.float32)

    mod.sample_recovered_tokens_native(
        recovered_token_ids=recovered,
        cu_num_draft_tokens=cu,
        draft_token_ids=draft_ids,
        draft_probs=None,
        target_probs=target_probs,
        q=q,
        vocab_size=3,
        PADDED_VOCAB_SIZE=4,
        NO_DRAFT_PROBS=True,
    )

    assert torch.equal(recovered, torch.tensor([2, 0], dtype=torch.int32))


def test_select_tokens_by_accepted_with_fill_mask(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((2, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    accepted = torch.tensor([True, False, True], dtype=torch.bool)
    cu = torch.tensor([2, 3], dtype=torch.int64)
    draft = torch.tensor([10, 11, 12], dtype=torch.int32)
    recovered = torch.tensor([20, 21, 22], dtype=torch.int32)
    bonus = torch.tensor([[30], [31]], dtype=torch.int32)
    fill_this_time = torch.tensor([True, False], dtype=torch.bool)

    mod.select_tokens_by_accepted(
        output_token_ids=output,
        accepted=accepted,
        cu_num_draft_tokens=cu,
        draft_token_ids=draft,
        recovered_token_ids=recovered,
        bonus_token_ids=bonus,
        fill_this_time=fill_this_time,
        max_spec_len=2,
    )

    expected = torch.tensor([[10, 21, mod.PLACEHOLDER_TOKEN_ID], [mod.PLACEHOLDER_TOKEN_ID] * 3], dtype=torch.int32)
    assert torch.equal(output, expected)


def test_rejection_greedy_sample_native_basic(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)

    mod.rejection_greedy_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        target_argmax=torch.tensor([1, 5], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        is_greedy=None,
        max_spec_len=2,
    )

    assert torch.equal(output, torch.tensor([[1, 5, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_rejection_random_sample_native_with_draft_probs(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    target_probs = torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32)
    draft_probs = torch.tensor([[0.95, 0.05], [0.05, 0.95]], dtype=torch.float32)

    mod.rejection_random_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([0, 1], dtype=torch.int32),
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        recovered_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        uniform_probs=torch.tensor([0.5, 0.95], dtype=torch.float32),
        is_greedy=torch.tensor([False]),
        max_spec_len=2,
        vocab_size=2,
        NO_DRAFT_PROBS=False,
    )

    assert torch.equal(output, torch.tensor([[0, 4, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_simple_verify_returns_expected_shape(rejection_mod):
    mod, _, _ = rejection_mod
    out = mod.simple_verify(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        target_token_ids=torch.tensor([1, 3], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
    )

    expected = torch.tensor([[1, 3, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    assert torch.equal(out, expected)


def test_generate_random_sequence_respects_seeded_generators(rejection_mod):
    mod, _, _ = rejection_mod
    probs = torch.zeros((3, 4), dtype=torch.float32)
    spec_meta = SimpleNamespace(num_draft_tokens=[1, 0])

    g0 = torch.Generator(device=probs.device.type).manual_seed(42)
    out1 = mod.generate_random_sequence(
        probs=probs,
        sampling_metadata=SimpleNamespace(generators={0: g0}),
        spec_metadata=spec_meta,
    )
    g0 = torch.Generator(device=probs.device.type).manual_seed(42)
    out2 = mod.generate_random_sequence(
        probs=probs,
        sampling_metadata=SimpleNamespace(generators={0: g0}),
        spec_metadata=spec_meta,
    )

    assert out1.shape == probs.shape
    assert torch.allclose(out1[:2], out2[:2])


def test_compute_probs_and_sample_all_greedy_path(rejection_mod):
    mod, _, _ = rejection_mod
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
    token_ids, out_logits = mod.compute_probs_and_sample(
        logits=logits.clone(),
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        sampling_metadata=SimpleNamespace(all_greedy=True),
        metadata=SimpleNamespace(),
        stream=object(),
    )

    assert torch.equal(token_ids, torch.tensor([1, 0], dtype=torch.int32))
    assert torch.allclose(out_logits, logits)


def test_compute_probs_and_sample_non_greedy_calls_npu(rejection_mod, monkeypatch):
    mod, fake_torch_npu, fake_default_stream = rejection_mod
    called = {}

    monkeypatch.setattr(
        mod,
        "generate_random_sequence",
        lambda probs, sampling_metadata, metadata: torch.ones_like(probs, dtype=torch.float32),
    )

    def fake_sample(logits, top_k, top_p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["top_k"] = top_k.clone()
        called["top_p"] = top_p.clone()
        called["q"] = q.clone()
        called["is_need_logits"] = is_need_logits
        return torch.tensor([1, 0], dtype=torch.int32), logits + 2

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    stream = object()
    logits = torch.tensor([[4.0, 2.0], [6.0, 3.0]], dtype=torch.float32)
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([1.0], dtype=torch.float32),
        top_k=None,
        top_p=None,
        generators={},
    )
    metadata = SimpleNamespace(num_draft_tokens=[1])

    token_ids, out_logits = mod.compute_probs_and_sample(
        logits=logits,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        sampling_metadata=sampling_metadata,
        metadata=metadata,
        stream=stream,
    )

    assert called["logits_dtype"] == torch.bfloat16
    assert torch.equal(called["top_k"], torch.tensor([2, 2], dtype=torch.int32))
    assert torch.all(called["top_p"] == 1)
    assert called["q"].dtype == torch.float32
    assert called["is_need_logits"] is True
    assert fake_default_stream.waited_streams[-1] is stream
    assert torch.equal(token_ids, torch.tensor([1, 0], dtype=torch.int32))
    assert out_logits.dtype == torch.bfloat16


def test_rejection_sample_all_greedy_short_circuit(rejection_mod):
    mod, _, _ = rejection_mod
    output = mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 1], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_probs=None,
        target_probs=torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=True,
            all_random=False,
            temperature=torch.tensor([mod.GREEDY_TEMPERATURE], dtype=torch.float32),
            generators={},
        ),
        stream=object(),
    )

    assert torch.equal(output, torch.tensor([[0, mod.PLACEHOLDER_TOKEN_ID, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_rejection_sample_random_path_invokes_helpers(rejection_mod, monkeypatch):
    mod, _, _ = rejection_mod
    called = {}

    monkeypatch.setattr(
        mod,
        "generate_uniform_probs",
        lambda *args, **kwargs: torch.tensor([0.2, 0.9], dtype=torch.float32),
    )

    def fake_sample_recovered_tokens(*args, **kwargs):
        called["sample_recovered_tokens_called"] = True
        return torch.tensor([4, 5], dtype=torch.int32)

    monkeypatch.setattr(mod, "sample_recovered_tokens", fake_sample_recovered_tokens)

    def fake_rejection_random_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS,
    ):
        called["random_called"] = True
        called["no_draft_probs"] = NO_DRAFT_PROBS
        output_token_ids[:] = torch.tensor([[4, 5, 6]], dtype=torch.int32)

    monkeypatch.setattr(mod, "rejection_random_sample_native", fake_rejection_random_sample_native)

    out = mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_probs=None,
        target_probs=torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([0.7], dtype=torch.float32),
            generators={},
        ),
        stream=object(),
    )

    assert called["sample_recovered_tokens_called"] is True
    assert called["random_called"] is True
    assert called["no_draft_probs"] is True
    assert torch.equal(out, torch.tensor([[4, 5, 6]], dtype=torch.int32))

