# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import types
import unittest
from unittest import mock


class _TensorIs:
    """Matcher for unittest.mock that compares tensors by identity (is), not equality (==)."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __eq__(self, other):
        return other is self.tensor

    def __repr__(self):  # pragma: no cover
        return f"_TensorIs({type(self.tensor).__name__}@{id(self.tensor)})"


def _ensure_torch_npu_stubs():
    """
    Ensure `torch_npu` exists so we can import the module under test in a
    CPU-only CI environment.
    """
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # pragma: no cover
        # Keep unit suite green in minimal environments; in real CI this should
        # be installed so assertions run.
        raise unittest.SkipTest("PyTorch (`torch`) is not installed; skipping prefetch unit tests.")

    import torch  # type: ignore[import-not-found]  # noqa: E402

    # Provide a fake `torch_npu` module if it isn't installed.
    try:
        import torch_npu  # type: ignore  # noqa: F401
        return sys.modules["torch_npu"]
    except Exception:
        # Minimal stub to allow importing omni_npu modules in CPU-only CI.
        fake_mod = types.ModuleType("torch_npu")
        fake_mod.npu = types.SimpleNamespace()

        def _npu_prefetch(*_args, **_kwargs):
            return None

        fake_mod.npu_prefetch = _npu_prefetch  # type: ignore[attr-defined]
        sys.modules["torch_npu"] = fake_mod
        return fake_mod


def _import_prefetch_module():
    """
    Import the prefetch module in a way that works both:
    - when the package is installed (import path `omni_npu...`)
    - when running from source tree without installation (`src.omni_npu...`)
    """
    _ensure_torch_npu_stubs()
    try:
        return importlib.import_module("omni_npu.v1.layers.prefetch")
    except Exception:
        return importlib.import_module("src.omni_npu.v1.layers.prefetch")


def _reload_prefetch_module():
    """
    Reload the module under test so module-level state resets between test cases.
    """
    mod = _import_prefetch_module()
    return importlib.reload(mod)


class TestPrefetchWeight(unittest.TestCase):
    def setUp(self):
        _ensure_torch_npu_stubs()
        self.prefetch = _reload_prefetch_module()

        import torch  # type: ignore[import-not-found]  # noqa: E402

        self.weight = torch.randn(2, 3)
        self.trigger = torch.randn(1)

    def test_prefetch_weight_noop_on_none_inputs_or_non_positive_prefetch(self):
        with mock.patch("torch_npu.npu_prefetch", autospec=True) as npu_prefetch:
            # weight is None
            self.prefetch.PrefetcherBase.prefetch_weight(None, self.trigger, 16)
            # trigger is None
            self.prefetch.PrefetcherBase.prefetch_weight(self.weight, None, 16)
            # prefetch is None
            self.prefetch.PrefetcherBase.prefetch_weight(self.weight, self.trigger, None)
            # prefetch <= 0
            self.prefetch.PrefetcherBase.prefetch_weight(self.weight, self.trigger, 0)
            self.prefetch.PrefetcherBase.prefetch_weight(self.weight, self.trigger, -1)

            npu_prefetch.assert_not_called()

    def test_prefetch_weight_calls_torch_npu_prefetch(self):
        with mock.patch("torch_npu.npu_prefetch", autospec=True) as npu_prefetch:
            self.prefetch.PrefetcherBase.prefetch_weight(self.weight, self.trigger, 128)
            npu_prefetch.assert_called_once_with(
                _TensorIs(self.weight),
                _TensorIs(self.trigger),
                128 * (2**20),
            )


class TestPrefetcherBaseMethods(unittest.TestCase):
    def setUp(self):
        _ensure_torch_npu_stubs()
        self.prefetch = _reload_prefetch_module()

        import torch  # type: ignore[import-not-found]  # noqa: E402

        self.trigger = torch.randn(1)
        self.w1 = torch.randn(4, 4)
        self.w2 = torch.randn(4, 4)

        class _P(self.prefetch.PrefetcherBase):
            pass

        self.obj = _P()

    def test_prefetch_attention_noop_when_map_missing_or_empty(self):
        self.obj.prefetch_tensors_map = None
        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_attention(self.trigger)
            m.assert_not_called()

        self.obj.prefetch_tensors_map = {}
        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_attention(self.trigger)
            m.assert_not_called()

    def test_prefetch_attention_iterates_map(self):
        self.obj.prefetch_tensors_map = {
            "w1": (self.w1, 11),
            "w2": (self.w2, 22),
        }
        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_attention(self.trigger)

            self.assertEqual(m.call_count, 2)
            m.assert_any_call(_TensorIs(self.w1), _TensorIs(self.trigger), 11)
            m.assert_any_call(_TensorIs(self.w2), _TensorIs(self.trigger), 22)

    def test_prefetch_kvcache_uses_first_tensor_and_max_prefetch(self):
        import torch  # type: ignore[import-not-found]  # noqa: E402

        k_cache = torch.randn(2, 2)
        v_cache = torch.randn(2, 2)

        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_kvcache(self.trigger, (k_cache, v_cache))
            m.assert_called_once_with(_TensorIs(k_cache), _TensorIs(self.trigger), self.obj.max_prefetch_size)

        empty_k = torch.empty(0)
        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_kvcache(self.trigger, (empty_k, v_cache))
            m.assert_not_called()

    def test_prefetch_kvcache_noop_on_none_or_non_tuple(self):
        import torch  # type: ignore[import-not-found]  # noqa: E402

        k_cache = torch.randn(2, 2)
        v_cache = torch.randn(2, 2)

        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            self.obj.prefetch_kvcache(self.trigger, None)
            m.assert_not_called()

        with mock.patch.object(self.prefetch.PrefetcherBase, "prefetch_weight") as m:
            # list should not match the `isinstance(..., tuple)` check in implementation
            self.obj.prefetch_kvcache(self.trigger, [k_cache, v_cache])  # type: ignore[arg-type]
            m.assert_not_called()

    def test_prefetch_moe_is_noop_base_impl(self):
        # Base class default is a no-op; it should not raise.
        self.obj.prefetch_moe(self.trigger)


if __name__ == "__main__":
    unittest.main()

