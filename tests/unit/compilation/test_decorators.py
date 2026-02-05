from typing import TypeVar
from unittest.mock import patch, MagicMock  

import torch
import torch.nn as nn
import os
import pytest

import vllm.compilation.decorators as _dec_mododule
from vllm.config import VllmConfig, CUDAGraphMode

from omni_npu.compilation.decorators import (
    support_ge_compile,
    _bypass_prefill,
    _wrap_call,
    patch_compile_decorators,
)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Before the test, back up all the global states. 
    After the test, restore them to avoid environmental contamination between test cases.
    """
    original_support_compile = _dec_mododule._support_torch_compile
    original_ge_env = os.environ.get("TORCH_COMPILE_GE", None)
    yield
    _dec_mododule._support_torch_compile = original_support_compile
    if original_ge_env:
        os.environ["TORCH_COMPILE_GE"] = original_ge_env
    else:
        os.environ.pop("TORCH_COMPILE_GE", None)


class TestModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class TestModelWithTuple(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x * 2, )


class TestModelWithList(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x * 2]


def test_support_ge_compile():
    """Test the GE compilation patch: dynamic inheritance + overriding init/call, no duplicate inheritance"""
    dynamic_arg_dims = {"x": 0}
    from omni_npu.compilation.ge_wrapper import TorchNpuCompilerWrapperWithCustomDispatcher

    with patch("vllm.compilation.counter.compilation_counter") as mock_counter:
        # first execution, dynamically applying patches
        patched_cls = support_ge_compile(TestModel, dynamic_arg_dims)
        # check whether to dynamically add the parent class
        assert TorchNpuCompilerWrapperWithCustomDispatcher in patched_cls.__bases__
        # The second execution: Pass in the patched class to avoid duplicate inheritance
        patched_cls_2 = support_ge_compile(patched_cls, dynamic_arg_dims)
        assert patched_cls_2 is patched_cls


@pytest.mark.parametrize(
    "num_prefills, cudagraph_runtime_mode, expected_hit", 
    [
        (1, CUDAGraphMode.FULL, True),
        (0, CUDAGraphMode.FULL, False),
        (1, CUDAGraphMode.FULL, True),
        (0, CUDAGraphMode.NONE, True)
    ]
)
def test_bypass_prefill(num_prefills, cudagraph_runtime_mode, expected_hit):
    """Test whether bypass compilation is required and proceed with the native forward method."""
    test_model = TestModel(vllm_config=VllmConfig(), prefix="test")
    mock_attn_metadata = {"0": MagicMock(num_prefills=num_prefills)}
    test_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with patch("omni_npu.compilation.decorators.get_forward_context") as mock_get_forward_context:
        mock_get_forward_context.return_value.attn_metadata = mock_attn_metadata
        mock_get_forward_context.return_value.cudagraph_runtime_mode = cudagraph_runtime_mode

        hit, retval = _bypass_prefill(test_model, test_tensor)

        assert hit == expected_hit
        if expected_hit:
            assert torch.allclose(retval, test_tensor * 2)
        else:
            assert retval is None


def test_patch_compile_decorators_ge_compile():
    """Test set TORCH_COMPILE_GE to True and replace it with the GE compilation function."""
    os.environ["TORCH_COMPILE_GE"] = "True"
    with patch("omni_npu.compilation.decorators.support_ge_compile") as mock_ge_compile_func:
        patch_compile_decorators()
        assert _dec_mododule._support_torch_compile == mock_ge_compile_func


def test_patch_compile_decorators_no_ge_compile():
    """Test set TORCH_COMPILE_GE to False and replace it with the native wrapper function."""
    os.environ["TORCH_COMPILE_GE"] = "False"
    original_decorator = MagicMock(return_value = TestModel)
    _dec_mododule._support_torch_compile = original_decorator

    with patch("omni_npu.compilation.decorators._wrap_call", MagicMock(side_effect=lambda x: x)) as mock_wrap:
        patch_compile_decorators()

        patched_cls = _dec_mododule._support_torch_compile(TestModel, {})

        original_decorator.assert_called_once()
        mock_wrap.assert_called_once_with(TestModel.__call__)
        assert issubclass(patched_cls, nn.Module)


@pytest.mark.parametrize("env", ["True", "TRUE", "true", "False", "FALSE", "false"])
def test_patch_compile_decorators_ge_env(env):
    """Test the case-insensitivity of the TORCH_COMPILE_GE size."""
    os.environ["TORCH_COMPILE_GE"] = env
    with patch("omni_npu.compilation.decorators.support_ge_compile") as mock_ge_compile_func:
        patch_compile_decorators()
        if env.lower() == "true":
            assert _dec_mododule._support_torch_compile == mock_ge_compile_func
        else:
            assert _dec_mododule._support_torch_compile.__name__ == "_patched_support_torch_compile"


def test_patch_compile_decorators_no_env():
    """Test not set the TORCH_COMPILE_GE, default uses the native packaging branch."""
    os.environ.pop("TORCH_COMPILE_GE", None)
    patch_compile_decorators()
    assert _dec_mododule._support_torch_compile.__name__ == "_patched_support_torch_compile"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])