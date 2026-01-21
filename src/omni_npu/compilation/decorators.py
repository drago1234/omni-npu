# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import functools
from typing import TypeVar, Union

import torch
import torch.nn as nn

import vllm.compilation.decorators as _dec_mododule
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger


logger = init_logger(__name__)


_T = TypeVar('_T', bound=type[nn.Module])

def support_ge_compile(
        cls: _T,
        dynamic_arg_dims: dict[str, Union[int, list[int]]],
        *args, **kwargs
) -> _T:
    from omni_npu.compilation.ge_wrapper import TorchNpuCompilerWrapperWithCustomDispatcher
    from vllm.compilation.counter import compilation_counter
    from vllm.config import VllmConfig

    if TorchNpuCompilerWrapperWithCustomDispatcher in cls.__bases__:
        return cls

    cls.__bases__ = cls.__bases__ + (TorchNpuCompilerWrapperWithCustomDispatcher,)

    old_init = cls.__init__

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.vllm_config = vllm_config
        compilation_counter.num_models_seen += 1
        TorchNpuCompilerWrapperWithCustomDispatcher.__init__(
            self, vllm_config, dynamic_arg_dims)

    cls.__init__ = __init__
    cls.__call__ = TorchNpuCompilerWrapperWithCustomDispatcher.__call__

    return cls

def _bypass_prefill(self, *args, **kwargs):
    """
    patch vllm's _support_torch_compile's __call__
    If any prefill request exists, torch.all_to_all_single will be used
    in MoE layers, which involves CPU operations and cannot be compiled.
    We use the non-compiled forward for this case.
    """
    attn_metadata = get_forward_context().attn_metadata
    batch_descriptor = get_forward_context().batch_descriptor
    uniform = batch_descriptor.uniform if batch_descriptor is not None else False
    has_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0
    if has_prefill or not uniform:
        logger.debug(f"<<< use original forward")
        return True, self.forward(*args, **kwargs)
    return False, None

def _wrap_call(original_call):
    @functools.wraps(original_call)
    def _new_call(self, *args, **kwargs):
        hit, retval = _bypass_prefill(self, *args, **kwargs)
        logger.debug(f"<<< {hit=}, {retval=}")
        if hit:
            return retval
        logger.debug(f"<<< {hit=}, {retval=}, use original_call")
        model_output = original_call(self, *args, **kwargs)
        if isinstance(model_output, (tuple, list)) and len(model_output) == 1:
            hidden_states = model_output[0]
            if isinstance(hidden_states, list) and \
                    len(hidden_states) == 1 and \
                    isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
            return hidden_states
        else:
            return model_output
    return _new_call

def patch_compile_decorators():
    import os
    use_gegraph = os.getenv("TORCH_COMPILE_GE", "False").lower() == "true"
    if use_gegraph:
        logger.debug("<<< patch_compile_decorators:use ge graph!")
        _dec_mododule._support_torch_compile = support_ge_compile
    else:
        _original_decorator = _dec_mododule._support_torch_compile
        def _patched_support_torch_compile(cls, *args, **kwargs):
            cls = _original_decorator(cls, *args, **kwargs)

            cls.__call__ = _wrap_call(cls.__call__)
            logger.debug("<<< cls.__call__ wrapped!")
            return cls

        _dec_mododule._support_torch_compile = _patched_support_torch_compile
        logger.debug("<<< _patched_support_torch_compile applied!")

