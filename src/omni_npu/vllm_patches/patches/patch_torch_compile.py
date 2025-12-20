# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.logger import logger
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
import torch

origin_TorchCompileWithNoGuardsWrapper_init = TorchCompileWithNoGuardsWrapper.__init__

@register_patch("TorchCompileWithNoGuardsWrapper", TorchCompileWithNoGuardsWrapper)
class TorchCompileWithNoGuardsWrapperPatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']
    def __init__(self):
        def patch_torch_compile(fun):
            def new_fun(*args, **kwargs):
                key = "options"
                assert key in kwargs
                kwargs[key] = None
                logger.warning("skip torch compile error, torch compile option set None")
                return fun(*args, **kwargs)
            return new_fun

        old = torch.compile
        torch.compile = patch_torch_compile(torch.compile)
        origin_TorchCompileWithNoGuardsWrapper_init(self)
        torch.compile = old

