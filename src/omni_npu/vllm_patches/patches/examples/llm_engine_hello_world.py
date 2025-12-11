# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


from vllm.tasks import SupportedTask
from vllm.v1.engine.llm_engine import LLMEngine
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

@register_patch("LLMEngineHelloWorld", LLMEngine)
class LLMEngineHelloWorldPatch(VLLMPatch):
    """
    Makes LLMEngines print 'Hello World' when get supported tasks.
    """

    _attr_names_to_apply = ['print_hello_world', 'get_supported_tasks']

    @staticmethod
    def print_hello_world():
        print("Hello World")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        self.print_hello_world()
        return self.engine_core.get_supported_tasks()
