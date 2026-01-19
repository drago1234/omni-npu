# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


__all__ = ["register_lazy_parsers"]

from omni_npu.v1.parsers.pangu_reasoning_parser import PanguReasoningParser
from omni_npu.v1.parsers.pangu_tool_parser import PanguToolParser
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

"""
tool and reasoning parser
"""

parser_name = "pangu"

_TOOL_PARSERS_TO_REGISTER = {
    parser_name: PanguToolParser
}

_REASONING_PARSERS_TO_REGISTER = {
    parser_name: PanguReasoningParser
}


def register_lazy_parsers():
    for name, parser_cls in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = parser_cls.__module__
        class_name = parser_cls.__name__
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)
    for name, parser_cls in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = parser_cls.__module__
        class_name = parser_cls.__name__
        ToolParserManager.register_lazy_module(name, module_path, class_name)
