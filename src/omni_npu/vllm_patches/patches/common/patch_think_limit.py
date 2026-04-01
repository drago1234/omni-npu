# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm import SamplingParams


"""
Passing args thinking_token_budget  
"""

_to_sampling_params = ChatCompletionRequest.to_sampling_params


@register_patch("ChatCompletionRequestPatch", ChatCompletionRequest)
class ChatCompletionRequestPatch(VLLMPatch):
    """
    Patch to ChatCompletionRequest to support the 'thinking_token_budget'
    parameter in API requests, allowing per-request control over reasoning length.
    """

    _attr_names_to_apply = ['to_sampling_params']

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict,
    ) -> SamplingParams:
        # adapter reasoning_config and thinking_token_budget
        thinking_token_budget = None
        if hasattr(self, "model_extra") and self.model_extra:
            thinking_token_budget = self.model_extra.get("thinking_token_budget")

        sampling_params: SamplingParams = _to_sampling_params(
            self, max_tokens, logits_processor_pattern, default_sampling_params
        )

        # Serialized and passed to the worker process
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args["thinking_token_budget"] = thinking_token_budget
        setattr(sampling_params, "thinking_token_budget", thinking_token_budget)

        return sampling_params


@register_patch("SamplingParamsPatch", SamplingParams)
class SamplingParamsPatch(VLLMPatch):
    _attr_names_to_apply = ['from_optional']

    @staticmethod
    def from_optional(thinking_token_budget: int | None = None, **kwargs) -> "SamplingParams":
        budget = kwargs.pop('thinking_token_budget', thinking_token_budget)

        if kwargs.get('logit_bias'):
            lb = kwargs.get("logit_bias")
            kwargs["logit_bias"] = {int(k): min(100.0, max(-100.0, v)) for k, v in lb.items()}

        instance = SamplingParams(**kwargs)
        # Serialized and passed to the worker process
        if instance.extra_args is None:
            instance.extra_args = {}
        instance.extra_args["thinking_token_budget"] = budget

        return instance


"""
Adding LogitsProcessors to BUILTIN_LOGITS_PROCESSORS
"""
import vllm.v1.sample.logits_processor as lp_module
from omni_npu.v1.sample.logits_processor import ThinkingTokenBudgetLogitsProcessor

if ThinkingTokenBudgetLogitsProcessor not in lp_module.BUILTIN_LOGITS_PROCESSORS:
    lp_module.BUILTIN_LOGITS_PROCESSORS.append(ThinkingTokenBudgetLogitsProcessor)

processor_name = ThinkingTokenBudgetLogitsProcessor.__name__

if processor_name not in lp_module.__all__:
    lp_module.__all__.append(processor_name)
