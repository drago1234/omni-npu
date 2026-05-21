# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.tokenizers import TokenizerLike
from vllm.entrypoints.openai.protocol import ChatCompletionRequest

class PanguReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for the Pangu model.

    The Pangu model uses either [unused16]...[unused17] or <think>...</think>
    tokens to enclose reasoning text within its output. This parser dynamically
    identifies the available tokens and extracts the reasoning content.
    It also handles edge cases in streaming where the start token and initial
    reasoning text are combined in a single chunk.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.delta_token_ids = []
        self.is_reasoning_end_count = 0
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Pangu defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        think = chat_kwargs.get("think", True)
        thinking = chat_kwargs.get("thinking", True)
        
        self.thinking_enabled = True
        if not think or not thinking:
            self.thinking_enabled = False
    
    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>" if self.vocab.get("<think>") else "[unused16]"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>" if self.vocab.get("</think>") else "[unused17]"

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self.end_token_id in input_ids and self.end_token_id in self.delta_token_ids:
            self.is_reasoning_end_count += 1
            return self.is_reasoning_end_count == 1
    
        if input_ids[-1] == self.end_token_id:
            return True

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not self.thinking_enabled:
            return  DeltaMessage(content=delta_text)
            
        self.delta_token_ids = delta_token_ids

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if (
                ret is not None
                and self.start_token_id in delta_token_ids
                and self.end_token_id not in delta_token_ids
        ):
            # multi token case: start token in delta such as '<think>Hello' extract 'Hello' to reasoning
            start_index = delta_text.find(self.start_token)
            delta_text = delta_text[start_index + len(self.start_token):]
            ret = DeltaMessage(reasoning=delta_text)
        return ret

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output
        
        if self.end_token not in model_output:
            return model_output, None
        else:
            # Extract reasoning content from the model output.
            reasoning, _, content = model_output.partition(self.end_token)

            final_content = content or None
            return reasoning, final_content