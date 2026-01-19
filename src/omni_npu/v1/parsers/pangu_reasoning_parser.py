# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.tokenizers import TokenizerLike


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
        self.delta_token_ids = None
        self.is_reasoning_end_count = 0

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
        if self.is_reasoning_end_count == 1:
            return True
        else:
            return False

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
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
