# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field, dataclass

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.tokenizers import cached_tokenizer_from_config


@config
@dataclass
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings that
    delimit the reasoning block. The token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String that indicates the end of reasoning content."""
    think_start_str: str = ""
    """Deprecated alias for `reasoning_start_str`."""
    think_end_str: str = ""
    """Deprecated alias for `reasoning_end_str`."""
    tool_call_start_token: str = ""
    """String that indicates the start of tool call."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=True
    )
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=True
    )
    _tool_call_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=True
    )
    _enabled: bool = field(default=False, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        return self._reasoning_end_token_ids
    
    @property
    def tool_call_start_token_ids(self) -> list[int] | None:
        return self._tool_call_start_token_ids

    @classmethod
    def as_argparse_dict(cls):
        """
        Convert the documentation and structure of ReasoningConfig into the add_argument parameters of argparse.
        """
        doc = cls.__doc__.strip() if cls.__doc__ else "Reasoning configuration."
        return {
            "type": str,
            "default": None,
            "help": f"{doc} (Provide as a JSON string or path to a JSON file)."
        }

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return

        reasoning_start_str = self.reasoning_start_str or self.think_start_str
        reasoning_end_str = self.reasoning_end_str or self.think_end_str
        tool_call_start_token = self.tool_call_start_token
        if not reasoning_start_str or not reasoning_end_str:
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )
        
        self._tool_call_start_token_ids = tokenizer.encode(
            tool_call_start_token, add_special_tokens=False
        )

        if not self._reasoning_start_token_ids or not self._reasoning_end_token_ids:
            raise ValueError(
                "ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{reasoning_start_str}', "
                f"reasoning_end_str='{reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
