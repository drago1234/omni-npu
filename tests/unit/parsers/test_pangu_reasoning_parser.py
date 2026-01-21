# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
from importlib import import_module
from unittest.mock import MagicMock

import pytest
from omni_npu.v1.parsers import register_lazy_parsers
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from .utils import run_reasoning_extraction

register_lazy_parsers()

parser_name = "pangu"
start_token = "<think>"
end_token = "</think>"


@pytest.fixture(scope="module")
def pangu_tokenizer():
    tokenizer = MagicMock()
    mock_vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "hello": 102,
        "world": 103,
        "<think>": 45981,
        "This": 39334,
        "is": 41824,
        "a": 45509,
        "reasoning": 5755,
        "section": 12206,
        "</think>": 45982,
        "the": 35740,
        "rest": 25719,
    }

    tokenizer.get_vocab.return_value = mock_vocab
    tokenizer.vocab = mock_vocab

    return tokenizer


EMPTY = {
    "output": "",
    "reasoning": "",
    "content": None,
    "is_reasoning_end": False,
}
EMPTY_STREAMING = {
    "output": "",
    "reasoning": None,
    "content": None,
    "is_reasoning_end": False,
}

TEST_CASES = [
    pytest.param(
        False,
        EMPTY,
        id="empty",
    ),
    pytest.param(
        True,
        EMPTY_STREAMING,
        id="empty_streaming",
    ),
]


@pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
def test_reasoning(
        streaming: bool,
        param_dict: dict,
        pangu_tokenizer,
):
    output = pangu_tokenizer.tokenize(param_dict["output"])
    # decode everything to tokens
    output_tokens: list[str] = [
        pangu_tokenizer.convert_tokens_to_string([token]) for token in output
    ]
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(
        pangu_tokenizer
    )

    reasoning, content = run_reasoning_extraction(
        parser, output_tokens, streaming=streaming
    )

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]

    # Test is_reasoning_end
    output_ids = pangu_tokenizer.convert_tokens_to_ids(output)
    parser.delta_token_ids = output_ids
    is_reasoning_end = parser.is_reasoning_end(output_ids)
    assert is_reasoning_end == param_dict["is_reasoning_end"]

    # Test extract_content
    if param_dict["content"] is not None:
        content = parser.extract_content_ids(output_ids)
        assert content == pangu_tokenizer.convert_tokens_to_ids(
            pangu_tokenizer.tokenize(param_dict["content"])
        )
    else:
        content = parser.extract_content_ids(output)
        assert content == []


single_token_output = [
    "<think>",
    "Some ",
    "reasoning ",
    "content",
    "</think>",
    "Final ",
    "answer",
]
mutil_tokens_output = [
    "<think>This ",
    "is a ",
    "reasoning process ",
    "section</think>",
    "This is ",
    "the rest",
]

SIMPLE_REASONING = {
    "output": single_token_output,
    "reasoning": "Some reasoning content",
    "content": "Final answer",
    "is_reasoning_end": True,
}

MUTIL_TOKENS_REASONING = {
    "output": mutil_tokens_output,
    "reasoning": "This is a reasoning process section",
    "content": "This is the rest",
    "is_reasoning_end": True,
}

TEST_CASES = [
    pytest.param(
        False,
        SIMPLE_REASONING,
        id="simple_reasoning",
    ),
    pytest.param(
        False,
        MUTIL_TOKENS_REASONING,
        id="mutil_tokens_reasoning",
    ),
]


class TestPanguReasoningParserStreaming:
    """Test streaming functionality of PanguReasoningParser."""

    @pytest.mark.parametrize("streaming, param_dict", TEST_CASES)
    def test_pangu_reasoning_extraction(self, pangu_tokenizer, streaming, param_dict):
        """
        Test basic reasoning extraction in streaming modes.
        """
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(parser_name)(pangu_tokenizer)

        reasoning, content = run_reasoning_extraction(
            parser, param_dict["output"], streaming=streaming
        )
        assert reasoning == param_dict["reasoning"]
        assert content == param_dict["content"]
