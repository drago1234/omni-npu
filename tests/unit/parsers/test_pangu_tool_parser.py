# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json

import pytest

from omni_npu.v1.parsers.pangu_tool_parser import PanguToolParser
from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall

from unittest.mock import MagicMock


@pytest.fixture(scope="module")
def pangu_tokenizer():
    tokenizer = MagicMock()
    mock_vocab = {
        "[unused11]": 101,
        "[unused12]": 102
    }

    tokenizer.get_vocab.return_value = mock_vocab
    tokenizer.vocab = mock_vocab

    return tokenizer


@pytest.fixture
def pangu_tool_parser(pangu_tokenizer):
    return PanguToolParser(pangu_tokenizer)


def assert_tool_calls(
        actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
            actual_tool_calls, expected_tool_calls
    ):
        assert isinstance(actual_tool_call.id, str)
        assert len(actual_tool_call.id) > 0

        assert actual_tool_call.type == "function"
        assert actual_tool_call.function.name == expected_tool_call.function.name
        # Compare arguments as JSON objects to handle formatting differences
        actual_args = json.loads(actual_tool_call.function.arguments)
        expected_args = json.loads(expected_tool_call.function.arguments)
        assert actual_args == expected_args


def test_extract_tool_calls_no_tools(pangu_tool_parser):
    model_output = "This is a test"
    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    assert not extracted_tool_calls.tools_called
    assert extracted_tool_calls.tool_calls == []
    assert extracted_tool_calls.content == model_output


@pytest.mark.parametrize(
    ids=[
        "single_tool_call",
        "multiple_tool_calls",
        "tool_call_with_content_before",
        "tool_call_with_mixed_args",
        "tool_call_with_chinese_content",
    ],
    argnames=["model_output", "expected_tool_calls", "expected_content"],
    argvalues=[
        (
                """[unused11]
        [{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]
        [unused12]""",
                [
                    ToolCall(
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "Dallas",
                                    "state": "TX",
                                    "unit": "fahrenheit",
                                }
                            ),
                        )
                    )
                ],
                None,
        ),
        (
                """[unused11]
        [{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]
        [unused12]
        [unused11]
        [{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}]
        [unused12]""",
                [
                    ToolCall(
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "Dallas",
                                    "state": "TX",
                                    "unit": "fahrenheit",
                                }
                            ),
                        )
                    ),
                    ToolCall(
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "Orlando",
                                    "state": "FL",
                                    "unit": "fahrenheit",
                                }
                            ),
                        )
                    ),
                ],
                None,
        ),
        (
                """I'll help you check the weather.[unused11]
        [{"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "celsius"}}]
        [unused12]""",
                [
                    ToolCall(
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "Seattle",
                                    "state": "WA",
                                    "unit": "celsius",
                                }
                            ),
                        )
                    )
                ],
                "I'll help you check the weather.",
        ),
        (
                """[unused11]
        [{"name": "get_current_weather", "arguments": {"city": "New York", "state": "NY", "unit": "celsius"}}]
        [unused12]""",
                [
                    ToolCall(
                        function=FunctionCall(
                            name="get_current_weather",
                            arguments=json.dumps(
                                {
                                    "city": "New York",
                                    "state": "NY",
                                    "unit": "celsius",
                                }
                            ),
                        )
                    )
                ],
                None,
        ),
        (
                """I will help you get the weather.[unused11]
        [{"name": "get_weather", "arguments": {"city": "Beijing", "date": "2025-08-01"}}]
        [unused12]""",
                [
                    ToolCall(
                        function=FunctionCall(
                            name="get_weather",
                            arguments=json.dumps(
                                {
                                    "city": "Beijing",
                                    "date": "2025-08-01",
                                }
                            ),
                        )
                    )
                ],
                "I will help you get the weather.",
        ),
    ],
)
def test_extract_tool_calls(
        pangu_tool_parser, model_output, expected_tool_calls, expected_content
):
    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    print(f"extracted_tool_calls: {extracted_tool_calls}")
    assert extracted_tool_calls.tools_called
    assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)

    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_with_thinking_tags(pangu_tool_parser):
    """Test tool extraction when thinking tags are present."""
    model_output = """<think>I want to get the weather.</think>

I will help you get the weather.[unused11]
[{"name": "check_sports_event", "arguments": {"matchCategory": "足球", "leagueName": "英超", "round": "第二轮"}}]
[unused12]"""

    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    print(f"extracted_tool_calls:{extracted_tool_calls}")
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 1
    assert extracted_tool_calls.tool_calls[0].function.name == "check_sports_event"

    expected_content = """<think>I want to get the weather.</think>

I will help you get the weather."""
    assert extracted_tool_calls.content == expected_content


def test_extract_tool_calls_malformed_xml(pangu_tool_parser):
    """Test that malformed XML is handled gracefully."""
    model_output = '[unused11][{"name": "get_weather", "arguments": {"city": "Seattle", "incomplete_arg": "value"}}][unused12]'

    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]

    # Should handle malformed XML gracefully
    # The parser should either extract what it can or return no tool calls
    # depending on how robust we want the parsing to be
    assert isinstance(extracted_tool_calls.tools_called, bool)
    assert isinstance(extracted_tool_calls.tool_calls, list)


def test_extract_tool_calls_empty_arguments(pangu_tool_parser):
    """Test tool calls with no arguments."""
    model_output = """[unused11]get_current_time[unused12]"""

    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    print(f"extracted_tool_calls: {extracted_tool_calls}")
    assert not extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 0
    assert extracted_tool_calls.content == "[unused11]get_current_time[unused12]"


def test_extract_tool_calls_mixed_content(pangu_tool_parser):
    """Test extraction with mixed content and multiple tool calls."""
    model_output = """I plan to go to the UK to watch the Premier League.[unused11]
[{"name": "check_sports_event", "arguments": {"matchCategory": "足球", "leagueName": "英超", "round": "第二轮"}}]
[unused12]

meaningwhile, I will also check the weather in Shanghai.

[unused11]
[{"name": "search_restaurants", "arguments": {"city": "上海", "cuisine": "本帮菜", "price_range": "300-500", "reservation": true}}]
[unused12]"""

    extracted_tool_calls = pangu_tool_parser.extract_tool_calls(
        model_output, request=None
    )  # type: ignore[arg-type]
    print(f"extracted_tool_calls: {extracted_tool_calls}")
    assert extracted_tool_calls.tools_called
    assert len(extracted_tool_calls.tool_calls) == 2

    # Check first tool call
    assert extracted_tool_calls.tool_calls[0].function.name == "check_sports_event"
    args1 = json.loads(extracted_tool_calls.tool_calls[0].function.arguments)
    assert args1["matchCategory"] == "足球"
    assert args1["leagueName"] == "英超"
    assert args1["round"] == "第二轮"

    # Check second tool call
    assert extracted_tool_calls.tool_calls[1].function.name == "search_restaurants"
    args2 = json.loads(extracted_tool_calls.tool_calls[1].function.arguments)
    assert args2["city"] == "上海"
    assert args2["cuisine"] == "本帮菜"
    assert args2["price_range"] == "300-500"
    assert args2["reservation"]

    # Content should be everything before the first tool call
    assert extracted_tool_calls.content == "I plan to go to the UK to watch the Premier League."


def test_streaming_basic_functionality(pangu_tool_parser):
    """Test basic streaming functionality."""
    # Reset streaming state
    pangu_tool_parser.current_tool_name_sent = False
    pangu_tool_parser.prev_tool_call_arr = []
    pangu_tool_parser.current_tool_id = -1
    pangu_tool_parser.streamed_args_for_tool = []

    # Test with a simple tool call
    current_text = """[unused11][{"name": "get_weather", "arguments": {"city": "Beijing"}}][unused12]"""

    # Mock token IDs for testing
    tool_call_start_id = pangu_tool_parser.tool_call_start_token_id or 12345
    tool_call_end_id = pangu_tool_parser.tool_call_end_token_id or 12346

    result = pangu_tool_parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=current_text,
        delta_text="[unused12]",
        previous_token_ids=[],
        current_token_ids=[tool_call_start_id, tool_call_end_id],
        delta_token_ids=[tool_call_end_id],
        request=None,
    )

    # The result behavior depends on the streaming state
    # This test mainly ensures no exceptions are thrown
    assert result is None or hasattr(result, "tool_calls") or hasattr(result, "content")


def test_streaming_no_tool_calls(pangu_tool_parser):
    """Test streaming when there are no tool calls."""
    current_text = "This is just regular text without any tool calls."

    result = pangu_tool_parser.extract_tool_calls_streaming(
        previous_text="This is just regular text",
        current_text=current_text,
        delta_text=" without any tool calls.",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )

    # Should return the delta text as content
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == " without any tool calls."


def test_streaming_with_content_before_tool_calls(pangu_tool_parser):
    """Test streaming when there's content before tool calls."""
    # Reset streaming state
    pangu_tool_parser.current_tool_name_sent = False
    pangu_tool_parser.prev_tool_call_arr = []
    pangu_tool_parser.current_tool_id = -1
    pangu_tool_parser.streamed_args_for_tool = []

    current_text = "I will help you get the weather[unused11]"

    result = pangu_tool_parser.extract_tool_calls_streaming(
        previous_text="I will help you",
        current_text=current_text,
        delta_text="get the weather.[unused11]",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=None,
    )
    # Should return content when no tool call tokens are detected
    assert result is not None
    assert hasattr(result, "content")
    assert result.content == "get the weather."
