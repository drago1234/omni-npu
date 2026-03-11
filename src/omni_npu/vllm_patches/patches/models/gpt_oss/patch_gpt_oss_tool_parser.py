import json
from collections.abc import Sequence

from vllm.entrypoints.openai.tool_parsers.openai_tool_parser import OpenAIToolParser
from vllm.entrypoints.harmony_utils import parse_output_into_messages
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


@register_patch("GptOssToolParserTruncatedContentPatch", OpenAIToolParser)
class GptOssToolParserTruncatedContentPatch(VLLMPatch):
    _attr_names_to_apply = ["extract_tool_calls"]

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."
            )

        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        final_content = None
        commentary_content = None

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if len(msg.content) < 1:
                    continue
                msg_text = msg.content[0].text
                if msg.recipient and msg.recipient.startswith("functions."):
                    if not msg.content_type or "json" in msg.content_type:
                        try:
                            tool_args = json.dumps(json.loads(msg_text))
                        except json.JSONDecodeError:
                            logger.exception(
                                "Error decoding JSON tool call from response."
                            )
                            tool_args = msg_text
                    else:
                        tool_args = msg_text
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=tool_args,
                            ),
                        )
                    )
                elif msg.channel == "final":
                    final_content = msg_text
                elif msg.channel == "commentary" and not msg.recipient:
                    commentary_content = msg_text

        # Extract partial content from the parser state if the generation was truncated
        if parser.current_content:
            if parser.current_channel == "final":
                final_content = parser.current_content
            elif parser.current_channel == "commentary" and not parser.current_recipient:
                commentary_content = parser.current_content

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=final_content or commentary_content,
        )
