# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import unittest
from unittest.mock import MagicMock
from omni_npu.v1.parsers import PanguReasoningParser
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.protocol import DeltaMessage


class TestPanguReasoningParserExtractReasoning(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {
            "<think>": 100,
            "</think>": 101,
            "[unused16]": 102,
            "[unused17]": 103,
        }
        self.mock_tokenizer.tokenizer = self.mock_tokenizer

    def test_extract_reasoning_with_think_tags(self):
        """case: special token is <think> and </think>"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>正在思考如何写代码...</think>这是你的代码：print('hello')"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "正在思考如何写代码...")
        self.assertEqual(content, "这是你的代码：print('hello')")

    def test_extract_reasoning_with_unused_tags(self):
        """case: special token is [unused16] and [unused17]"""
        self.mock_tokenizer.get_vocab.return_value = {
            "[unused16]": 102,
            "[unused17]": 103
        }

        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "[unused16]分析用户需求中...[unused17]完成提取。"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "分析用户需求中...")
        self.assertEqual(content, "完成提取。")

    def test_extract_reasoning_with_only_reasoning(self):
        """case: only reasoning"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>思考到一半"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "思考到一半")
        self.assertIsNone(content)

    def test_extract_reasoning_with_empty_content(self):
        """case: content is empty"""
        parser = PanguReasoningParser(self.mock_tokenizer)
        request = MagicMock(spec=ChatCompletionRequest)

        model_output = "<think>思考完了</think>"
        reasoning, content = parser.extract_reasoning(model_output, request)

        self.assertEqual(reasoning, "思考完了")
        self.assertIsNone(content)


class TestPanguReasoningParserExtractReasoningStreaming(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.vocab = {
            "<think>": 10,
            "</think>": 11,
            "Hello": 20,
            "World": 21
        }
        self.mock_tokenizer.get_vocab.return_value = self.vocab
        self.mock_tokenizer.tokenizer = self.mock_tokenizer

        self.parser = PanguReasoningParser(self.mock_tokenizer)

    def test_is_reasoning_end(self):
        """测试推理结束标记的计数逻辑"""
        self.parser.delta_token_ids = [11]  # 模拟当前 delta 包含结束符

        # 第一次检测到结束符，应该返回 True
        input_ids = [10, 20, 11]
        self.assertTrue(self.parser.is_reasoning_end(input_ids))
        self.assertEqual(self.parser.is_reasoning_end_count, 1)

        # 第二次调用（模拟重复触发或其他逻辑），计数器变为 2，应返回 False
        self.assertFalse(self.parser.is_reasoning_end(input_ids))
        self.assertEqual(self.parser.is_reasoning_end_count, 2)

    def test_extract_reasoning_streaming_with_multi_token(self):
        """测试起始标签和推理文本在同一个 chunk 中的场景: '<think>Hello'"""
        # 模拟输入参数
        # previous: 空
        # delta: '<think>Hello' (假设对应 token IDs [10, 20])
        previous_text = ""
        current_text = "<think>Hello"
        delta_text = "<think>Hello"
        previous_token_ids = []
        current_token_ids = [10, 20]
        delta_token_ids = [10, 20]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning, "Hello")
        self.assertIsNone(result.content)

    def test_extract_reasoning_streaming_with_normal_reasoning(self):
        """测试正常的推理过程流（标签已在之前出现过）"""
        # 模拟之前已经有了 <think>
        previous_text = "<think>"
        current_text = "<think>Thinking..."
        delta_text = "Thinking..."
        previous_token_ids = [10]
        current_token_ids = [10, 25, 26]  # 假设 25, 26 是 Thinking...
        delta_token_ids = [25, 26]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        self.assertEqual(result.reasoning, "Thinking...")

    def test_extract_reasoning_streaming_whit_end(self):
        """测试推理结束的流场景"""
        previous_text = "<think>Done"
        current_text = "<think>Done</think>Answer"
        delta_text = "</think>Answer"
        previous_token_ids = [10, 30]
        current_token_ids = [10, 30, 11, 40]
        delta_token_ids = [11, 40]

        result = self.parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids
        )

        # 此时应分别提取出最后一段推理和起始正文
        self.assertEqual(result.reasoning, "")  # </think> 前面没有新推理
        self.assertEqual(result.content, "Answer")


if __name__ == '__main__':
    unittest.main()
