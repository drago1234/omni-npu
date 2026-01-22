# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import unittest
from unittest.mock import MagicMock, patch

from omni_npu.v1.config import ReasoningConfig
from vllm.config.model import ModelConfig


class TestReasoningConfig(unittest.TestCase):

    def setUp(self):
        self.model_config = MagicMock(spec=ModelConfig)

        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.tokenize.side_effect = lambda x: [x] if x else []
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: [1001] if x == ["<think>"] else (
            [1002] if x == ["</think>"] else [])

        def mock_convert(tokens):
            mapping = {"token_<think>": [1001], "token_</think>": [1002]}
            return mapping.get(tokens[0], [999])

        self.mock_tokenizer.convert_tokens_to_ids.side_effect = mock_convert

    def test_thinking_enabled(self):
        """测试 is_thinking_enabled 的各种组合情况"""
        # case1: 全为空
        cfg = ReasoningConfig()
        self.assertFalse(cfg.is_thinking_enabled)

        # case2: 只有开始 Token
        cfg.think_start_token_ids = [1001]
        self.assertFalse(cfg.is_thinking_enabled)

        # case3: 开始和结束都有 (启用)
        cfg.think_end_token_ids = [1002]
        self.assertTrue(cfg.is_thinking_enabled)

        # case4: 列表为空
        cfg.think_start_token_ids = []
        self.assertFalse(cfg.is_thinking_enabled)

    def test_as_argparse_dict(self):
        """测试 argparse 参数转换逻辑"""
        arg_dict = ReasoningConfig.as_argparse_dict()

        self.assertIn("type", arg_dict)
        self.assertEqual(arg_dict["type"], str)
        self.assertIn("help", arg_dict)
        self.assertTrue("JSON string" in arg_dict["help"])

    @patch('omni_npu.v1.config.reasoning_config.cached_tokenizer_from_config')
    def test_initialize_token_ids_fail(self, mock_get_tokenizer):
        """
        测试短路逻辑：如果 String 没配对，就不应该去拿 Tokenizer
        """
        cfg = ReasoningConfig(think_start_str="<think>", think_end_str=None)

        cfg.initialize_token_ids(self.model_config)

        mock_get_tokenizer.assert_not_called()

    @patch('omni_npu.v1.config.reasoning_config.cached_tokenizer_from_config')
    def test_initialize_token_ids_success(self, mock_get_tokenizer):
        """initialize_token_ids success case"""
        mock_get_tokenizer.return_value = self.mock_tokenizer

        cfg = ReasoningConfig(
            think_start_str="<think>",
            think_end_str="</think>"
        )

        self.mock_tokenizer.tokenize.side_effect = lambda x: [x]
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = lambda x: [1001] if x == ["<think>"] else [1002]

        cfg.initialize_token_ids(self.model_config)

        mock_get_tokenizer.assert_called_once_with(model_config=self.model_config)

        self.assertEqual(cfg.think_start_token_ids, [1001])
        self.assertEqual(cfg.think_end_token_ids, [1002])
        self.assertTrue(cfg.is_thinking_enabled)


if __name__ == '__main__':
    unittest.main()
