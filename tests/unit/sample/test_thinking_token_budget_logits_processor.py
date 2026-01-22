# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import unittest
from unittest.mock import MagicMock
import torch

from omni_npu.v1.config import ReasoningConfig
from omni_npu.v1.sample.logits_processor import ThinkingTokenBudgetLogitsProcessor
from vllm.config import VllmConfig, SchedulerConfig
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality


class TestThinkingTokenBudgetLogitsProcessor(unittest.TestCase):

    def setUp(self):
        self.scheduler_config = MagicMock(spec=SchedulerConfig)
        self.scheduler_config.max_num_seqs = 4

        self.reasoning_config = MagicMock(spec=ReasoningConfig)
        self.reasoning_config.is_thinking_enabled = True
        self.reasoning_config.think_start_token_ids = [100]  # <think>
        self.reasoning_config.think_end_token_ids = [200]  # </think>
        self.reasoning_config.thinking_token_budget = 5

        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.scheduler_config = self.scheduler_config
        self.vllm_config.reasoning_config = self.reasoning_config

        self.device = torch.device("cpu")
        self.processor = ThinkingTokenBudgetLogitsProcessor(
            self.vllm_config, self.device, is_pin_memory=False
        )

    def test_init_state_entry(self):
        """测试从 Prompt 恢复思维状态的逻辑"""
        # case1: Prompt 中包含未闭合的 <think>
        prompt_ids = [1, 2, 100, 3, 4]  # think_start 是 100
        state = self.processor._init_state_entry(prompt_ids, thinking_token_budget=10)

        self.assertTrue(state["in_think"])
        self.assertEqual(state["think_count"], 2)  # [3, 4] 两个词
        self.assertFalse(state["in_end"])

        # case2: Prompt 中思维已结束
        prompt_ids = [1, 100, 3, 200, 5]  # think_end 是 200
        state = self.processor._init_state_entry(prompt_ids, thinking_token_budget=10)
        self.assertFalse(state["in_think"])
        self.assertEqual(state["think_count"], 0)

    def test_update_think_state_transition(self):
        """测试生成过程中从 in_think 转换到 in_end 的触发"""
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 4,
            "thinking_token_budget": 5,
            "output_tok_ids": [10, 11, 12, 13, 14],  # 刚生成了第 5 个词
            "prev_output_length": 4,
            "check_count_down": 1
        }

        # 执行更新
        self.processor._update_think_state(state)

        # 预算是 5，已生成 5 个，应该进入强制结束模式
        self.assertTrue(state["in_end"])
        self.assertFalse(state["in_think"])
        self.assertEqual(state["end_count"], 0)

    def test_apply(self):
        """测试在 in_end 模式下是否正确修改了 Logits"""
        # 设置一个处于强行结束状态的 request 0
        self.processor._state[0] = {
            "in_end": True,
            "end_count": 0,
            "output_tok_ids": [100, 1, 2, 3, 4, 5]
        }

        # 模拟 2 个 batch 的 logits (Vocab size = 1000)
        logits = torch.zeros((2, 1000), dtype=torch.float32)

        updated_logits = self.processor.apply(logits)

        # 验证 request 0 的 think_end_token_ids [200] 被设为极高值
        self.assertEqual(updated_logits[0, 200], 1e9)
        # 验证没有状态的 request 1 未受影响
        self.assertEqual(updated_logits[1, 200], 0)

    def test_update_state_with_batch_management_swap(self):
        """测试 BatchUpdate 的 SWAP 移动逻辑"""
        # 1. 初始状态：index 0 有数据
        mock_params = MagicMock()
        mock_params.extra_args = {"thinking_token_budget": 10}

        batch_update = MagicMock(spec=BatchUpdate)
        batch_update.added = [(0, mock_params, [1, 2, 100], [])]
        batch_update.removed = []
        batch_update.moved = []

        self.processor.update_state(batch_update)
        self.assertIn(0, self.processor._state)
        self.assertTrue(self.processor._state[0]["in_think"])

        # 2. 执行 SWAP (0 -> 1)
        # 模拟 direction == MoveDirectionality.SWAP
        batch_update.added = []
        batch_update.moved = [(0, 1, MoveDirectionality.SWAP)]
        self.processor.update_state(batch_update)

        self.assertEqual(self.processor._state[1]["prompt_tok_ids"], [1, 2, 100])
        self.assertEqual(self.processor._state[0], {})

    def test_update_state_with_batch_management_move(self):
        """测试非 SWAP 类型的移动逻辑 (例如迁移)"""
        # 初始：index 5 有数据
        self.processor._state[5] = {"data": "test"}

        # 模拟移动 5 -> 10 (非 SWAP)
        batch_update = MagicMock(spec=BatchUpdate)
        batch_update.added = []
        batch_update.removed = []
        # 此时会执行：self._state[i2] = self._state.pop(i1, {})
        batch_update.moved = [(5, 10, "OTHER_DIRECTION")]

        self.processor.update_state(batch_update)

        # 验证 5 被 pop 掉了，10 拿到了数据
        self.assertNotIn(5, self.processor._state)
        self.assertEqual(self.processor._state[10]["data"], "test")

    def test_find_last_sequence_index(self):
        """测试工具函数：查找最后一次出现的序列索引"""
        target = [1, 2, 3, 1, 2, 3, 4]
        sub = [1, 2]
        idx = ThinkingTokenBudgetLogitsProcessor._find_last_sequence_index(target, sub)
        self.assertEqual(idx, 3)  # 第二次出现的 [1, 2] 起始于下标 3

    def test_update_think_state_with_recent_start_pos_GEQ_0(self):
        """recent_start_pos >= 0"""
        state = {
            "in_think": False,
            "in_end": False,
            "think_count": 0,
            "prev_output_length": 0,
            "output_tok_ids": [102, 103, 100],  # [hello, world, <think>]
            "thinking_token_budget": 10,
            "check_count_down": 0
        }

        self.processor._update_think_state(state)

        assert state["in_think"] is True
        assert state["think_count"] == 0

    def test_update_think_state_with_recent_start_and_end_pos_GEQ_0(self):
        """recent_start_pos >= 0 and recent_end_pos >= 0"""
        self.processor.think_start_token_ids = [100]
        self.processor.think_end_token_ids = [101]

        state = {
            "in_think": False,
            "in_end": False,
            "think_count": 0,
            "prev_output_length": 0,
            "output_tok_ids": [101, 104, 100],
            "thinking_token_budget": 10,
            "check_count_down": 0
        }

        self.processor._update_think_state(state)
        assert state["in_think"] is True

    def test_update_think_state_with_in_end_mode(self):
        """case: in end mode"""
        state = {
            "in_think": False,
            "in_end": True,  # True 模拟预算耗尽后的状态
            "end_count": 0,
            "prev_output_length": 5,
            "output_tok_ids": [1, 2, 3, 4, 5, 6],  # 新增一个 token
            "thinking_token_budget": 10,
            "check_count_down": 0
        }

        # 第一次更新，end_count 变成 1
        self.processor._update_think_state(state)

        # 此时如果 self.think_end_token_ids 长度为 1
        assert state["in_end"] is False
        assert state["end_count"] == 0
        assert state["check_count_down"] == 10


if __name__ == '__main__':
    unittest.main()
