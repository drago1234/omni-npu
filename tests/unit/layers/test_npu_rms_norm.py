# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
from unittest.mock import MagicMock, patch
import torch
from src.omni_npu.layers.npu_rms_norm import NPURMSNorm


class TestNPURMSNorm(unittest.TestCase):
    def setUp(self):
        """initialize the test environment"""
        self.hidden_size = 1024
        self.batch_size = 4
        self.tp_size = 8

        self.mock_tp_group = MagicMock()
        def mock_all_gather_func(tensor, dim=0):
            return torch.cat([tensor] * self.tp_size, dim=dim)
        self.mock_tp_group.all_gather = MagicMock(side_effect=mock_all_gather_func)
        self.tp_group_patch = patch("vllm.distributed.get_tp_group", return_value=self.mock_tp_group)
        self.tp_group_patch.start()

        self.npu_add_rms_norm_mock = MagicMock(side_effect=lambda x, r, w, e: (x.clone(), None, r.clone()))
        self.npu_rms_norm_mock = MagicMock(side_effect=lambda x, w, e: (x.clone(),))
        self.npu_dynamic_quant_mock = MagicMock(side_effect=lambda x: (x.clone().to(torch.int8), torch.ones(x.shape[0])))

        self.patch_add = patch("torch_npu.npu_add_rms_norm", new=self.npu_add_rms_norm_mock)
        self.patch_rms = patch("torch_npu.npu_rms_norm", new=self.npu_rms_norm_mock)
        self.patch_quant = patch("torch_npu.npu_dynamic_quant", new=self.npu_dynamic_quant_mock)

        self.patch_add.start()
        self.patch_rms.start()
        self.patch_quant.start()

        import vllm.distributed.parallel_state as ps
        ps._TP = self.mock_tp_group
        self.rms_norm = NPURMSNorm(self.hidden_size, eps=1e-6)

    def tearDown(self):
        """clear test environment"""
        self.tp_group_patch.stop()
        self.patch_add.stop()
        self.patch_rms.stop()
        self.patch_quant.stop()

        import vllm.distributed.parallel_state as ps
        ps._TP = None

    def test_forward_oot_no_residual_basic(self):
        """Case: residual=None -> Only RMSNorm"""
        x = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=None, quant_symbol=True, y_transform="AG")

        self.npu_rms_norm_mock.assert_called_once()
        self.npu_add_rms_norm_mock.assert_not_called()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_without_AG_without_quant(self):
        """Case: residual=Tensor, No AG, No Quant -> AddRMSNorm"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="", quant_symbol=False)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_not_called()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        normed_x, new_res = result
        self.assertEqual(normed_x.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_with_AG_without_quant(self):
        """Case: residual=Tensor, AG -> AddRMSNorm + AllGather"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="AG", quant_symbol=False)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_called_once()
        self.npu_dynamic_quant_mock.assert_not_called()

        normed_x, new_res = result
        expected_rows = self.batch_size * self.tp_size
        self.assertEqual(normed_x.shape, (expected_rows, self.hidden_size))
        self.assertEqual(new_res.shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_without_AG_with_quant(self):
        """Case: residual=Tensor, Quant -> AddRMSNorm + DynamicQuant"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="", quant_symbol=True)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_not_called()
        self.npu_dynamic_quant_mock.assert_called_once()

        output_dict, new_res = result
        self.assertIsInstance(output_dict, dict)
        self.assertIn("x_int8", output_dict)
        self.assertIn("pertoken_scale", output_dict)
        self.assertEqual(output_dict["x_int8"].shape, (self.batch_size, self.hidden_size))

    def test_forward_oot_with_residual_with_AG_with_quant(self):
        """Case: residual=Tensor, AG + Quant -> AddRMSNorm + AllGather + DynamicQuant"""
        x = torch.randn(self.batch_size, self.hidden_size)
        res = torch.randn(self.batch_size, self.hidden_size)

        result = self.rms_norm(x, residual=res, y_transform="AG", quant_symbol=True)

        self.npu_add_rms_norm_mock.assert_called_once()
        self.mock_tp_group.all_gather.assert_called_once()
        self.npu_dynamic_quant_mock.assert_called_once()

        quant_args = self.npu_dynamic_quant_mock.call_args[0][0]
        expected_rows = self.batch_size * self.tp_size
        self.assertEqual(quant_args.shape[0], expected_rows)
        output_dict, new_res = result
        self.assertEqual(output_dict["x_int8"].shape, (expected_rows, self.hidden_size))
        self.assertEqual(new_res.shape, (self.batch_size, self.hidden_size))


if __name__ == "__main__":
    unittest.main()