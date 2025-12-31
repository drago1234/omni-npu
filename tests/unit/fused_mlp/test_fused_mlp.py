import unittest
from unittest.mock import MagicMock, patch
import torch
from omni_npu.v1.fused_mlp import FusedMLP, UnquantizedFusedMLPMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class TestFusedMLP(unittest.TestCase):

    @patch("omni_npu.v1.fused_mlp.get_tensor_model_parallel_world_size", return_value=1)
    @patch("omni_npu.v1.fused_mlp.get_tensor_model_parallel_rank", return_value=0)
    @patch("omni_npu.v1.fused_mlp.tensor_model_parallel_all_gather", side_effect=lambda x, dim=0: x)
    @patch("omni_npu.v1.fused_mlp.tensor_model_parallel_all_reduce", side_effect=lambda x: x)
    @patch("omni_npu.v1.fused_mlp.tensor_model_parallel_reduce_scatter", side_effect=lambda x: x)
    @patch("omni_npu.v1.fused_mlp.get_tp_group")
    @patch("omni_npu.v1.fused_mlp.MergedColumnParallelFlashCommLinear")
    @patch("omni_npu.v1.fused_mlp.RowParallelFlashCommLinear")
    @patch("omni_npu.v1.fused_mlp.SiluAndMul")
    def test_forward_unquantized(self,
                                 mock_act_fn,
                                 mock_row_linear,
                                 mock_col_linear,
                                 mock_tp_group,
                                 mock_reduce_scatter,
                                 mock_all_reduce,
                                 mock_all_gather,
                                 mock_tp_rank,
                                 mock_tp_size):

        # Mock linear layers的forward
        dummy_tensor = torch.randn(2, 4)
        mock_col_linear.return_value = MagicMock(return_value=(dummy_tensor, None))
        mock_row_linear.return_value = MagicMock(return_value=(dummy_tensor, None))
        mock_act_fn.return_value = lambda x: x
        mock_tp_group.return_value.all_to_all = lambda x: x

        # 初始化 FusedMLP
        mlp = FusedMLP(hidden_size=4, intermediate_size=8, hidden_act="silu", quant_config=None)
        self.assertIsInstance(mlp.quant_method, UnquantizedFusedMLPMethod)

        # 输入 tensor
        x = torch.randn(2, 4)

        # 调用 forward
        output = mlp.forward(x)
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.shape, x.shape)

    @patch("omni_npu.v1.fused_mlp.get_tensor_model_parallel_world_size", return_value=2)
    @patch("omni_npu.v1.fused_mlp.get_tensor_model_parallel_rank", return_value=1)
    @patch("omni_npu.v1.fused_mlp.tensor_model_parallel_all_reduce", side_effect=lambda x: x)
    @patch("omni_npu.v1.fused_mlp.tensor_model_parallel_reduce_scatter", side_effect=lambda x: x)
    def test_forward_with_tp_reduce(self,
                                    mock_reduce_scatter,
                                    mock_all_reduce,
                                    mock_tp_rank,
                                    mock_tp_size):

        mlp = FusedMLP(hidden_size=4, intermediate_size=8, hidden_act="silu", quant_config=None)
        x = torch.randn(2, 4)

        output = mlp.forward(x)
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.shape, x.shape)

if __name__ == "__main__":
    unittest.main()
