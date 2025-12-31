import unittest
from unittest.mock import MagicMock, patch
import torch
from src.omni_npu.v1.fused_mlp_methods import W8A8Int8MlpMethod


class DummyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.weight_scale = torch.nn.Parameter(torch.ones(out_features))
        self.skip_bias_add = False
        self.tp_rank = 0
        self.orig_dtype = torch.float32
        self.x_transform = None
        self.x_dim = 0
        self.y_transform = None
        self.y_dim = 0

class DummyLayer(torch.nn.Module):
    def __init__(self, in_features=4, hidden_features=8, out_features=4):
        super().__init__()
        self.layer_name = "dummy_layer"
        self.gate_up_proj = DummyLinear(in_features, hidden_features)
        self.down_proj = DummyLinear(hidden_features, out_features)

class TestW8A8Int8MlpMethod(unittest.TestCase):
    @patch("src.omni_npu.v1.fused_mlp_methods.torch_npu.npu_dynamic_quant")
    @patch("src.omni_npu.v1.fused_mlp_methods.torch_npu.npu_quant_matmul")
    @patch("src.omni_npu.v1.fused_mlp_methods.torch_npu.npu_dequant_swiglu_quant")
    @patch("src.omni_npu.v1.fused_mlp_methods.layer_parallel_all_reduce")
    @patch("src.omni_npu.v1.fused_mlp_methods.layer_parallel_all_gather")
    @patch("src.omni_npu.v1.fused_mlp_methods.layer_parallel_reduce_scatter")
    @patch("src.omni_npu.v1.fused_mlp_methods.layer_parallel_all2all_single")
    @patch("src.omni_npu.v1.fused_mlp_methods.get_npu_execution_type", return_value=None)
    def test_apply(self,
                   mock_get_npu_execution_type,
                   mock_all2all,
                   mock_reduce_scatter,
                   mock_all_gather,
                   mock_all_reduce,
                   mock_dequant,
                   mock_quant_matmul,
                   mock_dynamic_quant):

        mock_dynamic_quant.side_effect = lambda x, smooth_scales=None: (x, torch.ones_like(x))
        mock_quant_matmul.side_effect = lambda **kwargs: kwargs["x1"]
        mock_dequant.side_effect = lambda *args, **kwargs: (args[0], torch.ones_like(args[0]))
        mock_all2all.side_effect = lambda x, *args, **kwargs: x
        mock_all_gather.side_effect = lambda x, *args, **kwargs: x
        mock_all_reduce.side_effect = lambda x, *args, **kwargs: x
        mock_reduce_scatter.side_effect = lambda x, *args, **kwargs: x

        layer = DummyLayer()
        method = W8A8Int8MlpMethod(quant_config={})

        x = torch.randn(2, 4)

        output = method.apply(layer, x)

        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.shape[0], x.shape[0])
        self.assertEqual(output.shape[1], layer.down_proj.weight.shape[0])

        mock_dynamic_quant.assert_called()
        mock_quant_matmul.assert_called()
        mock_dequant.assert_called()

if __name__ == "__main__":
    unittest.main()
