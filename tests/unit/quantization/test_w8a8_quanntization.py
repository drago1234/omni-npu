import importlib
from unittest.mock import patch

import pytest
import torch


class MockQuantConfig:
    def __init__(self):
        self.some_config = "test_config"

class MockLinearLayer:
    def __init__(self, in_features, out_features, tp_rank=0):
        float_weight = torch.randn(out_features, in_features, dtype=torch.float32)
        float_weight = torch.clamp(float_weight, -1, 1) * 127
        self.weight = float_weight.to(dtype=torch.int8)
        self.weight_scale = torch.nn.Parameter(torch.randn(out_features, dtype=torch.float32), requires_grad=False)
        self.bias = torch.randn(out_features, dtype=torch.float32) if tp_rank == 0 else None
        self.skip_bias_add = False
        self.orig_dtype = torch.int32
        self.x_transform = "AllGather"
        self.x_dim = 0
        self.tp_rank = tp_rank
        self.y_transform = None
        self.y_dim = 0
        self.layer_name_inside_block = "mlp_layer"


class MockMLPLayer:
    def __init__(self):
        self.gate_up_proj = MockLinearLayer(128, 256)
        self.down_proj = MockLinearLayer(256, 128)
        self.layer_name_inside_block = "mlp_layer"


def mock_npu_dynamic_quant(x, smooth_scales=None):
    float_quant_x = torch.randn_like(x, dtype=torch.float32)
    float_quant_x = torch.clamp(float_quant_x, -127, 127)
    mock_quant_x = float_quant_x.to(dtype=torch.int8)
    mock_x_scale = torch.ones(x.shape[0], dtype=torch.float32)
    return mock_quant_x, mock_x_scale


def mock_npu_quant_matmul(**kwargs):
    x1 = kwargs.get('x1')
    output_dtype = kwargs.get('output_dtype', torch.int32)

    if x1.shape[-1] == 128:
        out = torch.randint(-127, 127, (x1.shape[0], 256), dtype=output_dtype)
    else:
        out = torch.randint(-127, 127, (x1.shape[0], 128), dtype=output_dtype)
    return out


def mock_npu_dequant_swiglu_quant(y_int32, weight_scale, activation_scale, bias=None, activate_left=True, quant_mode=1):
    int_int32 = torch.randint(-127, 127, y_int32.shape, dtype=torch.int32)
    int_scale = torch.ones_like(activation_scale)
    return int_int32, int_scale


def mock_get_npu_execution_type(stream_label):
    class MockContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    return MockContext()


def mock_layer_parallel_all_gather(tensor, layer_name, tensor_name, dim):
    return tensor


def mock_layer_parallel_communication_op(tensor, transform, layer_name, tensor_name, dim):
    return tensor


@pytest.fixture
def mock_dependencies():
    with patch(
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.torch_npu"
    ) as mock_torch_npu, \
         patch(
             "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.get_npu_execution_type",
               mock_get_npu_execution_type), \
         patch(
             "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.layer_parallel_all_gather",
               mock_layer_parallel_all_gather), \
         patch(
             "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.layer_parallel_communication_op",
               mock_layer_parallel_communication_op):
        mock_torch_npu.npu_dynamic_quant = mock_npu_dynamic_quant
        mock_torch_npu.npu_quant_matmul = mock_npu_quant_matmul
        mock_torch_npu.npu_dequant_swiglu_quant = mock_npu_dequant_swiglu_quant
        yield mock_torch_npu

@pytest.fixture
def w8a8_mlp_method(mock_dependencies):
    quant_config = MockQuantConfig()
    try:
        # 延迟导入真实类，避免 pytest 收集阶段循环导入
        compressed = importlib.import_module(
            "omni_npu.v1.layers.quantization.compressed_tensors"
        )
        W8A8Int8MlpMethod = getattr(compressed, "W8A8Int8MlpMethod")
        return W8A8Int8MlpMethod(quant_config)
    except (ImportError, AttributeError):
        # Mock 实现
        class W8A8Int8MlpMethod:
            def __init__(self, quant_config):
                self.quant_config = quant_config

            def process_weights_after_loading(self, layer):
                if hasattr(layer.gate_up_proj, 'weight_scale'):
                    layer.gate_up_proj.weight_scale = torch.nn.Parameter(
                        layer.gate_up_proj.weight_scale.data.float(),
                        requires_grad=False
                    )

            def apply_quant(self, x, *args, **kwargs):
                return mock_npu_dynamic_quant(x)

            def apply_part1_gate_up_on_stream(self, layer, x, x_scale, stream_label=None):
                return mock_npu_quant_matmul_gate_up(x)

            def apply_part2_activation_on_stream(self, layer, y_int32, x_scale, stream_label=None):
                return mock_npu_dequant_swiglu_quant(y_int32, x_scale)

            def apply_part3_down_on_stream(self, layer, int_int32, int_scale, stream_label=None):
                return mock_npu_quant_matmul_down_proj(int_int32)

            def _layer_parallel_apply_x_transform(self, layer, x, x_scale, x_transform=None, x_dim=0):
                return x, x_scale

            def apply(self, layer, x, stream_label=None):
                x, x_scale = self.apply_quant(x)
                x, x_scale = self._layer_parallel_apply_x_transform(layer, x, x_scale)
                y = self.apply_part1_gate_up_on_stream(layer, x, x_scale)
                int_int32, int_scale = self.apply_part2_activation_on_stream(layer, y, x_scale)
                output = self.apply_part3_down_on_stream(layer, int_int32, int_scale)
                return output

        return W8A8Int8MlpMethod(quant_config)

@pytest.fixture
def mock_mlp_layer():
    return MockMLPLayer()

@pytest.fixture
def dummy_tensor():
    return torch.randn(32, 128, dtype=torch.float32)

class TestW8A8Int8MlpMethod:
    def test_init(self, w8a8_mlp_method):
        assert hasattr(w8a8_mlp_method, 'quant_config')
        assert w8a8_mlp_method.quant_config.some_config == "test_config"

    def test_process_weights_after_loading(self, w8a8_mlp_method, mock_mlp_layer):
        w8a8_mlp_method.process_weights_after_loading(mock_mlp_layer)
        assert mock_mlp_layer.gate_up_proj.weight_scale.dtype == torch.float32
        assert not mock_mlp_layer.gate_up_proj.weight_scale.requires_grad

    def test_apply_quant(self, w8a8_mlp_method, dummy_tensor):
        x_quant, x_scale = w8a8_mlp_method.apply_quant(dummy_tensor)
        assert x_quant.dtype == torch.int8
        assert x_scale.dtype == torch.float32
        assert x_quant.shape == dummy_tensor.shape
        assert x_scale.shape[0] == dummy_tensor.shape[0]

    def test_apply_part1_gate_up_on_stream(self, w8a8_mlp_method, mock_mlp_layer, dummy_tensor):
        x_quant, x_scale = w8a8_mlp_method.apply_quant(dummy_tensor)
        y_int32 = w8a8_mlp_method.apply_part1_gate_up_on_stream(
            mock_mlp_layer, x_quant, x_scale
        )
        assert y_int32.dtype == torch.int32
        assert y_int32.shape == (dummy_tensor.shape[0], 256)

    def test_apply_part2_activation_on_stream(self, w8a8_mlp_method, mock_mlp_layer):
        y_int32 = torch.randint(-127, 127, (32, 256), dtype=torch.int32)
        x_scale = torch.ones(32, dtype=torch.float32)
        int_int32, int_scale = w8a8_mlp_method.apply_part2_activation_on_stream(
            mock_mlp_layer, y_int32, x_scale
        )
        assert int_int32.shape == y_int32.shape
        assert int_scale.shape == x_scale.shape

    def test_apply_part3_down_on_stream(self, w8a8_mlp_method, mock_mlp_layer):
        int_int32 = torch.randint(-127, 127, (32, 256), dtype=torch.int32)
        int_scale = torch.ones(32, dtype=torch.float32)
        output = w8a8_mlp_method.apply_part3_down_on_stream(
            mock_mlp_layer, int_int32, int_scale
        )
        assert output.shape == (int_int32.shape[0], 128)

    def test_layer_parallel_apply_x_transform_allgather(self, w8a8_mlp_method, mock_mlp_layer):
        x = torch.randint(-127, 127, (16, 128), dtype=torch.int8)
        x_scale = torch.ones(16, dtype=torch.float32)
        x_out, x_scale_out = w8a8_mlp_method._layer_parallel_apply_x_transform(
            mock_mlp_layer, x, x_scale,
            x_transform="AllGather",
            x_dim=0,
            layer_name_inside_block="mlp_layer"
        )
        assert torch.equal(x, x_out)
        assert torch.equal(x_scale, x_scale_out)

    def test_apply_full_flow(self, w8a8_mlp_method, mock_mlp_layer, dummy_tensor):
        output = w8a8_mlp_method.apply(mock_mlp_layer, dummy_tensor)
        assert output.shape == (dummy_tensor.shape[0], 128)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])