# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for config loader that do NOT require actual NPU hardware or file system.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch, mock_open
from contextlib import contextmanager
import sys
import json
import os

import pytest
import torch
from dataclasses import asdict


@contextmanager
def mock_dependencies():
    """Context manager to mock external dependencies"""
    with patch('torch.npu.get_device_name', return_value='Ascend910B'):
        with patch.dict(os.environ, {'ROLE': 'prefill', 'PREFILL_POD_NUM': '1', 'DECODE_POD_NUM': '1'}):
            yield


@pytest.mark.unit
class TestConfigLoaderUnit(unittest.TestCase):
    """Unit tests for config loader (no NPU hardware or file system required)"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock torch_npu availability
        self.torch_npu_mock = MagicMock()
        sys.modules['torch_npu'] = self.torch_npu_mock
        
        # Mock vllm dependencies
        self.vllm_logger_mock = MagicMock()
        sys.modules['vllm'] = MagicMock()
        sys.modules['vllm.logger'] = MagicMock()
        sys.modules['vllm.logger'].init_logger = MagicMock(return_value=self.vllm_logger_mock)
        
        # Mock features module
        self.features_mock = MagicMock()
        sys.modules['omni_npu.v1.models.config_loader.features'] = self.features_mock

    def tearDown(self):
        """Clean up after tests"""
        # Remove mocked modules
        modules_to_remove = [
            'torch_npu',
            'vllm',
            'vllm.logger',
            'omni_npu.v1.models.config_loader.features',
        ]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def test_model_operator_opt_config_post_init_enable_prefetch_true(self):
        """Test ModelOperatorOptConfig __post_init__ when enable_prefetch is True"""
        from omni_npu.v1.models.config_loader.loader import ModelOperatorOptConfig
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.v1.models.config_loader.loader.logger', mock_logger):
            config = ModelOperatorOptConfig(enable_prefetch=True)
        
        # When enable_prefetch is True, prefetch values should remain default
        self.assertEqual(config.expert_gate_up_prefetch, 50)
        self.assertEqual(config.attn_prefetch, 96)
        # Verify warning was not logged
        mock_logger.warning.assert_not_called()

    def test_model_operator_opt_config_post_init_enable_prefetch_false(self):
        """Test ModelOperatorOptConfig __post_init__ when enable_prefetch is False"""
        from omni_npu.v1.models.config_loader.loader import ModelOperatorOptConfig
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.v1.models.config_loader.loader.logger', mock_logger):
            config = ModelOperatorOptConfig(enable_prefetch=False)
        
        # When enable_prefetch is False, prefetch values should be set to 0
        self.assertEqual(config.expert_gate_up_prefetch, 0)
        self.assertEqual(config.expert_down_prefetch, 0)
        self.assertEqual(config.attn_prefetch, 0)
        self.assertEqual(config.dense_mlp_prefetch, 0)
        self.assertEqual(config.lm_head_prefetch, 0)
        self.assertEqual(config.shared_expert_gate_up_prefetch, 0)
        self.assertEqual(config.shared_expert_down_prefetch, 0)
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0."
        )

    def test_model_operator_opt_config_post_init_conflicting_comm_config(self):
        """Test ModelOperatorOptConfig __post_init__ raises error for conflicting comm config"""
        from omni_npu.v1.models.config_loader.loader import ModelOperatorOptConfig
        
        with self.assertRaises(ValueError) as context:
            ModelOperatorOptConfig(enable_pipeline_comm=True, enable_round_pipeline_comm=True)
        
        self.assertIn("Conflicting communication configuration", str(context.exception))

    def test_model_operator_opt_config_post_init_unquant_bmm_nz(self):
        """Test ModelOperatorOptConfig __post_init__ sets torch config for unquant_bmm_nz"""
        from omni_npu.v1.models.config_loader.loader import ModelOperatorOptConfig
        
        config = ModelOperatorOptConfig(unquant_bmm_nz=True)
        
        # Verify torch.npu.config.allow_internal_format was set
        self.torch_npu_mock.config.allow_internal_format = True

    def test_parse_hf_config_deepseek_v3(self):
        """Test parse_hf_config for deepseek_v3 model"""
        from omni_npu.v1.models.config_loader.loader import parse_hf_config
        
        # Mock hf_config
        hf_config_mock = MagicMock()
        hf_config_mock.model_type = "deepseek_v3"
        hf_config_mock.quantization_config = {
            'format': 'int-quantized',
            'config_groups': {
                'group_0': {
                    'weights': {'num_bits': 8},
                    'input_activations': {'num_bits': 8}
                }
            },
            'kv_cache_scheme': 'default'
        }
        
        model_name, quant_type = parse_hf_config(hf_config_mock)
        
        self.assertEqual(model_name, "deepseek_v3")
        self.assertEqual(quant_type, "w8a8c16")

    def test_parse_hf_config_bf16(self):
        """Test parse_hf_config for BF16 model without quantization"""
        from omni_npu.v1.models.config_loader.loader import parse_hf_config
        
        # Mock hf_config without quantization
        hf_config_mock = MagicMock()
        hf_config_mock.model_type = "some_model"
        del hf_config_mock.quantization_config  # Simulate no quantization
        
        model_name, quant_type = parse_hf_config(hf_config_mock)
        
        self.assertEqual(model_name, "some_model")
        self.assertEqual(quant_type, "bf16")

    def test_filter_dict_by_dataclass(self):
        """Test filter_dict_by_dataclass filters valid keys"""
        from omni_npu.v1.models.config_loader.loader import filter_dict_by_dataclass, ModelOperatorOptConfig
        
        data_dict = {
            'enable_prefetch': False,
            'invalid_key': 'value',
            'expert_gate_up_prefetch': 100
        }
        
        filtered = filter_dict_by_dataclass(ModelOperatorOptConfig, data_dict)
        
        self.assertIn('enable_prefetch', filtered)
        self.assertIn('expert_gate_up_prefetch', filtered)
        self.assertNotIn('invalid_key', filtered)

    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_loader_configs_data(self, mock_file):
        """Test _loader_configs_data loads JSON correctly"""
        from omni_npu.v1.models.config_loader.loader import _loader_configs_data
        
        result = _loader_configs_data('dummy_path.json')
        
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once_with('dummy_path.json', 'r')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_loader_configs_data_invalid_json(self, mock_file):
        """Test _loader_configs_data raises error for invalid JSON"""
        from omni_npu.v1.models.config_loader.loader import _loader_configs_data
        
        with self.assertRaises(RuntimeError) as context:
            _loader_configs_data('dummy_path.json')
        
        self.assertIn("Invalid JSON format", str(context.exception))

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "model_parallel_config": {"dense_mlp_tp_size": 2},
        "operator_optimization_config": {"enable_prefetch": False}
    }))
    def test_init_model_extra_config(self, mock_file, mock_exists):
        """Test _init_model_extra_config initializes configs correctly"""
        from omni_npu.v1.models.config_loader.loader import _init_model_extra_config, TaskConfig, model_extra_config
        
        task_config = TaskConfig()
        
        # Mock _get_best_practice_config to return config data
        with patch('omni_npu.v1.models.config_loader.loader._get_best_practice_config', return_value={
            "model_parallel_config": {"dense_mlp_tp_size": 2},
            "operator_optimization_config": {"enable_prefetch": False}
        }):
            _init_model_extra_config(task_config)
            
            self.assertEqual(model_extra_config.parall_config.dense_mlp_tp_size, 2)
            self.assertEqual(model_extra_config.operator_opt_config.enable_prefetch, False)
            # Verify __post_init__ was called and prefetch values were reset
            self.assertEqual(model_extra_config.operator_opt_config.expert_gate_up_prefetch, 0)

    def test_update_task_config(self):
        """Test update_task_config updates task_config correctly"""
        from omni_npu.v1.models.config_loader.loader import update_task_config, model_extra_config
        
        update_task_config(model_name="test_model", quant_type="w8a8")
        
        self.assertEqual(model_extra_config.task_config.model_name, "test_model")
        self.assertEqual(model_extra_config.task_config.quant_type, "w8a8")

    def test_print_model_config(self):
        """Test _print_model_config logs config correctly"""
        from omni_npu.v1.models.config_loader.loader import _print_model_config, model_extra_config
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.v1.models.config_loader.loader.logger', mock_logger):
            _print_model_config()
        
        # Verify logger.info was called
        mock_logger.info.assert_called()

    def test_load_model_extra_config(self):
        """Test load_model_extra_config function with mocked dependencies"""
        from omni_npu.v1.models.config_loader.loader import load_model_extra_config
        
        # Mock all required classes and dependencies
        mock_model_config = MagicMock()
        mock_model_config.hf_config.model_type = "deepseek_v3"
        mock_model_config.hf_config.quantization_config = {
            'format': 'int-quantized',
            'config_groups': {'group_0': {'weights': {'num_bits': 8}, 'input_activations': {'num_bits': 8}}},
            'kv_cache_scheme': 'default'
        }
        mock_model_config.enforce_eager = False
        
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = None
        mock_vllm_config.npu_compilation_config.use_gegraph = False
        mock_vllm_config.parallel_config.enable_eplb = False
        
        mock_scheduler_config = MagicMock()
        mock_scheduler_config.enable_chunked_prefill = False
        
        # Mock external dependencies
        with patch('omni_npu.v1.models.config_loader.loader.parse_hf_config', return_value=('deepseek_v3', 'w8a8c16')), \
             patch('omni_npu.v1.models.config_loader.loader.update_task_config') as mock_update, \
             patch('omni_npu.v1.models.config_loader.loader._validate_config') as mock_validate, \
             patch('omni_npu.v1.models.config_loader.loader._print_model_config') as mock_print:
            
            load_model_extra_config(mock_model_config, mock_vllm_config, mock_scheduler_config)
            
            # Verify that update_task_config was called
            mock_update.assert_called_once()
            # Verify that _validate_config was called
            mock_validate.assert_called_once()
            # Verify that _print_model_config was called
            mock_print.assert_called_once()


if __name__ == '__main__':
    unittest.main()
