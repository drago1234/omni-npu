# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pytest
from unittest.mock import patch, MagicMock
from omni_npu.connector.utils import get_p_start_rank, get_config_from_dict_or_env


class TestGetPStartRank:
    """Test get_p_start_rank function with various scenarios"""

    def test_valid_parameters(self):
        """Test with valid parameters that should succeed"""
        result = get_p_start_rank(
            p_tp_size=4, p_dp_size=1, d_tp_size=2, d_dp_size=2,
            d_node_num=2, cur_d_node=0, cur_d_rank=0
        )
        assert isinstance(result, int)
        assert result >= 0

        # Test another valid combination
        result2 = get_p_start_rank(
            p_tp_size=8, p_dp_size=1, d_tp_size=4, d_dp_size=1,
            d_node_num=1, cur_d_node=0, cur_d_rank=0
        )
        assert isinstance(result2, int)
        assert result2 >= 0

    def test_p_dp_size_not_1(self):
        """Test that p_dp_size must be 1"""
        with pytest.raises(ValueError, match="p_dp_size must be 1"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=2, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=0
            )

    def test_negative_p_tp_size(self):
        """Test negative p_tp_size"""
        with pytest.raises(ValueError, match="p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive"):
            get_p_start_rank(
                p_tp_size=-1, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=0
            )

    def test_zero_d_tp_size(self):
        """Test zero d_tp_size"""
        with pytest.raises(ValueError, match="p_tp_size, d_tp_size, d_dp_size, d_node_num must be positive"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=1, d_tp_size=0, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=0
            )

    def test_negative_cur_d_node(self):
        """Test negative current decode node"""
        with pytest.raises(ValueError, match="cur_d_node < 0 or cur_d_node >= d_node_num"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=-1, cur_d_rank=0
            )

    def test_cur_d_node_out_of_range(self):
        """Test current decode node out of range"""
        with pytest.raises(ValueError, match="cur_d_node < 0 or cur_d_node >= d_node_num"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=2, cur_d_rank=0
            )

    def test_negative_cur_d_rank(self):
        """Test negative current decode rank"""
        with pytest.raises(ValueError, match="cur_d_rank < 0"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=-1
            )

    def test_cur_d_rank_out_of_range(self):
        """Test current decode rank out of range"""
        with pytest.raises(ValueError, match="cur_d_rank >= devices_per_node"):
            get_p_start_rank(
                p_tp_size=4, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=5
            )

    def test_p_tp_size_not_divisible_by_kv_group_size(self):
        """Test when p_tp_size is not divisible by kv_group_size"""
        with pytest.raises(ValueError, match="p_tp_size % kv_group_size != 0"):
            get_p_start_rank(
                p_tp_size=5, p_dp_size=1, d_tp_size=2, d_dp_size=2,
                d_node_num=2, cur_d_node=0, cur_d_rank=0
            )

    def test_edge_case_single_device(self):
        """Test edge case with single device"""
        result = get_p_start_rank(
            p_tp_size=1, p_dp_size=1, d_tp_size=1, d_dp_size=1,
            d_node_num=1, cur_d_node=0, cur_d_rank=0
        )
        assert result == 0

    def test_complex_scenario(self):
        """Test complex scenario with multiple nodes and devices"""
        result = get_p_start_rank(
            p_tp_size=16, p_dp_size=1, d_tp_size=4, d_dp_size=2,
            d_node_num=4, cur_d_node=1, cur_d_rank=3
        )
        assert isinstance(result, int)
        assert 0 <= result < 16


class TestGetConfigFromDictOrEnv:
    """Test get_config_from_dict_or_env function with various scenarios"""

    def test_env_variable_priority(self):
        """Test that environment variable has highest priority"""
        with patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            config = {'test_var': 'dict_value'}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default_value', str
            )
            assert result == 'env_value'

    def test_dict_config_when_no_env(self):
        """Test dictionary config when no environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            config = {'test_var': 'dict_value'}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default_value', str
            )
            assert result == 'dict_value'

    def test_object_config_when_no_env(self):
        """Test object config when no environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            class ConfigObject:
                def __init__(self):
                    self.test_var = 'object_value'

            config = ConfigObject()
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default_value', str
            )
            assert result == 'object_value'

    def test_default_value_when_no_env_or_config(self):
        """Test default value when no environment variable or config"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default_value', str
            )
            assert result == 'default_value'

    def test_no_default_value_raises_error(self):
        """Test that error is raised when no value found and no default"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            with pytest.raises(ValueError, match="ENV TEST_VAR or args test_var should not be None"):
                get_config_from_dict_or_env(
                    config, 'test_var', 'TEST_VAR', None, str
                )

    def test_type_conversion(self):
        """Test that value is converted to specified type"""
        with patch.dict(os.environ, {'TEST_VAR': '42'}):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 0, int
            )
            assert result == 42
            assert isinstance(result, int)

    def test_type_conversion_with_default(self):
        """Test type conversion with default value"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', '100', int
            )
            assert result == 100
            assert isinstance(result, int)

    def test_empty_dict_config(self):
        """Test with empty dictionary config"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default', str
            )
            assert result == 'default'

    def test_none_config_object(self):
        """Test with None config object"""
        with patch.dict(os.environ, {}, clear=True):
            config = None
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default', str
            )
            assert result == 'default'

    def test_config_object_without_attribute(self):
        """Test config object without the requested attribute"""
        with patch.dict(os.environ, {}, clear=True):
            class ConfigObject:
                def __init__(self):
                    self.other_var = 'other_value'

            config = ConfigObject()
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default_value', str
            )
            assert result == 'default_value'


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_get_p_start_rank_large_numbers(self):
        """Test get_p_start_rank with large numbers"""
        result = get_p_start_rank(
            p_tp_size=64, p_dp_size=1, d_tp_size=8, d_dp_size=4,
            d_node_num=8, cur_d_node=4, cur_d_rank=15
        )
        assert isinstance(result, int)
        assert 0 <= result < 64

    def test_get_config_from_dict_or_env_special_chars(self):
        """Test get_config_from_dict_or_env with special characters"""
        with patch.dict(os.environ, {'TEST_VAR': 'special@value#123'}):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', 'default', str
            )
            assert result == 'special@value#123'

    def test_get_config_from_dict_or_env_boolean_type(self):
        """Test get_config_from_dict_or_env with boolean type conversion"""
        with patch.dict(os.environ, {'TEST_VAR': 'true'}):
            config = {}
            result = get_config_from_dict_or_env(
                config, 'test_var', 'TEST_VAR', False, lambda x: x.lower() == 'true'
            )
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])