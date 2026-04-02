# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omni_npu.compilation.utils module."""

import pytest
from unittest.mock import patch, MagicMock, call
import torch

from omni_npu.compilation.utils import (
    adjust_fia_graph_params_ref,
    capture_multi_fia_graph_size,
    capture_multi_fia_v2_graph_size,
    capture_multi_fia_sink_graph_size,
)


class TestAdjustFiaGraphParamsRef:
    """Tests for adjust_fia_graph_params_ref function."""

    def test_adjust_fia_graph_params_ref_with_valid_args(self):
        """Test that adjust_fia_graph_params_ref weak references valid tensor args."""
        query = torch.tensor([1.0, 2.0, 3.0])
        key = torch.tensor([4.0, 5.0, 6.0])
        value = torch.tensor([7.0, 8.0, 9.0])
        query_rope = torch.tensor([1.0, 0.0])
        key_rope = torch.tensor([0.0, 1.0])
        block_table = torch.tensor([0, 1, 2], dtype=torch.int32)
        atten_mask = torch.tensor([1.0, 0.0])

        const_args = {
            "query": query,
            "key": key,
            "value": value,
            "query_rope": query_rope,
            "key_rope": key_rope,
            "block_table": block_table,
            "atten_mask": atten_mask,
            "other_arg": "not_adjusted",
        }

        # 假设 adjust_list 内部定义了这些键
        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.side_effect = lambda x: f"weak_ref_{x}"

            adjust_fia_graph_params_ref(const_args)

            # Verify that weak_ref_tensors was called for all adjust_list items
            # 注意：这里假设 adjust_list 包含这 7 个键。如果实现中 adjust_list 是硬编码的，这个断言是安全的。
            adjust_calls = 7 
            assert mock_weak_ref.call_count == adjust_calls

            # Verify that the const_args were updated with weak references
            for adjust_item in [
                "query", "key", "value", "query_rope", "key_rope", "block_table", "atten_mask"
            ]:
                assert const_args[adjust_item].startswith("weak_ref_")

            # Verify that non-adjust items are unchanged
            assert const_args["other_arg"] == "not_adjusted"

    def test_adjust_fia_graph_params_ref_with_none_values(self):
        """Test that adjust_fia_graph_params_ref handles None values correctly."""
        const_args = {
            "query": None,
            "key": torch.tensor([1.0, 2.0]),
            "value": None,
        }

        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.return_value = "weak_ref"

            adjust_fia_graph_params_ref(const_args)

            # weak_ref_tensors should only be called for non-None values
            assert mock_weak_ref.call_count == 1
            assert const_args["query"] is None
            assert const_args["key"] == "weak_ref"
            assert const_args["value"] is None

    def test_adjust_fia_graph_params_ref_with_partial_args(self):
        """Test that adjust_fia_graph_params_ref works with only some adjust_list items present."""
        const_args = {
            "query": torch.tensor([1.0]),
            "value": torch.tensor([2.0]),
            "other": torch.tensor([3.0]),
        }

        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
            mock_weak_ref.return_value = "weak_ref"

            adjust_fia_graph_params_ref(const_args)

            # Only 'query' and 'value' are in the default adjust_list
            assert mock_weak_ref.call_count == 2 
            assert const_args["query"] == "weak_ref"
            assert const_args["value"] == "weak_ref"
            # 'other' is not in adjust_list, so it remains untouched
            assert torch.equal(const_args["other"], torch.tensor([3.0]))

    def test_adjust_fia_graph_params_ref_empty_args(self):
        """Test that adjust_fia_graph_params_ref handles empty dict."""
        const_args = {}

        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
            adjust_fia_graph_params_ref(const_args)
            mock_weak_ref.assert_not_called()


class TestCaptureMultiFiaGraphSize:
    """Tests for capture_multi_fia_graph_size function."""

    @pytest.fixture(autouse=True)
    def setup_graph_params(self):
        """Setup graph params before each test."""
        with patch("omni_npu.compilation.acl_graph.set_graph_params"):
            with patch("omni_npu.compilation.utils.get_graph_params") as mock_get:
                mock_graph_params = MagicMock()
                mock_graph_params.events = {16: {}}
                mock_graph_params.workspaces = {}
                mock_graph_params.handles = {16: {}}
                mock_graph_params.attn_params = {16: {}}
                mock_get.return_value = mock_graph_params
                yield mock_graph_params

    def test_capture_multi_fia_graph_size_new_workspace(self, setup_graph_params):
        """Test capture_multi_fia_graph_size with new workspace calculation."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 16
        const_args = {
            "query": torch.randn(2, 4, 8),
            "key": torch.randn(2, 4, 8),
        }
        workspace = torch.tensor([100.0, 200.0])
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            with patch("torch.npu.ExternalEvent") as mock_event:
                mock_event_instance = MagicMock()
                mock_event.return_value = mock_event_instance

                # 统一使用 torch_npu 前缀
                with patch("torch_npu._npu_fused_infer_attention_score_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = workspace

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces") as mock_update_ws:
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin") as mock_begin:
                                with patch("torch_npu.npu_fused_infer_attention_score.out", create=True) as mock_fia_out:
                                    with patch("torch.npu.graph_task_group_end") as mock_end:
                                        mock_handle = MagicMock()
                                        mock_end.return_value = mock_handle

                                        capture_multi_fia_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name,
                                        )

                                        mock_get_ws.assert_called_once_with(**const_args)
                                        mock_update_ws.assert_called_once_with(num_tokens, workspace)

                                        mock_event.assert_called_once()
                                        mock_event_instance.wait.assert_called_once_with(mock_stream_instance)
                                        mock_event_instance.reset.assert_called_once_with(mock_stream_instance)

                                        assert mock_weak_ref.call_count > 0

                                        mock_begin.assert_called_once_with(mock_stream_instance)
                                        mock_fia_out.assert_called_once()
                                        mock_end.assert_called_once_with(mock_stream_instance)

    def test_capture_multi_fia_graph_size_existing_workspace(self, setup_graph_params):
        """Test capture_multi_fia_graph_size with existing workspace."""
        # 更新 fixture 中的状态以模拟已有 workspace
        setup_graph_params.workspaces = {16: torch.tensor([50.0, 100.0])}
        
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 16
        const_args = {"query": torch.randn(2, 4, 8)}
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            with patch("torch.npu.ExternalEvent") as mock_event:
                mock_event_instance = MagicMock()
                mock_event.return_value = mock_event_instance

                with patch("torch_npu._npu_fused_infer_attention_score_get_max_workspace", create=True) as mock_get_ws:
                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces") as mock_update_ws:
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin"):
                                with patch("torch_npu.npu_fused_infer_attention_score.out", create=True):
                                    with patch("torch.npu.graph_task_group_end") as mock_end:
                                        mock_end.return_value = MagicMock()

                                        capture_multi_fia_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        mock_get_ws.assert_not_called()
                                        mock_update_ws.assert_not_called()

    def test_capture_multi_fia_graph_size_op_name(self, setup_graph_params):
        """Test that op_name is set correctly."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 16
        const_args = {"query": torch.randn(2, 4, 8)}
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream"):
            with patch("torch.npu.ExternalEvent"):
                with patch("torch_npu._npu_fused_infer_attention_score_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = torch.tensor([100.0])

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces"):
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin"):
                                with patch("torch_npu.npu_fused_infer_attention_score.out", create=True):
                                    with patch("torch.npu.graph_task_group_end"):
                                        capture_multi_fia_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        captured_params = setup_graph_params.attn_params[num_tokens][layer_name]
                                        assert captured_params["op_name"] == "npu_fused_infer_attention_score"


class TestCaptureMultiFiaV2GraphSize:
    """Tests for capture_multi_fia_v2_graph_size function."""

    @pytest.fixture(autouse=True)
    def setup_graph_params(self):
        """Setup graph params before each test."""
        with patch("omni_npu.compilation.acl_graph.set_graph_params"):
            with patch("omni_npu.compilation.utils.get_graph_params") as mock_get:
                mock_graph_params = MagicMock()
                mock_graph_params.events = {32: {}}
                mock_graph_params.workspaces = {}
                mock_graph_params.handles = {32: {}}
                mock_graph_params.attn_params = {32: {}}
                mock_get.return_value = mock_graph_params
                yield mock_graph_params

    def test_capture_multi_fia_v2_graph_size_new_workspace(self, setup_graph_params):
        """Test capture_multi_fia_v2_graph_size with new workspace calculation."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 32
        const_args = {
            "query": torch.randn(2, 4, 8),
            "key": torch.randn(2, 4, 8),
        }
        workspace = torch.tensor([300.0, 400.0])
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            with patch("torch.npu.ExternalEvent") as mock_event:
                mock_event_instance = MagicMock()
                mock_event.return_value = mock_event_instance

                with patch("torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = workspace

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces") as mock_update_ws:
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin") as mock_begin:
                                with patch("torch_npu.npu_fused_infer_attention_score_v2.out", create=True) as mock_fia_out:
                                    with patch("torch.npu.graph_task_group_end") as mock_end:
                                        mock_end.return_value = MagicMock()

                                        capture_multi_fia_v2_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        mock_get_ws.assert_called_once_with(**const_args)
                                        mock_update_ws.assert_called_once_with(num_tokens, workspace)
                                        mock_begin.assert_called_once_with(mock_stream_instance)
                                        mock_fia_out.assert_called_once()
                                        mock_end.assert_called_once_with(mock_stream_instance)

    def test_capture_multi_fia_v2_graph_size_op_name(self, setup_graph_params):
        """Test that op_name is set to 'npu_fused_infer_attention_score_v2'."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 32
        const_args = {"query": torch.randn(2, 4, 8)}
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream"):
            with patch("torch.npu.ExternalEvent"):
                with patch("torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = torch.tensor([100.0])

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces"):
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin"):
                                with patch("torch_npu.npu_fused_infer_attention_score_v2.out", create=True):
                                    with patch("torch.npu.graph_task_group_end"):
                                        capture_multi_fia_v2_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        captured_params = setup_graph_params.attn_params[num_tokens][layer_name]
                                        assert captured_params["op_name"] == "npu_fused_infer_attention_score_v2"


class TestCaptureMultiFiaSinkGraphSize:
    """Tests for capture_multi_fia_sink_graph_size function."""

    @pytest.fixture(autouse=True)
    def setup_graph_params(self):
        """Setup graph params before each test."""
        with patch("omni_npu.compilation.acl_graph.set_graph_params"):
            with patch("omni_npu.compilation.utils.get_graph_params") as mock_get:
                mock_graph_params = MagicMock()
                mock_graph_params.events = {64: {}}
                mock_graph_params.workspaces = {}
                mock_graph_params.handles = {64: {}}
                mock_graph_params.attn_params = {64: {}}
                mock_get.return_value = mock_graph_params
                yield mock_graph_params

    def test_capture_multi_fia_sink_graph_size_new_workspace(self, setup_graph_params):
        """Test capture_multi_fia_sink_graph_size with new workspace calculation."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 64
        const_args = {
            "query": torch.randn(2, 4, 8),
            "key": torch.randn(2, 4, 8),
        }
        workspace = torch.tensor([500.0, 600.0])
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            with patch("torch.npu.ExternalEvent") as mock_event:
                mock_event_instance = MagicMock()
                mock_event.return_value = mock_event_instance

                with patch("torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = workspace

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces") as mock_update_ws:
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin") as mock_begin:
                                with patch("torch.ops.custom.npu_fused_infer_attention_sink.out", create=True) as mock_fia_out:
                                    with patch("torch.npu.graph_task_group_end") as mock_end:
                                        mock_end.return_value = MagicMock()

                                        capture_multi_fia_sink_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        mock_get_ws.assert_called_once_with(**const_args)
                                        mock_update_ws.assert_called_once_with(num_tokens, workspace)

                                        mock_event.assert_called_once()
                                        mock_event_instance.wait.assert_called_once_with(mock_stream_instance)
                                        mock_event_instance.reset.assert_called_once_with(mock_stream_instance)

                                        assert mock_weak_ref.call_count > 0

                                        mock_begin.assert_called_once_with(mock_stream_instance)
                                        mock_fia_out.assert_called_once()
                                        mock_end.assert_called_once_with(mock_stream_instance)

    def test_capture_multi_fia_sink_graph_size_existing_workspace(self, setup_graph_params):
        """Test capture_multi_fia_sink_graph_size with existing workspace."""
        # 更新 fixture 状态
        setup_graph_params.workspaces = {64: torch.tensor([250.0, 300.0])}
        
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 64
        const_args = {"query": torch.randn(2, 4, 8)}
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance

            with patch("torch.npu.ExternalEvent") as mock_event:
                mock_event_instance = MagicMock()
                mock_event.return_value = mock_event_instance

                with patch("torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace", create=True) as mock_get_ws:
                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces") as mock_update_ws:
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin"):
                                with patch("torch.ops.custom.npu_fused_infer_attention_sink.out", create=True):
                                    with patch("torch.npu.graph_task_group_end") as mock_end:
                                        mock_end.return_value = MagicMock()

                                        capture_multi_fia_sink_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        mock_get_ws.assert_not_called()
                                        mock_update_ws.assert_not_called()

    def test_capture_multi_fia_sink_graph_size_op_name(self, setup_graph_params):
        """Test that op_name is set to 'npu_fused_infer_attention_sink'."""
        attn_output = torch.randn(2, 4, 8)
        softmax_lse = torch.randn(2, 4)
        num_tokens = 64
        const_args = {"query": torch.randn(2, 4, 8)}
        layer_name = "test_layer_name"

        with patch("torch_npu.npu.current_stream"):
            with patch("torch.npu.ExternalEvent"):
                with patch("torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace", create=True) as mock_get_ws:
                    mock_get_ws.return_value = torch.tensor([100.0])

                    with patch("omni_npu.compilation.utils.update_graph_params_workspaces"):
                        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_weak_ref:
                            mock_weak_ref.side_effect = lambda x: x

                            with patch("torch.npu.graph_task_group_begin"):
                                with patch("torch.ops.custom.npu_fused_infer_attention_sink.out", create=True):
                                    with patch("torch.npu.graph_task_group_end"):
                                        capture_multi_fia_sink_graph_size(
                                            attn_output, softmax_lse, num_tokens, const_args, layer_name
                                        )

                                        captured_params = setup_graph_params.attn_params[num_tokens][layer_name]
                                        assert captured_params["op_name"] == "npu_fused_infer_attention_sink"