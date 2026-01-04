# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for layer parallel helper utilities.
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from tests.unit.distributed.test_communicator import VLLM_MODULES_TO_MOCK


class TestCommunicationOpExtensions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Mock torch_npu availability
        self.torch_npu_mock = MagicMock()
        sys.modules['torch_npu'] = self.torch_npu_mock
        
        # Mock all required vLLM modules
        for module_name in VLLM_MODULES_TO_MOCK:
            if module_name not in sys.modules:
                sys.modules[module_name] = MagicMock()

        # Ensure vllm.distributed and submodules exist as packages
        if 'vllm.distributed' not in sys.modules:
            sys.modules['vllm.distributed'] = MagicMock()
        if 'vllm.distributed.parallel_state' not in sys.modules:
            sys.modules['vllm.distributed.parallel_state'] = MagicMock()
        if 'vllm.distributed.device_communicators.cuda_communicator' not in sys.modules:
            sys.modules['vllm.distributed.device_communicators.cuda_communicator'] = MagicMock()

        # Import the actual extensions and bind to self
        from omni_npu.v1.distributed import (
            communication_op_ext,
            parallel_state_ext,
        )
        self.communication_op_ext = communication_op_ext
        self.parallel_state_ext = parallel_state_ext

        # Define a mock CudaCommunicator class
        class MockCudaCommunicator:
            def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
                self.cpu_group = cpu_group
                self.device = device
                self.device_group = device_group
                self.unique_name = unique_name
                self.world_size = 1
                self.rank_in_group = 0
                self.ranks = [0]

        # Patch it into the mocked module
        sys.modules['vllm.distributed.device_communicators.cuda_communicator'].CudaCommunicator = MockCudaCommunicator

    def tearDown(self):
        """Clean up after tests"""
        modules_to_remove = {
            'torch_npu',
            'omni_npu.distributed.communicator',
        } | VLLM_MODULES_TO_MOCK
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    @patch("omni_npu.v1.distributed.communication_op_ext.get_layer_parallel_group")
    def test_get_group_returns_none_when_world_size_le_one(self, mock_get_layer_group):
        group = Mock()
        group.world_size = 1
        mock_get_layer_group.return_value = group
        self.assertIsNone(self.communication_op_ext._get_group("layer"))

    @patch("omni_npu.v1.distributed.communication_op_ext._get_group", return_value=None)
    def test_all_reduce_returns_input_without_group(self, _):
        tensor = torch.ones(2, 2)
        result = self.communication_op_ext.layer_parallel_all_reduce(tensor, "layer")
        self.assertIs(result, tensor)

    @patch("omni_npu.v1.distributed.communication_op_ext._get_group")
    def test_all_reduce_uses_group_all_reduce(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.all_reduce.return_value = torch.full((2, 2), 7.0)
        mock_get_group.return_value = group

        result = self.communication_op_ext.layer_parallel_all_reduce(tensor, "layer")

        group.all_reduce.assert_called_once_with(tensor)
        torch.testing.assert_close(result, torch.full((2, 2), 7.0))

    @patch("omni_npu.v1.distributed.communication_op_ext._get_group")
    def test_all_gather_delegates_dim(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.all_gather.return_value = "out"
        mock_get_group.return_value = group

        result = self.communication_op_ext.layer_parallel_all_gather(
            tensor, "layer", dim=1
        )

        group.all_gather.assert_called_once_with(tensor, dim=1)
        self.assertEqual(result, "out")

    @patch("omni_npu.v1.distributed.communication_op_ext._get_group")
    def test_reduce_scatter_delegates_dim(self, mock_get_group):
        tensor = torch.randn(2, 2)
        group = Mock()
        group.reduce_scatter.return_value = "out"
        mock_get_group.return_value = group

        result = self.communication_op_ext.layer_parallel_reduce_scatter(
            tensor, "layer", dim=0
        )

        group.reduce_scatter.assert_called_once_with(tensor, dim=0)
        self.assertEqual(result, "out")


class TestParallelStateExtensions(unittest.TestCase):
    def setUp(self):
        """Import parallel_state_ext and bind to self"""
        from omni_npu.v1.distributed import parallel_state_ext
        self.parallel_state_ext = parallel_state_ext

    def test_normalize_comm_op_type_aliases_and_canonical(self):
        mapping = {
            None: "NoOp",
            "NoOp": "NoOp",
            "noop": "NoOp",
            "no-op": "NoOp",
            "ALL2ALL": "ALL2ALL",
            "all_to_all": "ALL2ALL",
            "AllReduce": "AllReduce",
            "ALL-REDUCE": "AllReduce",
            "AllGather": "AllGather",
            "reduce_scatter": "ReduceScatter",
            "unexpected": "NoOp",
            "": "NoOp",
        }

        for raw, expected in mapping.items():
            self.assertEqual(
                self.parallel_state_ext._normalize_comm_op_type(raw),
                expected,
                msg=f"{raw} -> {expected}",
            )

    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=4)
    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_enforces_full_coverage(
        self, mock_is_initialized, mock_get_world
    ):
        spec = [[0, 1], [2, 3]]
        result = self.parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertEqual(result, spec)
        mock_is_initialized.assert_called_once()

    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=3)
    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_requires_all_ranks(self, *_):
        spec = [[0, 1]]
        with self.assertRaises(RuntimeError) as context:
            self.parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertIn("must cover all ranks", str(context.exception))

    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.get_world_size", return_value=4)
    @patch("omni_npu.v1.distributed.parallel_state_ext.dist.is_initialized", return_value=True)
    def test_tp_size_or_ranks_list_detects_duplicates(self, *_):
        spec = [[0, 1], [1, 2]]
        with self.assertRaises(RuntimeError) as context:
            self.parallel_state_ext._tp_size_or_ranks_to_group_ranks(spec, "layer")
        self.assertIn("duplicate ranks across groups", str(context.exception))

    def test_ensure_layer_parallel_initialized_uses_passed_backend(self):
        # Force re-init within this test.
        self.parallel_state_ext._LAYER_COMM_DICT = None
        self.parallel_state_ext._LAYER_PARALLEL_GLOBAL_CFG = None

        vllm_config = Mock()
        vllm_config.parallel_config = Mock(local_rank=0)
        layer_parallel_config = {
            "input_split": False,
            "self_attn.q_proj": {"tp_size_or_ranks": [[0]]},
        }

        with patch(
            "omni_npu.v1.distributed.parallel_state_ext._load_layer_parallel_config_from_vllm",
            return_value=(vllm_config, layer_parallel_config),
        ), patch(
            "omni_npu.v1.distributed.parallel_state_ext.dist.is_initialized",
            return_value=True,
        ), patch(
            "omni_npu.v1.distributed.parallel_state_ext._create_group_from_tp_size_or_ranks",
            return_value=Mock(),
        ) as mock_create_group:
            self.parallel_state_ext.ensure_layer_parallel_initialized(backend="hccl")
            mock_create_group.assert_called()
            self.assertEqual(mock_create_group.call_args.args[2], "hccl")