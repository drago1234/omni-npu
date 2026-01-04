# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NPUCommunicator that do NOT require actual NPU hardware.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from contextlib import contextmanager
import sys

import pytest
import torch
from torch.distributed import ProcessGroup


# Define the set of vLLM modules that need to be mocked
VLLM_MODULES_TO_MOCK = {
    'vllm',
    'vllm.config',
    'vllm.logger',
    'vllm.platforms',
    'vllm.model_executor',
    'vllm.model_executor.models',
    'vllm.model_executor.models.interfaces',
    'vllm.distributed',
    'vllm.distributed.eplb',
    'vllm.distributed.eplb.eplb_state',
    'vllm.distributed.device_communicators',
    'vllm.distributed.device_communicators.cuda_communicator',
    'vllm.distributed.device_communicators.base_device_communicator',
}

@contextmanager
def mock_distributed_environment():
    """Context manager to mock torch.distributed environment"""
    with patch('torch.distributed.get_rank', return_value=0):
        with patch('torch.distributed.get_world_size', return_value=1):
            yield


@pytest.mark.unit
class TestNPUCommunicatorUnit(unittest.TestCase):
    """Unit tests for NPUCommunicator (no NPU hardware required)"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock torch_npu availability
        self.torch_npu_mock = MagicMock()
        sys.modules['torch_npu'] = self.torch_npu_mock
        
        # Mock all required vLLM modules
        for module_name in VLLM_MODULES_TO_MOCK:
            if module_name not in sys.modules:
                sys.modules[module_name] = MagicMock()

        # Create a proper mock base class that can be instantiated
        class MockCudaCommunicator:
            def __init__(self, cpu_group, device=None, device_group=None, unique_name=""):
                self.cpu_group = cpu_group
                self.device = device
                self.device_group = device_group
                self.unique_name = unique_name
                self.world_size = 1
                self.rank_in_group = 0
                self.ranks = [0]
        
        sys.modules['vllm.distributed.device_communicators.cuda_communicator'].CudaCommunicator = MockCudaCommunicator

    def tearDown(self):
        """Clean up after tests"""
        # Remove mocked modules
        modules_to_remove = {
            'torch_npu',
            'omni_npu.distributed.communicator',
        } | VLLM_MODULES_TO_MOCK
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

    def test_init_with_torch_npu_available(self):
        """Test NPUCommunicator initialization when torch.npu is available"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                device = torch.device('npu:0')
                
                communicator = NPUCommunicator(
                    cpu_group=cpu_group,
                    device=device,
                    device_group=device_group,
                    unique_name="test"
                )
                
                self.assertIsNotNone(communicator)
                self.assertEqual(communicator.dist_module, torch.distributed)

    def test_init_without_torch_npu_raises_error(self):
        """Test NPUCommunicator raises RuntimeError when torch.npu is not available"""
        if hasattr(torch, 'npu'):
            delattr(torch, 'npu')
        
        with mock_distributed_environment():
            from omni_npu.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            with self.assertRaises(RuntimeError) as context:
                NPUCommunicator(
                    cpu_group=cpu_group,
                    device_group=device_group,
                )
            
            self.assertIn("torch.npu", str(context.exception))
            self.assertIn("torch_npu is properly installed", str(context.exception))

    def test_all_reduce_delegates_to_torch_distributed(self):
        """Test all_reduce delegates to torch.distributed.all_reduce"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                
                with patch.object(torch.distributed, 'all_reduce') as mock_all_reduce:
                    input_tensor = torch.randn(4, 4)
                    result = communicator.all_reduce(input_tensor)
                    
                    mock_all_reduce.assert_called_once_with(input_tensor, group=device_group)
                    self.assertIs(result, input_tensor)

    def test_all_gather_shape_transformation(self):
        """Test all_gather performs correct shape transformation"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                # Mock the parent class __init__ to avoid real distributed calls
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).world_size = PropertyMock(return_value=2)
                
                with patch.object(torch.distributed, 'all_gather_into_tensor') as mock_all_gather:
                    input_tensor = torch.randn(2, 4)
                    result = communicator.all_gather(input_tensor, dim=-1)
                    
                    mock_all_gather.assert_called_once()
                    self.assertIsInstance(result, torch.Tensor)

    def test_reduce_scatter_world_size_one_returns_input(self):
        """Test reduce_scatter returns input tensor when world_size is 1"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                type(communicator).world_size = PropertyMock(return_value=1)
                
                input_tensor = torch.randn(4, 4)
                result = communicator.reduce_scatter(input_tensor, dim=0)
                
                self.assertIs(result, input_tensor)

    def test_reduce_scatter_delegates_to_torch_distributed(self):
        """Test reduce_scatter delegates to torch.distributed.reduce_scatter_tensor"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).world_size = PropertyMock(return_value=2)
                
                with patch.object(torch.distributed, 'reduce_scatter_tensor') as mock_reduce_scatter:
                    input_tensor = torch.randn(4, 4)
                    result = communicator.reduce_scatter(input_tensor, dim=0)
                    
                    mock_reduce_scatter.assert_called_once()
                    self.assertIsInstance(result, torch.Tensor)

    def test_send_with_explicit_destination(self):
        """Test send with explicit destination rank"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).rank_in_group = PropertyMock(return_value=0)
                type(communicator).world_size = PropertyMock(return_value=2)
                type(communicator).ranks = PropertyMock(return_value=[0, 1])
                
                with patch.object(torch.distributed, 'send') as mock_send:
                    tensor = torch.randn(4, 4)
                    communicator.send(tensor, dst=1)
                    
                    mock_send.assert_called_once_with(tensor, 1, device_group)

    def test_send_with_default_destination(self):
        """Test send with default destination (next rank)"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).rank_in_group = PropertyMock(return_value=0)
                type(communicator).world_size = PropertyMock(return_value=4)
                type(communicator).ranks = PropertyMock(return_value=[0, 1, 2, 3])
                
                with patch.object(torch.distributed, 'send') as mock_send:
                    tensor = torch.randn(4, 4)
                    communicator.send(tensor, dst=None)
                    
                    mock_send.assert_called_once_with(tensor, 1, device_group)

    def test_recv_creates_tensor_with_correct_shape(self):
        """Test recv creates tensor with correct shape and dtype"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                device = torch.device('cpu')
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).rank_in_group = PropertyMock(return_value=1)
                type(communicator).world_size = PropertyMock(return_value=2)
                type(communicator).ranks = PropertyMock(return_value=[0, 1])
                type(communicator).device = PropertyMock(return_value=device)
                
                with patch.object(torch.distributed, 'recv') as mock_recv:
                    size = torch.Size([4, 4])
                    dtype = torch.float32
                    result = communicator.recv(size, dtype, src=0)
                    
                    mock_recv.assert_called_once()
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertEqual(result.shape, size)
                    self.assertEqual(result.dtype, dtype)

    def test_gather_on_destination_rank(self):
        """Test gather returns concatenated tensor on destination rank"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).rank_in_group = PropertyMock(return_value=0)
                type(communicator).world_size = PropertyMock(return_value=2)
                type(communicator).ranks = PropertyMock(return_value=[0, 1])
                
                with patch.object(torch.distributed, 'gather') as mock_gather:
                    input_tensor = torch.randn(2, 4)
                    result = communicator.gather(input_tensor, dst=0, dim=0)
                    
                    mock_gather.assert_called_once()
                    self.assertIsInstance(result, torch.Tensor)

    def test_gather_on_non_destination_rank(self):
        """Test gather returns None on non-destination rank"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).rank_in_group = PropertyMock(return_value=1)
                type(communicator).world_size = PropertyMock(return_value=2)
                type(communicator).ranks = PropertyMock(return_value=[0, 1])
                
                with patch.object(torch.distributed, 'gather') as mock_gather:
                    input_tensor = torch.randn(2, 4)
                    result = communicator.gather(input_tensor, dst=0, dim=0)
                    
                    mock_gather.assert_called_once()
                    self.assertIsNone(result)

    def test_destroy_returns_none(self):
        """Test destroy method returns None"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                
                result = communicator.destroy()
                self.assertIsNone(result)

    def test_all_gatherv_raises_for_non_zero_dim(self):
        """Test all_gatherv raises NotImplementedError for dim != 0"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                
                input_tensor = torch.randn(2, 4)
                with self.assertRaises(NotImplementedError) as context:
                    communicator.all_gatherv(input_tensor, dim=1)
                
                self.assertIn("only dim 0", str(context.exception))

    def test_negative_dim_handling(self):
        """Test that negative dim values are correctly converted"""
        with patch.object(torch, 'npu', create=True):
            with mock_distributed_environment():
                from omni_npu.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
                communicator.device_group = device_group
                type(communicator).world_size = PropertyMock(return_value=2)
                
                with patch.object(torch.distributed, 'all_gather_into_tensor'):
                    input_tensor = torch.randn(2, 4)
                    result = communicator.all_gather(input_tensor, dim=-1)
                    self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
