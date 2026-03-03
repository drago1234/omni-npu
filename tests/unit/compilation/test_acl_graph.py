import pytest
from unittest.mock import patch, MagicMock

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig

from omni_npu.compilation.acl_graph import(
    weak_ref_tensor,
    weak_ref_tensors,
    GraphParams,
    set_graph_params,
    update_graph_params_workspaces,
    get_graph_params,
    ACLGraphWrapper,
    ACLGraphEntry
)


def test_weak_ref_tensor():
    """Test whether the weak reference shares data with the original tensor"""
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = weak_ref_tensor(tensor)
    assert torch.equal(result, tensor)


def test_weak_ref_tensors():
    """Test the handling of tuple and list tensor by weak_ref_tensors"""
    tensor_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    tensor_tuple = (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0]))
    tensor_dict = {"key": torch.tensor([1.0, 2.0])}

    weak_ref_tensor_list = weak_ref_tensors(tensor_list)
    assert all(torch.equal(wt, t) for wt, t in zip(weak_ref_tensor_list, tensor_list))

    weak_ref_tensor_tuple = weak_ref_tensors(tensor_tuple)
    assert all(torch.equal(wt, t) for wt, t in zip(weak_ref_tensor_tuple, tensor_tuple))

    with pytest.raises(ValueError, match="Invalid type for tensors"):
        weak_ref_tensors(tensor_dict)


def test_weak_ref_tensor_with_ascend_support():
    """Test torch.ops._C_ascend module contains a property named 'weak_ref_tensor'"""
    tensor = torch.tensor([1.0, 2.0, 3.0])

    # Simulate the torch.ops._C_ascend module and add the weak_ref_tensor attribute
    mock_weak_ref_tensor = MagicMock(return_value=tensor)
    mock_ascend = MagicMock()
    mock_ascend.weak_ref_tensor = mock_weak_ref_tensor

    with patch("torch.ops._C_ascend", new=mock_ascend):
        result = weak_ref_tensor(tensor)

        mock_weak_ref_tensor.assert_called_once_with(tensor)
        assert torch.equal(result, tensor)


def test_set_get_graph_params():
    """Test set_graph_params and get_graph_params function"""
    aclgraph_capture_sizes = {1, 2, 3}
    set_graph_params(aclgraph_capture_sizes)

    with pytest.raises(ValueError, match="Graph parameters have already been set!"):
        set_graph_params(aclgraph_capture_sizes)

    graph_params = get_graph_params()

    assert isinstance(graph_params, GraphParams)
    assert set(graph_params.events.keys()) == aclgraph_capture_sizes
    assert set(graph_params.workspaces.keys()) == aclgraph_capture_sizes
    assert set(graph_params.handles.keys()) == aclgraph_capture_sizes
    assert set(graph_params.attn_params.keys()) == aclgraph_capture_sizes


def test_update_graph_params_workspaces():
    """Test update_graph_params_workspaces function"""
    if get_graph_params() is None:
        aclgraph_capture_sizes = {1, 2, 3}
        set_graph_params(aclgraph_capture_sizes)
    
    workspace = torch.tensor([1.0, 2.0, 3.0])
    update_graph_params_workspaces(1, workspace)

    graph_params = get_graph_params()
    assert torch.equal(graph_params.workspaces[1], workspace)


class TestACLGraphWrapper:
    """Test class for ACLGraphWrapper"""

    @pytest.fixture
    def default_npu_graph(self):
        with patch("torch.npu.NPUGraph") as mock_npu_graph, \
             patch("torch.npu.graph") as mock_npu_graph_ctx, \
             patch("torch.npu.Stream") as mock_npu_stream:
            mock_graph_instance = MagicMock()
            mock_graph_instance.replay = MagicMock()
            mock_graph_instance.update = MagicMock()
            mock_npu_graph.return_value = mock_graph_instance

            # mock torch.npu.graph context manager to avoid executing the actual logic within the with statement
            mock_npu_graph_ctx.return_value.__enter__ = lambda self: None
            mock_npu_graph_ctx.return_value.__exit__ = lambda *args: None

            mock_npu_stream.return_value = MagicMock()

            yield mock_graph_instance


    @pytest.fixture
    def default_aclgraph_wrapper(self, default_npu_graph):
        with patch("omni_npu.compilation.acl_graph.get_forward_context") as mock_get_forward_context:
            mock_context = MagicMock()
            mock_batch_descriptor = MagicMock()
            mock_batch_descriptor.num_reqs = None
            mock_batch_descriptor.num_tokens = None
            mock_context.batch_descriptor = mock_batch_descriptor
            mock_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            mock_context.attn_metadata = None
            mock_get_forward_context.return_value = mock_context

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(),
                vllm_config=MagicMock(spec=VllmConfig),
                runtime_mode=CUDAGraphMode.PIECEWISE,
                graph_pool=MagicMock()
            )
            wrapper._forward_context = mock_context
            wrapper.concrete_aclgraph_entries = {}
            wrapper.aclgraph_options = MagicMock(gc_disable=False, debug_log_enable=False)
            wrapper.default_npu_graph = default_npu_graph
            yield wrapper


    def test_acl_graph_wrapper_init_normal(self):
        """Test ACLGraphWrapper initialization"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()
        cudagraph_options = MagicMock()

        wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, graph_pool, cudagraph_options)

        assert wrapper.runnable == runnable
        assert wrapper.vllm_config == vllm_config
        assert wrapper.graph_pool == graph_pool
        assert wrapper.runtime_mode == runtime_mode
        assert wrapper.aclgraph_options == cudagraph_options
        assert wrapper.first_run_finished is False
        assert isinstance(wrapper.concrete_aclgraph_entries, dict)


    def test_acl_graph_wrapper_init_with_none_graph_pool(self):
        """Test ACLGraphWrapper initialization when graph_pool is None"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        cudagraph_options = MagicMock()

        mock_graph_pool = MagicMock()
        with patch("omni_npu.compilation.acl_graph.current_platform.get_global_graph_pool", return_value=mock_graph_pool) as mock_get_global_graph_pool:
            wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, None, cudagraph_options)

            assert wrapper.graph_pool == mock_graph_pool
            mock_get_global_graph_pool.assert_called_once()


    def test_acl_graph_wrapper_init_with_none_cudagraph_options(self):
        """Test ACLGraphWrapper initialization when cudagraph_options is None"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()

        with patch("omni_npu.compilation.acl_graph.CUDAGraphOptions", return_value=MagicMock(spec=CUDAGraphOptions)) as mock_cudagraph_options:
            wrapper = ACLGraphWrapper(runnable, vllm_config, runtime_mode, graph_pool, None)

            # Verify whether the type of aclgraph_options is CUDAGraphOptions
            assert isinstance(wrapper.aclgraph_options, CUDAGraphOptions)
            mock_cudagraph_options.assert_called_once()

    
    def test_acl_graph_wrapper_getattr(self):
        """Test ACLGraphWrapper __get_attr__ method"""
        # Allow runnable to have the key attribute
        runnable = MagicMock(spec_set=["key"])
        runnable.key = "value"
        
        wrapper = ACLGraphWrapper(runnable, MagicMock(), CUDAGraphMode.PIECEWISE)

        assert wrapper.key == "value"
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attr


    def test_acl_graph_wrapper_unwrap(self):
        """Test ACLGraphWrapper unwrap method"""
        runnable = MagicMock()
        wrapper = ACLGraphWrapper(runnable, MagicMock(), CUDAGraphMode.PIECEWISE)
        assert wrapper.unwrap() == runnable


    def test_pad_list(self):
        """Test ACLGraphWrapper _pad_list function"""
        wrapper = ACLGraphWrapper(MagicMock(), MagicMock(), CUDAGraphMode.PIECEWISE)
        assert wrapper._pad_list([], 3) == []
        assert wrapper._pad_list([1, 2, 3], 3) == [1, 2, 3]
        assert wrapper._pad_list([1, 2, 3], 5) == [1, 2, 3, 3, 3]
        assert wrapper._pad_list([1, 2, 3], 2) == [1, 2]
    

    def test_call_non_aclgraph_runtime_mode(self, default_aclgraph_wrapper):
        """Test the direct invocation of the runnable in the CUDAGraphMode.NONE mode."""
        default_aclgraph_wrapper.runtime_mode = CUDAGraphMode.NONE
        default_aclgraph_wrapper._forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        
        result = default_aclgraph_wrapper(torch.tensor([1.0]))

        # Verify direct invocation of the runnable method
        default_aclgraph_wrapper.runnable.assert_called_once()
        # Verify the returned value is the return value of the runnable.
        assert result == default_aclgraph_wrapper.runnable.return_value
        # Verify not create a new entry for batch descriptor
        assert not default_aclgraph_wrapper.concrete_aclgraph_entries


    def test_call_with_non_none_aclgraph_runtime_mode(self, default_aclgraph_wrapper):
        """Test the aclgraph replay behavior in a non-NONE aclgraph_runtime_mode."""
        default_aclgraph_wrapper.runtime_mode = CUDAGraphMode.PIECEWISE
        default_aclgraph_wrapper._forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
        
        default_aclgraph_wrapper._forward_context.attn_metadata = {
            "metadata1": MagicMock(query_cumlens=torch.tensor([1]), seq_lens=torch.tensor([2]))
        }
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_reqs = 3
        batch_descriptor.num_tokens = 4
        default_aclgraph_wrapper.vllm_config.scheduler_config.max_num_seqs = 10

        with patch.object(default_aclgraph_wrapper, "concrete_aclgraph_entries", new_callable=dict) as mock_concrete_aclgraph_entries:
            # simulate the captured aclgraph
            mock_entry = MagicMock(spec=ACLGraphEntry)
            mock_aclgraph = MagicMock()
            mock_aclgraph.replay = MagicMock()
            mock_aclgraph.update = MagicMock()

            mock_entry.aclgraph = mock_aclgraph
            mock_entry.output = torch.tensor([2.0])
            mock_concrete_aclgraph_entries[batch_descriptor] = mock_entry

            result = default_aclgraph_wrapper(torch.tensor([1.0]))

            # Verify runnable should not be called
            default_aclgraph_wrapper.runnable.assert_not_called()
            # Verify replay and update should be called
            mock_aclgraph.replay.assert_called_once()
            mock_aclgraph.update.assert_called_once()
            assert result == mock_entry.output
            assert batch_descriptor in default_aclgraph_wrapper.concrete_aclgraph_entries


    def test_call_with_aslkv_none(self, default_aclgraph_wrapper):
        """Test __call__ method raise error when attn_metadata is None."""
        default_aclgraph_wrapper._forward_context.attn_metadata = None
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_reqs = 3
        batch_descriptor.num_tokens = 4
        with patch.object(default_aclgraph_wrapper, "concrete_aclgraph_entries", new_callable=dict) as mock_concrete_aclgraph_entries:
            mock_entry = MagicMock(spec=ACLGraphEntry)
            mock_aclgraph = MagicMock()
            mock_aclgraph.replay = MagicMock()
            mock_concrete_aclgraph_entries[batch_descriptor] = mock_entry

            with pytest.raises(RuntimeError):
                default_aclgraph_wrapper(torch.tensor([1.0]))
            
            mock_entry.aclgraph.replay.assert_called_once()


    def test_call_attn_metadata_gqa_mode(self, default_aclgraph_wrapper):
        """Test __call__ method executes normally under the GQA mode."""
        mock_attn_metadata = MagicMock(spec_set=["query_cumlens", "seq_lens"])
        mock_attn_metadata.query_cumlens = torch.tensor([1])
        mock_attn_metadata.seq_lens = torch.tensor([2])
        default_aclgraph_wrapper._forward_context.attn_metadata = {
            "metadata1": mock_attn_metadata
        }
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        result = default_aclgraph_wrapper(torch.tensor([1.0]))

        default_aclgraph_wrapper.runnable.assert_called_once()
        assert torch.equal(result, torch.tensor([1.0]))


    def test_call_attn_metadata_mla_mode(self, default_aclgraph_wrapper):
        """Test __call__ method executes normally under the MLA mode."""
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.decode = MagicMock()
        mock_attn_metadata.decode.query_cumlens = torch.tensor([1])
        mock_attn_metadata.decode.seq_lens = torch.tensor([2])
        mock_attn_metadata.decode.seq_sink_len = torch.tensor([3])
        default_aclgraph_wrapper._forward_context.attn_metadata = {
            "metadata1": mock_attn_metadata
        }
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        result = default_aclgraph_wrapper(torch.tensor([1.0]))

        default_aclgraph_wrapper.runnable.assert_called_once()
        assert torch.equal(result, torch.tensor([1.0]))


    def test_call_with_input_address_mismatch(self, default_aclgraph_wrapper):
        """Test the behavior when input address do not match in debugging mode."""
        default_aclgraph_wrapper.is_debugging_mode = True
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        test_tensor = torch.tensor([1.0])
        new_input_address = [test_tensor.data_ptr()]
        old_input_address = [new_input_address[0] + 1]

        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        default_aclgraph_wrapper.runnable = MagicMock(return_value=torch.tensor([1.0]))

        with patch.object(default_aclgraph_wrapper, "concrete_aclgraph_entries", new_callable=dict) as mock_concrete_aclgraph_entries:
            mock_entry = MagicMock(spec=ACLGraphEntry)
            mock_entry.input_addresses = old_input_address
            mock_concrete_aclgraph_entries[batch_descriptor] = mock_entry

            with pytest.raises(AssertionError):
                default_aclgraph_wrapper(test_tensor)

            mock_entry.aclgraph.replay.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])