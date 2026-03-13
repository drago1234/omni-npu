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
    ACLGraphEntry,
    weak_ref_workspaces
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
            "metadata1": MagicMock(query_start_loc=[0, 1], seq_lens=torch.tensor([2]))
        }
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_reqs = 3
        batch_descriptor.num_tokens = 3
        default_aclgraph_wrapper.vllm_config.scheduler_config.max_num_seqs = 10

        with patch.object(default_aclgraph_wrapper, "concrete_aclgraph_entries", new_callable=dict) as mock_concrete_aclgraph_entries:
            # simulate the captured aclgraph
            mock_entry = MagicMock(spec=ACLGraphEntry)
            mock_aclgraph = MagicMock()
            mock_aclgraph.replay = MagicMock()

            mock_entry.aclgraph = mock_aclgraph
            mock_entry.output = torch.tensor([2.0])
            mock_concrete_aclgraph_entries[batch_descriptor] = mock_entry
            
            with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
                 patch("torch.npu.stream"), \
                 patch("torch.npu.graph_task_update_begin"), \
                 patch("torch.npu.graph_task_update_end"):
                # Mock get_graph_params to provide valid graph params
                mock_graph_params = MagicMock()
                mock_graph_params.attn_params = {3:[]}
                mock_graph_params.handles = {3:[]}
                mock_graph_params.events = {3:[]}
                mock_graph_params.workspaces = {3:None}
                mock_get_graph_params.return_value = mock_graph_params

                result = default_aclgraph_wrapper(torch.tensor([1.0]))

            # Verify runnable should not be called
            default_aclgraph_wrapper.runnable.assert_not_called()
            # Verify replay and update should be called
            mock_aclgraph.replay.assert_called_once()
            assert result == mock_entry.output
            assert batch_descriptor in default_aclgraph_wrapper.concrete_aclgraph_entries


    def test_call_with_aslkv_none(self, default_aclgraph_wrapper):
        """Test __call__ method raise error when attn_metadata is None."""
        default_aclgraph_wrapper._forward_context.attn_metadata = None
        batch_descriptor = default_aclgraph_wrapper._forward_context.batch_descriptor
        batch_descriptor.num_reqs = 3
        batch_descriptor.num_tokens = 3
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
        mock_attn_metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        mock_attn_metadata.query_start_loc = [0, 1]
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



def test_weak_ref_workspaces():
    """Test weak_ref_workspaces function"""
    params = GraphParams(
        {1: None, 2: [], 3: []},
        {1: None, 2: None, 3: None},
        {1: [], 2: [], 3: []},
        {1: [], 2: [], 3: []}
    )

    # Test with None params
    weak_ref_workspaces(None)

    # Test with params containing None workspaces
    weak_ref_workspaces(params)
    assert params.workspaces.get(1) is None

    # Test with params containing tensor workspaces
    workspace_tensor = torch.tensor([1.0, 2.0, 3.0])
    params.workspaces[2] = [workspace_tensor]

    weak_ref_workspaces(params)

    # Verify weak_ref was applied
    result_tensor = params.workspaces[2][0]
    assert torch.equal(result_tensor, workspace_tensor)


class TestACLGraphWrapperUpdateMethods:
    """Test class for ACLGraphWrapper update method"""

    @pytest.fixture
    def wrapper_with_update_stream(self):
        with patch("torch.npu.NPUGraph"), \
             patch("torch.npu.graph"), \
             patch("torch.npu.Stream") as mock_stream, \
             patch("omni_npu.compilation.acl_graph.get_forward_context") as mock_get_forward_context:

            mock_stream_instance = MagicMock()
            mock_stream.return_value = mock_stream_instance
            mock_stream_instance.__enter__ = lambda self: mock_stream_instance
            mock_stream_instance.__exit__ = lambda *args: None

            mock_context = MagicMock()
            mock_batch_descriptor = MagicMock()
            mock_batch_descriptor.num_reqs = 3
            mock_batch_descriptor.num_tokens = 4
            mock_context.batch_descriptor = mock_batch_descriptor
            mock_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            mock_get_forward_context.return_value = mock_context

            vllm_config = MagicMock(spec=VllmConfig)
            vllm_config.model_config.use_mla = False
            vllm_config.scheduler_config.max_num_seqs = 10

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(),
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                update_stream=mock_stream_instance
            )
            wrapper._forward_context = mock_context

            yield wrapper, mock_context, mock_stream_instance

    def test_update_fia_params_with_list_aslkv(self, wrapper_with_update_stream):
        """Test _update_fia_params with list aslkv"""
        wrapper, forward_context, _ = wrapper_with_update_stream

        # Setup metadata without decode attribute (GQA mode)
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [1, 2]
        metadata.seq_lens = [2, 3]

        forward_context.attn_metadata = {0: metadata}

        asl, aslkv = wrapper._update_fia_params(forward_context, 0)

        # Verify padding was applied
        # query_start_loc[1:] = [2], after padding with num_tokens=4 becomes [2, 4]
        assert asl == [2, 2, 4]
        assert aslkv == [2, 3, 0]

    def test_update_fia_params_with_tensor_aslkv(self, wrapper_with_update_stream):
        """Test _update_fia_params with tensor aslkv"""
        wrapper, forward_context, _ = wrapper_with_update_stream

        # Setup metadata with decode attribute and tensor
        asl_tensor = torch.tensor([1, 2])
        aslkv_tensor = torch.tensor([2, 3])

        class DecodeMetadata:
            query_cumlens = asl_tensor
            seq_lens = aslkv_tensor

        class Metadata:
            decode = DecodeMetadata()

        forward_context.attn_metadata = {0: Metadata()}

        asl, aslkv = wrapper._update_fia_params(forward_context, 0)

        # Verify tensors are returned as-is
        assert torch.equal(asl, asl_tensor)
        assert torch.equal(aslkv, aslkv_tensor)

    def test_update_fia_params_with_seq_sink_len(self, wrapper_with_update_stream):
        """Test _update_fia_params with seq_sink_len"""
        wrapper, forward_context, _ = wrapper_with_update_stream

        # Setup metadata with decode and seq_sink_len
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.query_cumlens = [1, 2]
        metadata.decode.seq_lens = [2, 3]
        metadata.decode.seq_sink_len = [3, 4]

        forward_context.attn_metadata = {0: metadata}

        asl, aslkv = wrapper._update_fia_params(forward_context, 0)

        # Verify seq_sink_len is used for kv lengths
        assert asl == [1, 2, 4]
        assert aslkv == [2, 3, 0]

    def test_update_fia_params_with_zero_aslkv_last_element(self, wrapper_with_update_stream):
        """Test _update_fia_params when last element of aslkv is zero"""
        wrapper, forward_context, _ = wrapper_with_update_stream

        # Setup metadata with zero in last position
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [1, 2]
        metadata.seq_lens = [2, 0]

        forward_context.attn_metadata = {0: metadata}

        asl, aslkv = wrapper._update_fia_params(forward_context, 0)

        # Verify extension is not applied when last element is zero
        # query_start_loc[1:] = [2], after padding becomes [2, 4]
        assert asl == [2, 2, 4]
        assert aslkv == [2, 0, 0]

    def test_update_fia_params_with_none_num_reqs(self, wrapper_with_update_stream):
        """Test _update_fia_params when num_reqs is None"""
        wrapper, forward_context, _ = wrapper_with_update_stream

        # Setup metadata
        asl_list = [1, 2]
        aslkv_list = [2, 3]

        class Metadata:
            query_start_loc = asl_list
            seq_lens = aslkv_list

        forward_context.attn_metadata = {0: Metadata()}
        forward_context.batch_descriptor.num_reqs = None
        forward_context.batch_descriptor.num_tokens = 4

        asl, aslkv = wrapper._update_fia_params(forward_context, 0)

        # Verify padding uses max_num_seqs when num_reqs is None
        # When num_reqs is None, padding_lens = min(10, 4) = 4
        assert len(asl) == 4  # padding length is min(10, 4) = 4
        assert len(aslkv) == 4
        assert asl[-1] == 4  # num_tokens

    def test_update_attn_fia_params_with_none_attn_metadata(self, wrapper_with_update_stream):
        """Test _update_attn_fia_params raises error when attn_metadata is None"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        forward_context.attn_metadata = None

        with pytest.raises(RuntimeError, match="attn_metadata is None"):
            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

    def test_update_attn_fia_params_with_empty_attn_metadata(self, wrapper_with_update_stream):
        """Test _update_attn_fia_params with empty attn_metadata"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        forward_context.attn_metadata = {}

        # Should not raise error, just return early
        wrapper._update_attn_fia_params(update_stream, forward_context, 1)

    def _setup_graph_params(self, op_name="npu_fused_infer_attention_score", is_mla=False):
        """Helper to setup common graph params"""
        mock_graph_params = MagicMock()
        mock_attn_param = {
            "op_name": op_name,
            "query": MagicMock(), "key": MagicMock(), "value": MagicMock(),
            "num_key_value_heads": 8, "input_layout": "layout1",
            "block_table": MagicMock(), "block_size": 16, "sparse_mode": 0,
            "atten_mask": MagicMock(), "attn_output": MagicMock(), "softmax_lse": MagicMock(),
        }

        if op_name == "npu_fused_infer_attention_score":
            mock_attn_param.update({"num_heads": 8, "scale": 1.0})
        else:
            mock_attn_param.update({"num_query_heads": 8, "softmax_scale": 1.0})

        if is_mla:
            mock_attn_param.update({"query_rope": MagicMock(), "key_rope": MagicMock(),
                                    "pre_tokens": 2147483647, "next_tokens": 2147483647,})
            if op_name == "npu_fused_infer_attention_score":
                mock_attn_param.update({"antiquant_mode": 0, "softmax_lse_flag": False})
            else:
                mock_attn_param.update({"sink_number": 0})

        mock_graph_params.attn_params = {1: [mock_attn_param]}
        mock_graph_params.handles = {1: [MagicMock()]}
        mock_graph_params.events = {1: [MagicMock()]}
        mock_graph_params.workspaces = {1: MagicMock()}
        return mock_graph_params

    def _setup_metadata(self, is_mla=False):
        """Helper to setup common metadata"""
        metadata = MagicMock()
        if is_mla:
            metadata.decode.query_cumlens = [1, 2]
            metadata.decode.seq_lens = [2, 3]
        else:
            metadata.query_cumlens = [1, 2]
            metadata.seq_lens = [2, 3]
        return metadata

    @patch("torch_npu.npu_fused_infer_attention_score.out")
    def test_update_attn_fia_params_fia_v1(self, mock_fia_out, wrapper_with_update_stream):
        """Test _update_attn_fia_params with FIA v1"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:

            mock_get_graph_params.return_value = self._setup_graph_params("npu_fused_infer_attention_score", is_mla=False)
            forward_context.attn_metadata = {0: self._setup_metadata(is_mla=False)}

            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

            mock_update_begin.assert_called_once()
            mock_fia_out.assert_called_once()
            mock_update_end.assert_called_once()
            mock_get_graph_params.return_value.events[1][0].record.assert_called_once()

    @patch("torch_npu.npu_fused_infer_attention_score_v2.out")
    def test_update_attn_fia_params_fia_v2(self, mock_fia_out, wrapper_with_update_stream):
        """Test _update_attn_fia_params with FIA v2"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:

            mock_get_graph_params.return_value = self._setup_graph_params("npu_fused_infer_attention_score_v2", is_mla=False)
            forward_context.attn_metadata = {0: self._setup_metadata(is_mla=False)}

            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

            mock_update_begin.assert_called_once()
            mock_fia_out.assert_called_once()
            mock_update_end.assert_called_once()
            mock_get_graph_params.return_value.events[1][0].record.assert_called_once()

    @patch("torch.ops.custom.npu_fused_infer_attention_sink.out")
    def test_update_attn_fia_params_sink(self, mock_sink_out, wrapper_with_update_stream):
        """Test _update_attn_fia_params with npu_fused_infer_attention_sink op"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        # Setup graph params for sink attention
        mock_graph_params = MagicMock()
        mock_attn_param = {
            "op_name": "npu_fused_infer_attention_sink",
            "query": MagicMock(),
            "key": MagicMock(),
            "value": MagicMock(),
            "num_query_heads": 8,
            "num_key_value_heads": 8,
            "input_layout": "layout1",
            "softmax_scale": 1.0,
            "sink_number": 4,
            "pre_tokens": 2147483647,
            "next_tokens": 2147483647,
            "attn_output": MagicMock(),
            "softmax_lse": MagicMock(),
            "sparse_mode": 0,
            "atten_mask": MagicMock(),
            "block_table": MagicMock(),
            "block_size": 16,
        }
        mock_graph_params.attn_params = {1: [mock_attn_param]}
        mock_graph_params.handles = {1: [MagicMock()]}
        mock_graph_params.events = {1: [MagicMock()]}
        mock_graph_params.workspaces = {1: MagicMock()}

        # Setup metadata
        metadata = MagicMock()
        metadata.query_cumlens = [1, 2]
        metadata.seq_lens = [2, 3]
        forward_context.attn_metadata = {0: metadata}

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:

            mock_get_graph_params.return_value = mock_graph_params

            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

            # Verify sink branch was called
            mock_update_begin.assert_called_once()
            mock_sink_out.assert_called_once()
            mock_update_end.assert_called_once()
            mock_get_graph_params.return_value.events[1][0].record.assert_called_once()

            # Verify sink-specific parameters were passed
            call_kwargs = mock_sink_out.call_args.kwargs
            assert "sink_number" in call_kwargs
            assert call_kwargs["sink_number"] == 4

    @patch("torch.npu.stream")
    @patch("omni_npu.compilation.acl_graph.get_graph_params")
    def test_update_attn_fia_params_unsupported_op_name(self, mock_get_graph_params, mock_npu_stream,
                                                         wrapper_with_update_stream):
        """Test _update_attn_fia_params with unsupported op_name raises RuntimeError"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        # Setup graph params with unsupported op name
        mock_graph_params = MagicMock()
        mock_attn_param = {
            "op_name": "unsupported_attention_op",
            "query": MagicMock(),
            "key": MagicMock(),
            "value": MagicMock(),
        }
        mock_graph_params.attn_params = {1: [mock_attn_param]}
        mock_graph_params.handles = {1: [MagicMock()]}
        mock_graph_params.events = {1: [MagicMock()]}
        mock_graph_params.workspaces = {1: MagicMock()}

        # Setup metadata
        metadata = MagicMock()
        metadata.query_cumlens = [1, 2]
        metadata.seq_lens = [2, 3]
        forward_context.attn_metadata = {0: metadata}

        mock_get_graph_params.return_value = mock_graph_params

        with pytest.raises(RuntimeError, match="Unsupported op name for attention"):
            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

    @patch("torch_npu.npu_fused_infer_attention_score.out")
    def test_update_mla_attn_params_fia_v1(self, mock_fia_out, wrapper_with_update_stream):
        """Test _update_mla_attn_params with FIA v1"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:

            mock_get_graph_params.return_value = self._setup_graph_params("npu_fused_infer_attention_score", is_mla=True)
            forward_context.attn_metadata = {0: self._setup_metadata(is_mla=True)}

            wrapper._update_mla_attn_params(update_stream, forward_context, 1)

            mock_update_begin.assert_called_once()
            mock_fia_out.assert_called_once()
            mock_update_end.assert_called_once()
            mock_get_graph_params.return_value.events[1][0].record.assert_called_once()

    @patch("torch.ops.custom.npu_fused_infer_attention_sink.out")
    def test_update_mla_attn_params_fia_sink(self, mock_fia_out, wrapper_with_update_stream):
        """Test _update_mla_attn_params with FIA sink"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as mock_get_graph_params, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as mock_update_begin, \
             patch("torch.npu.graph_task_update_end") as mock_update_end:

            mock_get_graph_params.return_value = self._setup_graph_params("npu_fused_infer_attention_sink", is_mla=True)
            forward_context.attn_metadata = {0: self._setup_metadata(is_mla=True)}

            wrapper._update_mla_attn_params(update_stream, forward_context, 1)

            mock_update_begin.assert_called_once()
            mock_fia_out.assert_called_once()
            mock_update_end.assert_called_once()
            mock_get_graph_params.return_value.events[1][0].record.assert_called_once()

    def test_wrapper_initialization_with_update_stream(self):
        """Test ACLGraphWrapper initialization with update_stream parameter"""
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        runtime_mode = CUDAGraphMode.PIECEWISE
        graph_pool = MagicMock()
        update_stream = MagicMock(spec=torch.npu.Stream)

        wrapper = ACLGraphWrapper(
            runnable, vllm_config, runtime_mode, graph_pool,
            cudagraph_options=None, update_stream=update_stream
        )

        assert wrapper.update_stream == update_stream

    @patch("torch.npu.stream")
    @patch("omni_npu.compilation.acl_graph.get_graph_params")
    @patch("torch.npu.graph_task_update_begin")
    @patch("torch.npu.graph_task_update_end")
    def test_update_attn_fia_params_with_pre_tokens(self, mock_update_end, mock_update_begin,
                                                    mock_get_graph_params, mock_npu_stream, wrapper_with_update_stream):
        """Test _update_attn_fia_params with pre_tokens and next_tokens"""
        wrapper, forward_context, update_stream = wrapper_with_update_stream

        # Setup graph params with v2 FIA including pre_tokens and next_tokens
        mock_graph_params = MagicMock()
        mock_attn_param = {
            "op_name": "npu_fused_infer_attention_score_v2",
            "query": MagicMock(),
            "key": MagicMock(),
            "value": MagicMock(),
            "num_query_heads": 8,
            "num_key_value_heads": 8,
            "input_layout": "layout1",
            "softmax_scale": 1.0,
            "block_table": MagicMock(),
            "block_size": 16,
            "sparse_mode": 0,
            "atten_mask": MagicMock(),
            "attn_output": MagicMock(),
            "softmax_lse": MagicMock(),
            "pre_tokens": 10,
            "next_tokens": 20,
            "learnable_sink": MagicMock()
        }
        mock_graph_params.attn_params = {1: [mock_attn_param]}
        mock_graph_params.handles = {1: [MagicMock()]}
        mock_graph_params.events = {1: [MagicMock()]}
        mock_graph_params.workspaces = {1: MagicMock()}

        mock_get_graph_params.return_value = mock_graph_params

        # Setup metadata
        metadata = MagicMock()
        metadata.query_cumlens = [1, 2]
        metadata.seq_lens = [2, 3]
        forward_context.attn_metadata = {0: metadata}

        # Mock the FIA function to avoid actual execution
        with patch("torch_npu.npu_fused_infer_attention_score_v2.out") as mock_fia_out:
            wrapper._update_attn_fia_params(update_stream, forward_context, 1)

            # Verify update was called with pre_tokens and next_tokens
            call_args = mock_get_graph_params.return_value.attn_params[1][0]
            assert call_args.get("pre_tokens") == 10
            assert call_args.get("next_tokens") == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])