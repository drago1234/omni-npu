# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omni_npu.compilation.acl_graph module."""

import pytest
from unittest.mock import patch, MagicMock

import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig

import omni_npu.compilation.acl_graph as acl_graph_module
from omni_npu.compilation.acl_graph import (
    weak_ref_tensor,
    weak_ref_tensors,
    OpDescriptor,
    GraphTaskEntry,
    GraphParams,
    ACLGraphEntry,
    set_graph_params,
    get_graph_params,
    ensure_weak_ref_graph_params,
    ACLGraphWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_graph_params():
    """Reset the module-level _graph_params before each test."""
    acl_graph_module._graph_params = None
    yield
    acl_graph_module._graph_params = None


# ---------------------------------------------------------------------------
# weak_ref_tensor / weak_ref_tensors
# ---------------------------------------------------------------------------

class TestWeakRef:

    def test_weak_ref_tensor_passthrough(self):
        """Without _C_ascend.weak_ref_tensor, returns the same tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = weak_ref_tensor(tensor)
        assert torch.equal(result, tensor)

    def test_weak_ref_tensor_with_ascend_op(self):
        """When _C_ascend has weak_ref_tensor, delegates to it."""
        tensor = torch.tensor([1.0, 2.0])
        mock_op = MagicMock(return_value=tensor)
        mock_ascend = MagicMock()
        mock_ascend.weak_ref_tensor = mock_op

        with patch("torch.ops._C_ascend", new=mock_ascend):
            result = weak_ref_tensor(tensor)

        mock_op.assert_called_once_with(tensor)
        assert torch.equal(result, tensor)

    def test_weak_ref_tensors_single(self):
        tensor = torch.tensor([1.0])
        result = weak_ref_tensors(tensor)
        assert torch.equal(result, tensor)

    def test_weak_ref_tensors_list(self):
        tensors = [torch.tensor([1.0]), torch.tensor([2.0])]
        result = weak_ref_tensors(tensors)
        assert isinstance(result, list)
        assert all(torch.equal(r, t) for r, t in zip(result, tensors))

    def test_weak_ref_tensors_tuple(self):
        tensors = (torch.tensor([1.0]), torch.tensor([2.0]))
        result = weak_ref_tensors(tensors)
        assert isinstance(result, tuple)
        assert all(torch.equal(r, t) for r, t in zip(result, tensors))

    def test_weak_ref_tensors_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid type"):
            weak_ref_tensors({"key": torch.tensor([1.0])})


# ---------------------------------------------------------------------------
# GraphParams / set_graph_params / get_graph_params
# ---------------------------------------------------------------------------

class TestGraphParams:

    def test_set_and_get(self):
        sizes = {1, 4, 8}
        set_graph_params(sizes)

        params = get_graph_params()
        assert isinstance(params, GraphParams)
        assert set(params.task_entries.keys()) == sizes
        assert set(params.workspaces.keys()) == sizes
        # each size starts with an empty dict
        for size in sizes:
            assert params.task_entries[size] == {}
            assert params.workspaces[size] == {}

    def test_set_twice_raises(self):
        set_graph_params({1})
        with pytest.raises(ValueError, match="already been set"):
            set_graph_params({2})


# ---------------------------------------------------------------------------
# ensure_weak_ref_graph_params
# ---------------------------------------------------------------------------

class TestEnsureWeakRefGraphParams:

    def test_weak_refs_kwargs_and_out_tensors(self):
        """Verifies weak_ref_keys are weak-reffed, others untouched."""
        mock_workspace_fn = MagicMock()
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=mock_workspace_fn,
            weak_ref_keys=("query",),
        )
        query_tensor = torch.randn(2)
        entry = GraphTaskEntry(
            op_desc=op_desc,
            captured_kwargs={"query": query_tensor, "scale": 1.0},
            out_tensors=[torch.randn(2)],
            handle=MagicMock(),
            event=MagicMock(),
        )
        params = GraphParams(
            task_entries={4: {"layer0": entry}},
            workspaces={4: {mock_workspace_fn: torch.tensor([1.0])}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=params):
            ensure_weak_ref_graph_params()

        # scale (not in weak_ref_keys) should be unchanged
        assert entry.captured_kwargs["scale"] == 1.0

    def test_weak_refs_workspaces(self):
        """Workspace tensors are processed by weak_ref_tensors."""
        mock_fn = MagicMock()
        workspace = torch.tensor([1.0, 2.0])
        params = GraphParams(
            task_entries={4: {}},
            workspaces={4: {mock_fn: workspace}},
        )

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=params):
            ensure_weak_ref_graph_params()

        # workspace value should still be a tensor (weak_ref_tensor is
        # a pass-through without _C_ascend)
        assert isinstance(params.workspaces[4][mock_fn], torch.Tensor)


# ---------------------------------------------------------------------------
# ACLGraphWrapper
# ---------------------------------------------------------------------------

class TestACLGraphWrapperInit:

    def test_stores_attributes(self):
        runnable = MagicMock()
        vllm_config = MagicMock(spec=VllmConfig)
        pool = MagicMock()
        opts = MagicMock()

        wrapper = ACLGraphWrapper(
            runnable, vllm_config, CUDAGraphMode.PIECEWISE, pool, opts)

        assert wrapper.runnable is runnable
        assert wrapper.vllm_config is vllm_config
        assert wrapper.graph_pool is pool
        assert wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
        assert wrapper.aclgraph_options is opts
        assert wrapper.concrete_aclgraph_entries == {}

    def test_none_pool_uses_global(self):
        mock_pool = MagicMock()
        with patch("omni_npu.compilation.acl_graph.current_platform"
                    ".get_global_graph_pool", return_value=mock_pool):
            wrapper = ACLGraphWrapper(
                MagicMock(), MagicMock(spec=VllmConfig),
                CUDAGraphMode.PIECEWISE, graph_pool=None)

        assert wrapper.graph_pool is mock_pool

    def test_none_options_uses_default(self):
        with patch("omni_npu.compilation.acl_graph.CUDAGraphOptions",
                    return_value=MagicMock(spec=CUDAGraphOptions)) as mock_cls:
            wrapper = ACLGraphWrapper(
                MagicMock(), MagicMock(spec=VllmConfig),
                CUDAGraphMode.PIECEWISE, MagicMock(), cudagraph_options=None)

        mock_cls.assert_called_once()
        assert isinstance(wrapper.aclgraph_options, CUDAGraphOptions)


class TestACLGraphWrapperAttr:

    def test_getattr_delegates(self):
        runnable = MagicMock(spec_set=["some_attr"])
        runnable.some_attr = "value"
        wrapper = ACLGraphWrapper(
            runnable, MagicMock(), CUDAGraphMode.PIECEWISE)
        assert wrapper.some_attr == "value"

    def test_getattr_missing_raises(self):
        runnable = MagicMock(spec_set=["some_attr"])
        wrapper = ACLGraphWrapper(
            runnable, MagicMock(), CUDAGraphMode.PIECEWISE)
        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent

    def test_unwrap(self):
        runnable = MagicMock()
        wrapper = ACLGraphWrapper(
            runnable, MagicMock(), CUDAGraphMode.PIECEWISE)
        assert wrapper.unwrap() is runnable


class TestACLGraphWrapperCall:

    @pytest.fixture
    def wrapper_and_context(self):
        """Create a wrapper with a mocked forward context."""
        with patch("torch.npu.NPUGraph") as mock_graph_cls, \
             patch("torch.npu.graph") as mock_graph_ctx, \
             patch("omni_npu.compilation.acl_graph.get_forward_context") \
                as mock_get_ctx:

            mock_graph_instance = MagicMock()
            mock_graph_cls.return_value = mock_graph_instance
            mock_graph_ctx.return_value.__enter__ = lambda self: None
            mock_graph_ctx.return_value.__exit__ = lambda *a: None

            ctx = MagicMock()
            ctx.batch_descriptor = MagicMock()
            ctx.batch_descriptor.num_tokens = 4
            ctx.batch_descriptor.num_reqs = 2
            ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            ctx.attn_metadata = {}
            mock_get_ctx.return_value = ctx

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(return_value=torch.tensor([1.0])),
                vllm_config=MagicMock(spec=VllmConfig),
                runtime_mode=CUDAGraphMode.PIECEWISE,
                graph_pool=MagicMock(),
            )
            wrapper.aclgraph_options = MagicMock(
                gc_disable=False, debug_log_enable=False,
                weak_ref_output=False)

            yield wrapper, ctx, mock_graph_instance

    def test_none_mode_runs_directly(self, wrapper_and_context):
        wrapper, ctx, _ = wrapper_and_context
        ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE

        result = wrapper(torch.tensor([1.0]))

        wrapper.runnable.assert_called_once()
        assert result == wrapper.runnable.return_value
        assert not wrapper.concrete_aclgraph_entries

    def test_mismatched_mode_runs_directly(self, wrapper_and_context):
        wrapper, ctx, _ = wrapper_and_context
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL

        result = wrapper(torch.tensor([1.0]))

        wrapper.runnable.assert_called_once()
        assert not wrapper.concrete_aclgraph_entries

    def test_first_time_captures_graph(self, wrapper_and_context):
        wrapper, ctx, mock_graph = wrapper_and_context

        with patch("omni_npu.compilation.acl_graph."
                    "validate_cudagraph_capturing_enabled"), \
             patch("omni_npu.compilation.acl_graph."
                    "ensure_weak_ref_graph_params"):
            result = wrapper(torch.tensor([1.0]))

        wrapper.runnable.assert_called_once()
        bd = ctx.batch_descriptor
        assert bd in wrapper.concrete_aclgraph_entries
        entry = wrapper.concrete_aclgraph_entries[bd]
        assert entry.aclgraph is mock_graph

    def test_replay_returns_cached_output(self, wrapper_and_context):
        wrapper, ctx, _ = wrapper_and_context
        wrapper.is_debugging_mode = False
        bd = ctx.batch_descriptor

        mock_aclgraph = MagicMock()
        cached_output = torch.tensor([42.0])
        entry = ACLGraphEntry(batch_descriptor=bd)
        entry.aclgraph = mock_aclgraph
        entry.output = cached_output
        wrapper.concrete_aclgraph_entries[bd] = entry

        # task_entries and workspaces must be non-empty to pass the
        # emptiness check in _update_graph_tasks; the layer name won't
        # match attn_layer_names (empty by default) so no update runs.
        dummy_ws_fn = MagicMock()
        with patch("omni_npu.compilation.acl_graph.get_graph_params") \
                as mock_gp, \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin"), \
             patch("torch.npu.graph_task_update_end"):
            gp = MagicMock()
            gp.task_entries = {4: {"_dummy": MagicMock()}}
            gp.workspaces = {4: {dummy_ws_fn: MagicMock()}}
            mock_gp.return_value = gp

            result = wrapper(torch.tensor([1.0]))

        wrapper.runnable.assert_not_called()
        mock_aclgraph.replay.assert_called_once()
        assert torch.equal(result, cached_output)

    def test_debug_address_mismatch_raises(self, wrapper_and_context):
        wrapper, ctx, _ = wrapper_and_context
        wrapper.is_debugging_mode = True
        bd = ctx.batch_descriptor

        test_tensor = torch.tensor([1.0])
        entry = ACLGraphEntry(batch_descriptor=bd)
        entry.aclgraph = MagicMock()
        entry.input_addresses = [test_tensor.data_ptr() + 999]
        wrapper.concrete_aclgraph_entries[bd] = entry

        with pytest.raises(AssertionError, match="Input addresses"):
            wrapper(test_tensor)


# ---------------------------------------------------------------------------
# ACLGraphWrapper._update_graph_tasks
# ---------------------------------------------------------------------------

class TestUpdateGraphTasks:

    @pytest.fixture
    def wrapper_and_ctx(self):
        """Create a wrapper with update_stream and a forward context."""
        with patch("torch.npu.NPUGraph"), \
             patch("torch.npu.graph"), \
             patch("omni_npu.compilation.acl_graph.get_forward_context") \
                as mock_get_ctx:

            ctx = MagicMock()
            ctx.batch_descriptor = MagicMock()
            ctx.batch_descriptor.num_tokens = 4
            ctx.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            mock_get_ctx.return_value = ctx

            update_stream = MagicMock()

            wrapper = ACLGraphWrapper(
                runnable=MagicMock(),
                vllm_config=MagicMock(spec=VllmConfig),
                runtime_mode=CUDAGraphMode.PIECEWISE,
                graph_pool=MagicMock(),
                update_stream=update_stream,
                attn_layer_names=["layer0"],
            )

            yield wrapper, ctx, update_stream

    def _make_entry(self, compute_fn=None, op_out_fn=None):
        op_desc = OpDescriptor(
            op_out_fn=op_out_fn or MagicMock(),
            workspace_fn=MagicMock(),
            weak_ref_keys=("query",),
            compute_dynamic_kwargs=compute_fn,
        )
        return GraphTaskEntry(
            op_desc=op_desc,
            captured_kwargs={"query": MagicMock(), "scale": 1.0},
            out_tensors=[MagicMock()],
            handle=MagicMock(),
            event=MagicMock(),
        )

    def _make_graph_params(self, entry, runtime_shape=4):
        gp = MagicMock()
        gp.task_entries = {runtime_shape: {"layer0": entry}}
        gp.workspaces = {
            runtime_shape: {entry.op_desc.workspace_fn: MagicMock()}}
        return gp

    def test_none_metadata_raises(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = None

        with pytest.raises(RuntimeError, match="attn_metadata"):
            wrapper._update_graph_tasks(stream, ctx)

    def test_empty_entries_raises(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {}

        with patch("omni_npu.compilation.acl_graph.get_graph_params") as m:
            gp = MagicMock()
            gp.task_entries = {4: {}}
            gp.workspaces = {4: {}}
            m.return_value = gp

            with pytest.raises(RuntimeError, match="runtime_shape"):
                wrapper._update_graph_tasks(stream, ctx)

    def test_calls_compute_fn_and_issues_op(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {"layer0": MagicMock()}

        mock_op = MagicMock()
        dynamic = {"scale": 2.0}
        mock_compute = MagicMock(return_value=dynamic)
        entry = self._make_entry(compute_fn=mock_compute, op_out_fn=mock_op)
        gp = self._make_graph_params(entry)

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=gp), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as begin, \
             patch("torch.npu.graph_task_update_end") as end:

            wrapper._update_graph_tasks(stream, ctx)

        mock_compute.assert_called_once_with(
            ctx, "layer0", wrapper.vllm_config)
        begin.assert_called_once_with(stream, entry.handle)
        mock_op.assert_called_once()
        # verify dynamic kwargs were merged
        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["scale"] == 2.0
        end.assert_called_once_with(stream)
        entry.event.record.assert_called_once_with(stream)

    def test_skips_none_compute_fn(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {"layer0": MagicMock()}

        entry = self._make_entry(compute_fn=None)
        gp = self._make_graph_params(entry)

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=gp), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as begin, \
             patch("torch.npu.graph_task_update_end"):

            wrapper._update_graph_tasks(stream, ctx)

        begin.assert_not_called()
        entry.op_desc.op_out_fn.assert_not_called()
        entry.event.record.assert_called_once_with(stream)

    def test_skips_none_dynamic_result(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {"layer0": MagicMock()}

        mock_compute = MagicMock(return_value=None)
        entry = self._make_entry(compute_fn=mock_compute)
        gp = self._make_graph_params(entry)

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=gp), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as begin, \
             patch("torch.npu.graph_task_update_end"):

            wrapper._update_graph_tasks(stream, ctx)

        mock_compute.assert_called_once()
        begin.assert_not_called()
        entry.event.record.assert_called_once_with(stream)

    def test_skips_non_attn_layer(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {"other_layer": MagicMock()}
        wrapper.attn_layer_names = ["layer0"]

        mock_compute = MagicMock()
        entry = self._make_entry(compute_fn=mock_compute)
        gp = MagicMock()
        gp.task_entries = {4: {"other_layer": entry}}
        gp.workspaces = {4: {entry.op_desc.workspace_fn: MagicMock()}}

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=gp), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin") as begin, \
             patch("torch.npu.graph_task_update_end"):

            wrapper._update_graph_tasks(stream, ctx)

        mock_compute.assert_not_called()
        begin.assert_not_called()

    def test_passes_correct_workspace(self, wrapper_and_ctx):
        wrapper, ctx, stream = wrapper_and_ctx
        ctx.attn_metadata = {"layer0": MagicMock()}

        mock_op = MagicMock()
        mock_compute = MagicMock(return_value={"x": 1})
        entry = self._make_entry(compute_fn=mock_compute, op_out_fn=mock_op)
        mock_workspace = MagicMock()
        gp = MagicMock()
        gp.task_entries = {4: {"layer0": entry}}
        gp.workspaces = {4: {entry.op_desc.workspace_fn: mock_workspace}}

        with patch("omni_npu.compilation.acl_graph.get_graph_params",
                    return_value=gp), \
             patch("torch.npu.stream"), \
             patch("torch.npu.graph_task_update_begin"), \
             patch("torch.npu.graph_task_update_end"):

            wrapper._update_graph_tasks(stream, ctx)

        call_kwargs = mock_op.call_args.kwargs
        assert call_kwargs["workspace"] is mock_workspace


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
