# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omni_npu.compilation.utils module."""

import pytest
from unittest.mock import patch, MagicMock

import torch

from omni_npu.compilation.acl_graph import (
    GraphTaskEntry,
    OpDescriptor,
)
from omni_npu.compilation.utils import (
    _pad_list,
    _extract_fia_params,
    _compute_fia_dynamic_kwargs,
    _get_or_create_workspace,
    _capture_kwargs,
    capture_graph_task,
)


# ---------------------------------------------------------------------------
# _pad_list
# ---------------------------------------------------------------------------

class TestPadList:

    def test_empty_returns_empty(self):
        assert _pad_list([], 3) == []

    def test_exact_length_unchanged(self):
        assert _pad_list([1, 2, 3], 3) == [1, 2, 3]

    def test_pads_with_last_element(self):
        assert _pad_list([1, 2], 4) == [1, 2, 2, 2]

    def test_pads_with_explicit_fill(self):
        assert _pad_list([1, 2], 4, fill=0) == [1, 2, 0, 0]

    def test_truncates(self):
        assert _pad_list([1, 2, 3, 4], 2) == [1, 2]


# ---------------------------------------------------------------------------
# _extract_fia_params
# ---------------------------------------------------------------------------

class TestExtractFiaParams:

    @pytest.fixture
    def batch_descriptor(self):
        bd = MagicMock()
        bd.num_reqs = 3
        bd.num_tokens = 4
        return bd

    @pytest.fixture
    def vllm_config(self):
        cfg = MagicMock()
        cfg.scheduler_config.max_num_seqs = 10
        return cfg

    def test_gqa_with_list_seqlens(self, vllm_config, batch_descriptor):
        """GQA metadata (no decode attr): list seq_lens are padded."""
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [0, 1, 2]
        metadata.seq_lens = [5, 6]

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert len(q) == 3  # padded to num_reqs=3
        assert len(kv) == 3

    def test_gqa_with_tensor_seqlens(self, vllm_config, batch_descriptor):
        """GQA metadata with tensor seq_lens: returned directly (early return)."""
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [0, 1]
        metadata.seq_lens = torch.tensor([5, 6])

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert isinstance(kv, torch.Tensor)
        assert torch.equal(kv, torch.tensor([5, 6]))

    def test_mla_with_list_seqlens(self, vllm_config, batch_descriptor):
        """MLA metadata (has decode): list seq_lens are padded."""
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.query_cumlens = [1, 2]
        metadata.decode.seq_lens = [5, 6]

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert len(q) == 3  # padded to num_reqs=3
        assert len(kv) == 3

    def test_mla_with_tensor_seqlens(self, vllm_config, batch_descriptor):
        """MLA metadata with tensor seq_lens: returned directly."""
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.query_cumlens = torch.tensor([1, 2])
        metadata.decode.seq_lens = torch.tensor([5, 6])

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert isinstance(kv, torch.Tensor)

    def test_pads_to_num_reqs(self, vllm_config, batch_descriptor):
        """When num_reqs is set, pad to that length."""
        batch_descriptor.num_reqs = 5
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [0, 1]
        metadata.seq_lens = [3]

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert len(q) == 5
        assert len(kv) == 5

    def test_pads_to_min_when_no_num_reqs(self, vllm_config, batch_descriptor):
        """When num_reqs is None, pad to min(max_num_seqs, num_tokens)."""
        batch_descriptor.num_reqs = None
        batch_descriptor.num_tokens = 4
        vllm_config.scheduler_config.max_num_seqs = 10

        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [0, 1]
        metadata.seq_lens = [3]

        q, kv = _extract_fia_params(metadata, vllm_config, batch_descriptor)

        assert len(q) == 4  # min(10, 4) = 4
        assert len(kv) == 4


# ---------------------------------------------------------------------------
# _compute_fia_dynamic_kwargs
# ---------------------------------------------------------------------------

class TestComputeFiaDynamicKwargs:

    def test_layer_not_in_metadata_returns_none(self):
        ctx = MagicMock()
        ctx.attn_metadata = {}

        result = _compute_fia_dynamic_kwargs(
            ctx, "layer0", MagicMock(),
            seq_len_q_key="ql", seq_len_kv_key="kvl")

        assert result is None

    def test_returns_mapped_keys(self):
        metadata = MagicMock(spec_set=["query_start_loc", "seq_lens"])
        metadata.query_start_loc = [0, 1]
        metadata.seq_lens = [3]

        ctx = MagicMock()
        ctx.attn_metadata = {"layer0": metadata}
        ctx.batch_descriptor.num_reqs = 2
        ctx.batch_descriptor.num_tokens = 2

        cfg = MagicMock()
        cfg.scheduler_config.max_num_seqs = 10

        result = _compute_fia_dynamic_kwargs(
            ctx, "layer0", cfg,
            seq_len_q_key="actual_seq_lengths",
            seq_len_kv_key="actual_seq_lengths_kv")

        assert "actual_seq_lengths" in result
        assert "actual_seq_lengths_kv" in result


# ---------------------------------------------------------------------------
# _get_or_create_workspace
# ---------------------------------------------------------------------------

class TestGetOrCreateWorkspace:

    @pytest.fixture(autouse=True)
    def setup_graph_params(self):
        with patch("omni_npu.compilation.utils.get_graph_params") as mock_get:
            mock_gp = MagicMock()
            mock_gp.workspaces = {}
            mock_get.return_value = mock_gp
            yield mock_gp

    def test_new_workspace_created_and_cached(self, setup_graph_params):
        workspace = torch.tensor([100.0])
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(return_value=workspace),
        )
        op_kwargs = {"query": torch.randn(2, 4)}

        with patch("omni_npu.compilation.utils.weak_ref_tensors",
                    side_effect=lambda x: x):
            result = _get_or_create_workspace(op_desc, op_kwargs, 16)

        op_desc.workspace_fn.assert_called_once_with(**op_kwargs)
        assert torch.equal(result, workspace)
        assert setup_graph_params.workspaces[16][op_desc.workspace_fn] \
            is workspace

    def test_existing_workspace_reused(self, setup_graph_params):
        existing = torch.tensor([50.0])
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(),
        )
        setup_graph_params.workspaces = {
            16: {op_desc.workspace_fn: existing}}

        result = _get_or_create_workspace(op_desc, {}, 16)

        op_desc.workspace_fn.assert_not_called()
        assert torch.equal(result, existing)


# ---------------------------------------------------------------------------
# _capture_kwargs
# ---------------------------------------------------------------------------

class TestCaptureKwargs:

    def test_weak_refs_listed_keys(self):
        query = torch.randn(2)
        other = torch.randn(2)
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(),
            weak_ref_keys=("query",),
        )
        op_kwargs = {"query": query, "other": other}

        def fake_weak_ref(x):
            return f"weak_{id(x)}"

        with patch("omni_npu.compilation.utils.weak_ref_tensors",
                    side_effect=fake_weak_ref):
            result = _capture_kwargs(op_desc, op_kwargs)

        assert result["query"] == f"weak_{id(query)}"
        assert torch.equal(result["other"], other)

    def test_skips_none_values(self):
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(),
            weak_ref_keys=("query",),
        )
        op_kwargs = {"query": None, "other": torch.randn(2)}

        with patch("omni_npu.compilation.utils.weak_ref_tensors") as mock_wr:
            result = _capture_kwargs(op_desc, op_kwargs)

        assert result["query"] is None
        mock_wr.assert_not_called()

    def test_original_unchanged(self):
        tensor = torch.randn(2)
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(),
            weak_ref_keys=("query",),
        )
        op_kwargs = {"query": tensor}

        with patch("omni_npu.compilation.utils.weak_ref_tensors",
                    return_value="weak"):
            _capture_kwargs(op_desc, op_kwargs)

        assert op_kwargs["query"] is tensor


# ---------------------------------------------------------------------------
# capture_graph_task
# ---------------------------------------------------------------------------

class TestCaptureGraphTask:

    @pytest.fixture(autouse=True)
    def setup_graph_params(self):
        with patch("omni_npu.compilation.utils.get_graph_params") as mock_get:
            mock_gp = MagicMock()
            mock_gp.task_entries = {16: {}}
            mock_get.return_value = mock_gp
            yield mock_gp

    def test_event_wait_and_reset(self, setup_graph_params):
        op_desc = OpDescriptor(
            op_out_fn=MagicMock(),
            workspace_fn=MagicMock(return_value=torch.tensor([1.0])),
            weak_ref_keys=("query",),
        )

        with patch("torch_npu.npu.current_stream") as mock_stream, \
             patch("torch.npu.ExternalEvent") as mock_event, \
             patch("omni_npu.compilation.utils._get_or_create_workspace",
                    return_value=MagicMock()), \
             patch("omni_npu.compilation.utils._capture_kwargs",
                    return_value={}), \
             patch("omni_npu.compilation.utils.weak_ref_tensors",
                    side_effect=lambda x: x), \
             patch("torch.npu.graph_task_group_begin"), \
             patch("torch.npu.graph_task_group_end",
                    return_value=MagicMock()):

            stream_inst = MagicMock()
            mock_stream.return_value = stream_inst
            event_inst = MagicMock()
            mock_event.return_value = event_inst

            capture_graph_task(
                op_desc, {}, [torch.randn(2)], 16, "layer0")

            event_inst.wait.assert_called_once_with(stream_inst)
            event_inst.reset.assert_called_once_with(stream_inst)

    def test_task_group_brackets_op(self, setup_graph_params):
        mock_op = MagicMock()
        op_desc = OpDescriptor(
            op_out_fn=mock_op,
            workspace_fn=MagicMock(return_value=torch.tensor([1.0])),
            weak_ref_keys=("query",),
        )

        with patch("torch_npu.npu.current_stream", return_value=MagicMock()), \
             patch("torch.npu.ExternalEvent", return_value=MagicMock()), \
             patch("omni_npu.compilation.utils._get_or_create_workspace",
                    return_value=MagicMock()), \
             patch("omni_npu.compilation.utils._capture_kwargs",
                    return_value={}), \
             patch("omni_npu.compilation.utils.weak_ref_tensors",
                    side_effect=lambda x: x), \
             patch("torch.npu.graph_task_group_begin") as begin, \
             patch("torch.npu.graph_task_group_end",
                    return_value=MagicMock()) as end:

            capture_graph_task(
                op_desc, {}, [torch.randn(2)], 16, "layer0")

            begin.assert_called_once()
            mock_op.assert_called_once()
            end.assert_called_once()

    def test_stores_entry(self, setup_graph_params):
        mock_op = MagicMock()
        op_desc = OpDescriptor(
            op_out_fn=mock_op,
            workspace_fn=MagicMock(return_value=torch.tensor([1.0])),
            weak_ref_keys=("query",),
        )
        mock_handle = MagicMock()
        mock_snapshot = {"query": MagicMock()}

        with patch("torch_npu.npu.current_stream", return_value=MagicMock()), \
             patch("torch.npu.ExternalEvent", return_value=MagicMock()), \
             patch("omni_npu.compilation.utils._get_or_create_workspace",
                    return_value=MagicMock()), \
             patch("omni_npu.compilation.utils._capture_kwargs",
                    return_value=mock_snapshot), \
             patch("omni_npu.compilation.utils.weak_ref_tensors",
                    side_effect=lambda x: x), \
             patch("torch.npu.graph_task_group_begin"), \
             patch("torch.npu.graph_task_group_end",
                    return_value=mock_handle):

            capture_graph_task(
                op_desc, {"query": torch.randn(2)},
                [torch.randn(2)], 16, "layer0")

        entry = setup_graph_params.task_entries[16]["layer0"]
        assert isinstance(entry, GraphTaskEntry)
        assert entry.op_desc is op_desc
        assert entry.handle is mock_handle
        assert entry.captured_kwargs is mock_snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
