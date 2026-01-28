from collections.abc import Callable
import torch
import torch._inductor.compile_fx
import torch.fx as fx

import pytest
from unittest.mock import patch, MagicMock, ANY

from omni_npu.compilation.npugraph_ex import NpuGraphExAdaptor


@pytest.fixture
def default_npugraph_ex_adaptor():
    return NpuGraphExAdaptor()


@pytest.fixture
def mock_inputs():
    graph = MagicMock(spec=fx.GraphModule)
    example_inputs = [torch.tensor([1, 2, 3])]
    compiler_config = {"key": "value"}
    return graph, example_inputs, compiler_config


class TestNpuGraphExAdaptor:
    def test_compile_with_tuple_output(self, default_npugraph_ex_adaptor, mock_inputs):
        """Test graph_returns_tuple output is tplue"""
        with patch("torch._inductor.compile_fx.graph_returns_tuple") as mock_graph_returns_tuple:
            with patch("torchair.get_npu_backend") as mock_get_npu_backend:
                mock_graph_returns_tuple.return_value = True

                graph, example_inputs, compiler_config = mock_inputs

                mock_npugraph_ex = MagicMock()
                mock_npugraph_ex.return_value = MagicMock(spec=Callable)
                mock_get_npu_backend.return_value = mock_npugraph_ex

                result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

                assert isinstance(result, tuple)
                assert len(result) == 2
                assert callable(result[0])
                assert result[1] is None

                mock_graph_returns_tuple.assert_called_once_with(graph)
                mock_get_npu_backend.assert_called_once_with(compiler_config=ANY)
                mock_npugraph_ex.assert_called_once_with(graph, example_inputs)


    def test_compile_with_non_tuple_output(self, default_npugraph_ex_adaptor, mock_inputs):
        """Test graph_returns_tuple output is not tplue"""
        with patch("torch._inductor.compile_fx.graph_returns_tuple") as mock_graph_returns_tuple:
            with patch("torchair.get_npu_backend") as mock_get_npu_backend:
                # Trigger the logic for rewriting the FX Graph
                mock_graph_returns_tuple.return_value = False

                graph, example_inputs, compiler_config = mock_inputs
                # mock fx graph
                mock_fx_graph = MagicMock()
                mock_output_node = MagicMock()
                mock_return_value = MagicMock()
                mock_output_node.args = (mock_return_value, )
                mock_fx_graph.output_node.return_value = mock_output_node
                # mock create_node and recompile function
                mock_fx_graph.create_node = MagicMock()
                mock_fx_graph.inserting_before = MagicMock()
                graph.graph = mock_fx_graph

                mock_npugraph_ex = MagicMock()
                mock_npugraph_ex.return_value = MagicMock(spec=Callable)
                mock_get_npu_backend.return_value = mock_npugraph_ex

                result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

                assert isinstance(result, tuple)
                assert len(result) == 2
                assert callable(result[0])
                assert result[1] is None

                mock_graph_returns_tuple.assert_called_once_with(graph)
                mock_get_npu_backend.assert_called_once_with(compiler_config=ANY)
                mock_npugraph_ex.assert_called_once_with(graph, example_inputs)

                # Verify that the logic for rewriting the FX Graph has been executed
                mock_fx_graph.output_node.assert_called_once()
                mock_fx_graph.create_node.assert_called_once_with("call_function", tuple, args=([mock_return_value],))
                graph.recompile.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])