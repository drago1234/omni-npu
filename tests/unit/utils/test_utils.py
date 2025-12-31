import unittest
from unittest.mock import patch, MagicMock
from contextlib import nullcontext
import torch
import torchair

from omni_npu.v1.utils import get_npu_execution_type

class TestGetNPUExecutionType(unittest.TestCase):

    @patch("torchair.scope.npu_stream_switch")
    @patch("torch.npu.stream")
    @patch("torch.compiler.is_compiling", return_value=False)
    def test_get_npu_execution_type(self, mock_is_compiling, mock_npu_stream, mock_npu_stream_switch):
        ctx = get_npu_execution_type(None)
        self.assertIsInstance(ctx, nullcontext)

        mock_npu_stream_switch.return_value = "mock_stream_switch"
        ctx = get_npu_execution_type("stream_1")
        self.assertEqual(ctx, "mock_stream_switch")
        mock_npu_stream_switch.assert_called_with("stream_1")

        stream_obj = object()
        mock_npu_stream.return_value = "mock_npu_stream"
        ctx = get_npu_execution_type(stream_obj)
        self.assertEqual(ctx, "mock_npu_stream")
        mock_npu_stream.assert_called_with(stream_obj)

    @patch("torch.compiler.is_compiling", return_value=True)
    def test_get_npu_execution_type_compiling(self, mock_is_compiling):
        # when compiling, fallback to nullcontext
        stream_obj = object()
        ctx = get_npu_execution_type(stream_obj)
        self.assertIsInstance(ctx, nullcontext)

if __name__ == "__main__":
    unittest.main()
