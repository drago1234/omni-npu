import torch
import unittest

try:
    import torch_npu
    NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.device_count() > 0
except ImportError:
    NPU_AVAILABLE = False

skipif_no_npu = unittest.skipIf(not NPU_AVAILABLE, "NPU hardware not available")