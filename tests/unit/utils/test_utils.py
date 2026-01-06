# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext

import pytest
import torch

from omni_npu.v1.layers.utils import get_npu_execution_type


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
@pytest.mark.parametrize("stream_input", [
    None,
    "stream_1",
])
def test_get_npu_execution_type_basic(stream_input):
    ctx = get_npu_execution_type(stream_input)
    assert ctx is not None

    if stream_input is None:
        assert isinstance(ctx, nullcontext)
    else:
        assert ctx is not None


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
def test_get_npu_execution_type_with_npu_stream():
    stream_obj = torch.npu.Stream()
    ctx = get_npu_execution_type(stream_obj)
    assert ctx is not None

    with ctx:
        t = torch.ones(2, 2, device="npu")
        assert t.device.type == "npu"


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
def test_get_npu_execution_type_other_types():
    ctx = get_npu_execution_type(12345)
    assert isinstance(ctx, nullcontext)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-m", "npu"]))
