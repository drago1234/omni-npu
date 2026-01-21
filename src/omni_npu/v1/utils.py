# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch


ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29
_current_stream = None


def get_nth_last_sep_pos(s: str, sep: str = '.', n: int = 2) -> int:
    if n < 1 or not sep:
        return -1
    
    current_pos = len(s)
    for _ in range(n):
        current_pos = s.rfind(sep, 0, current_pos)
        if current_pos == -1:
            return -1
    return current_pos


def get_last_two_parts(s: str, sep: str = '.') -> str:
    second_last_sep_pos = get_nth_last_sep_pos(s, sep=sep, n=2)

    if second_last_sep_pos == -1:
        return s
    
    return s[second_last_sep_pos + 1:]


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _current_stream = torch.npu.current_stream()
    return _current_stream
