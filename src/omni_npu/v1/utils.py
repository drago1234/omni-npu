# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

ACL_FORMAT_FRACTAL_NZ = 29

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
