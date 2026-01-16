# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC
from typing import Tuple, Dict, Optional

import torch, torch_npu

class PrefetcherBase(ABC):   
    '''
    Base utility for coordinating torch_npu prefetch calls.

    Attributes
    ----------
    min_prefetch_size:
        Minimum prefetch size in bytes.
    max_prefetch_size:
        Default cap (in bytes) used when a caller does not specify a size.
    prefetch_tensors_map_map:
        Mapping of tensor names to (tensor, prefetch) tuples that
        subclasses populate before prefetching attention weights.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_prefetch_size = 0
        self.max_prefetch_size = 90
        self.prefetch_tensors_map: Optional[Dict[str, Tuple[torch.Tensor, int]]] = None
    
    @staticmethod
    def prefetch_weight(weight: Optional[torch.Tensor], trigger: Optional[torch.Tensor], prefetch_size: int) -> None:
        '''
        Parameters
        ----------
        weight:
            The tensor whose data should be prefetched.
        trigger:
            A tensor on the target device used to trigger the prefetch.
        prefetch_size:
            Prefetch byte size; no-op when zero or negative.
        '''
        if weight is None or trigger is None or prefetch_size is None or prefetch_size <= 0:
            return   
        torch_npu.npu_prefetch(weight, trigger, prefetch_size * (2 ** 20))

    def prefetch_moe(self, 
                     trigger: torch.Tensor,
                     prefetch_experts: bool = True,
                     prefetch_shared_experts: bool = True) -> None:
        '''
        Prefetch MoE-specific weights.
        Subclasses should override to enqueue the exact MoE weights they need.
        '''
        pass

    def prefetch_attention(self, trigger: torch.Tensor) -> None:
        '''
        Prefetch attention weights from the next layer when available.
        '''
        if not self.prefetch_tensors_map:
            return
        for name, (weight, prefetch_size) in self.prefetch_tensors_map.items():
            self.prefetch_weight(weight, trigger, prefetch_size)

    def prefetch_kvcache(self, trigger: torch.Tensor, kv_prefetch: Optional[Tuple[torch.Tensor, ...]] = None) -> None:
        '''
        Parameters
        ----------
        trigger:
            Tensor on the destination device to trigger the prefetch.
        kv_prefetch:
            Optional tuple containing KV cache tensors; only the first element
            is used here because downstream callers pass (k_cache, v_cache).
        '''
        if not kv_prefetch or not isinstance(kv_prefetch, Tuple) or not kv_prefetch[0].numel():
            return
        if kv_prefetch is not None and isinstance(kv_prefetch, Tuple) and kv_prefetch[0].numel():
            self.prefetch_weight(kv_prefetch[0], trigger, self.max_prefetch_size)

