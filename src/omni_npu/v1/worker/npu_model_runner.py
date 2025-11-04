#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import copy
import gc
import os
import time
from typing import TYPE_CHECKING, Dict, Optional, Union, Any, List
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_world_size, get_dp_group, get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors, VLLM_INVALID_TOKEN_ID
from vllm.utils import (DeviceMemoryProfiler, is_pin_memory_available,
                        LayerBlockType, LazyLoader, cdiv)
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             KVConnectorOutput,
                             DraftTokenIds, LogprobsLists, LogprobsTensors,
                             ModelRunnerOutput, PoolerOutput, SamplerOutput)
from vllm.v1.worker.ubatch_splitting import (check_ubatch_thresholds,
                                             ubatch_split)
from vllm.v1.worker.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_model_runner import (AttentionGroup, GPUModelRunner,
                                             PerLayerAttnMetadata)
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, AttentionMetadataBuilder, CommonAttentionMetadata,
    create_fast_prefill_custom_backend,
    reorder_batch_to_split_decodes_and_prefills, split_attn_metadata)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.core.sched.output import SchedulerOutput


@contextmanager
def switch_torch_device():
    origin_cuda = torch.cuda
    torch.cuda = torch.npu
    try:
        yield
    finally:
        torch.cuda = origin_cuda


class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with switch_torch_device():
            super().__init__(vllm_config, device)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int64)

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        super().may_reinitialize_input_batch(kv_cache_config)
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_tensors = attn_backend.reshape_kv_cache(
                        raw_tensor,
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        dtype=kv_cache_spec.dtype,
                    )
                    kv_caches[layer_name] = kv_cache_tensors
                elif isinstance(kv_cache_spec, MambaSpec):
                    raise NotImplementedError("Mamba functionality is in progress.")
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.npu.get_device_properties
        """
        self.device_properties = torch.npu.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.npu.synchronize()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # TODO(tronzhang): error with event synchronize...
        return sampled_token_ids.tolist()
