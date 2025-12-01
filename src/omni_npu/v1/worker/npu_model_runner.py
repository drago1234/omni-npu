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

from contextlib import contextmanager, nullcontext

import torch
from vllm.config import VllmConfig, CUDAGraphMode

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from omni_npu.v1.sample.sampler import AscendSamplerV1
from omni_npu.layers import fused_moe
from vllm.logger import logger
from vllm.utils import DeviceMemoryProfiler
from vllm.model_executor.model_loader import get_model
from omni_npu.compilation.acl_graph import (ACLGraphWrapper,
                                               set_graph_params,
)
from typing import Optional
from dataclasses import dataclass
@contextmanager
def switch_torch_device():
    origin_cuda = torch.cuda
    torch.cuda = torch.npu
    try:
        yield
    finally:
        torch.cuda = origin_cuda

@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with switch_torch_device():
            super().__init__(vllm_config, device)

        # NOTE:(runze) query_lens and seq_lens arguments need to be int64 in FIA op,
        # otherwise an implicit conversion would happen which might hurt performance.
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1,
                                                 dtype=torch.int64)
        self.seq_lens = self._make_buffer(self.max_num_reqs,
                                          dtype=torch.int64)

        # FIXME(runze): reusing VLLM's sampler fails, this sampler class is from omni_infer.
        # need to check why and try to remove it.
        self.sampler = AscendSamplerV1()

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

    # Wrap original model with ACLGraphWrapper
    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # wrap the model with full graph wrapper if needed.
        logger.debug(f"<<< {self.compilation_config.cudagraph_mode.has_full_cudagraphs()=}")
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            set_graph_params(self.compilation_config.cudagraph_capture_sizes)
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)
            logger.debug("<<< Wrapped original model with ACLGraphWrapper")

    def capture_model(self) -> int:
        logger.debug("<<< Capturing model in npu_model_runner")
        with switch_torch_device():
            super().capture_model()