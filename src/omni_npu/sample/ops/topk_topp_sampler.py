# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch_npu
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler as V1TopKTopPSampler



logger = init_logger(__name__)

def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    if k is None and p is None:
        return logits
    logits = logits.type(torch.bfloat16)
    if p is not None:
        p = p.type(torch.bfloat16)
    else:
        p = torch.ones(logits.shape[0], dtype=torch.bfloat16, device=logits.device)
    if k is not None:
        k = k.type(torch.int32)
    else:
        k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
    _, logits = torch_npu.npu_top_k_top_p_sample(logits, k, p, None, is_need_logits=True)
    return logits

# edit from vllm.v1.sample.ops.topk_topp_sampler.random_sample
def generate_coins(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    stream,
):
    with torch_npu.npu.stream(stream):
        q = torch.empty_like(probs, dtype=torch.float32)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.default_stream().wait_stream(stream)
    return q

class NPUTopKTopPSampler(V1TopKTopPSampler):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs", dsa_stream = None) -> None:
        super().__init__(logprobs_mode)
        self.apply_top_k_top_p = apply_top_k_top_p_npu
        self.forward = self.forward_npu
        self.dsa_stream = dsa_stream if dsa_stream is not None else torch_npu.npu.Stream()
    
    def forward_npu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = logits.type(torch.bfloat16)
        if p is not None:
            p = p.type(torch.bfloat16)
        else:
            p = torch.ones(logits.shape[0], dtype=torch.bfloat16, device=logits.device)
        if k is not None:
            k = k.type(torch.int32)
        else:
            k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
        q = generate_coins(logits, generators, self.dsa_stream)
        token_ids, logits = torch_npu.npu_top_k_top_p_sample(logits, k, p, q, is_need_logits=True)

        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        return token_ids, logits_to_return
