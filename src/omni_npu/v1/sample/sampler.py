# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that samples the next tokens from the model's outputs."""
from typing import Optional
import torch
import torch_npu

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler as TopKTopPSampler
from vllm.v1.sample.sampler import Sampler as SamplerV1
from vllm.v1.outputs import SamplerOutput as SamplerOutputV1

FP32_EPS = 2 ** -24
USE_SORT_OP_MIN_BS = 2
USE_SORT_OP_MAX_BS = 48
flashinfer_top_k_top_p_sampling = None
UNINITIALIZED_CACHED_K_NUM = -1


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits, None

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k), None

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits_sort, logits_idx


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def random_sample(
    probs: torch.Tensor,
    idx: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
    stream
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    with torch_npu.npu.stream(stream) :
        q = torch.empty_like(probs)
        # NOTE(woosuk): To batch-process the requests without their own seeds,
        # which is the common case, we first assume that every request does
        # not have its own seed. Then, we overwrite the values for the requests
        # that have their own seeds.
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.default_stream().wait_stream(stream)
    res = probs.div_(q).argmax(dim=-1).view(-1)
    if idx == None:
        return res
    else:
        return torch.gather(idx, 1, res.unsqueeze(1)).view(-1)

def generate_random_sequence(
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        stream
) -> torch.Tensor:
    with torch_npu.npu.stream(stream):
        q = torch.empty_like(logits, dtype=torch.float32)
        if len(generators) != logits.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.default_stream().wait_stream(stream)
    return q

class NPUTopKTopPSamplerV1(TopKTopPSampler):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self):
        super().__init__()
        self.dsa_stream = torch_npu.npu.Stream()
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        use_npu_top_k_top_p_sample = False  # TODO(runze): read from config
        if use_npu_top_k_top_p_sample == False or p is None or k is None:
            logits, idx = apply_top_k_top_p(logits, k, p)
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, idx, generators, self.dsa_stream), None
        else:
            logits = logits.type(torch.bfloat16)
            p = p.type(torch.bfloat16)
            k = k.type(torch.int32)
            q = generate_random_sequence(logits, generators, self.dsa_stream)
            res = torch_npu.npu_top_k_top_p_sample(logits, k, p, q)
            return res[0], None

def _apply_penalties_v1(logits: torch.Tensor, prompt_mask: torch.Tensor,
                    output_mask: torch.Tensor,
                    output_bin_counts: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor,
                    do_presence_penalties,
                    do_frequency_penalties,
                    do_repetition_penalties) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    if do_repetition_penalties:
        repetition_penalties = (repetition_penalties - 1)[:, None].repeat(1, vocab_size)
        repetition_penalties = repetition_penalties * (prompt_mask[:num_seqs] | output_mask[:num_seqs]) + 1
        logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)

    if do_frequency_penalties:
        logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts[:num_seqs]

    if do_presence_penalties:
        logits -= presence_penalties.unsqueeze(dim=1) * output_mask[:num_seqs]

    return logits

class NPUSamplerV1(SamplerV1):
    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = NPUTopKTopPSamplerV1()
        self.penalty_cache = None

    def apply_penalties(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            output_token_ids: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        if self.penalty_cache is None:
            return logits
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = _apply_penalties_v1(
                logits,
                self.penalty_cache.prompt_mask,
                self.penalty_cache.output_mask,
                self.penalty_cache.output_bin_counts,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                self.penalty_cache.do_presence_penalties,
                self.penalty_cache.do_frequency_penalties,
                self.penalty_cache.do_repetition_penalties
            )
        return logits

    def forward(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            update_penalty: Optional[bool] = True,
            predict_bonus_token: bool = False,
    ) -> SamplerOutputV1:
        result = super().forward(logits, sampling_metadata)
        if self.penalty_cache is not None and update_penalty:
            self.penalty_cache.save_token_ids(result.sampled_token_ids)
        return result
