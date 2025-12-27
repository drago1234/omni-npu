# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
from dataclasses import replace
import torch
import torch_npu

from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.rejection_sampler import (
    RejectionSampler,
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
    MAX_SPEC_LEN,
)

logger = init_logger(__name__)

from omni_npu.v1.sample.sampler import apply_top_k_top_p


class NPURejectionSampler(RejectionSampler):
    def __init__(self, sampler: Sampler):
        super().__init__(sampler)


    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True
            # Apdator: NPU sampler does not have logprobs_mode_override parameter
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = self.apply_logits_processors(
            raw_target_logits, sampling_metadata, metadata
        )


        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        target_probs_or_sampled_tokens = compute_probs(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            metadata,
            do_sample=True
        )

        output_token_ids = rejection_sample(
            metadata,
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs_or_sampled_tokens.to(bonus_token_ids.dtype),
            bonus_token_ids,
            sampling_metadata,
        )
        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )



def rejection_sample(
    metadata: SpecDecodeMetadata,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs_or_sampled_tokens: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1

    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs_or_sampled_tokens.is_contiguous()
    assert bonus_token_ids.is_contiguous()

    output_token_ids = simple_verify(
        metadata,
        cu_num_draft_tokens,
        draft_token_ids,
        target_probs_or_sampled_tokens,
        bonus_token_ids,
    )
    return output_token_ids.to(torch.int32)


def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
    do_sample: bool = True
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1

    num_tokens = logits.shape[0]
    if sampling_metadata.temperature is None:
        temperature = torch.ones((num_tokens,), device=logits.device, dtype=torch.float32)
    else:
        temperature = expand_batch_to_tokens(
            sampling_metadata.temperature,
            cu_num_draft_tokens,
            num_tokens,
            replace_from=GREEDY_TEMPERATURE,
            replace_to=1,
        )
    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )
    # TODO: apply min-p
    if not sampling_metadata.all_greedy:
        # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
        logits.div_(temperature.unsqueeze(-1))

        if do_sample:
            logits = logits.type(torch.bfloat16)
            if top_p is not None:
                top_p = top_p.type(torch.bfloat16)
            else:
                top_p = torch.ones(logits.shape[0], dtype=torch.bfloat16, device=logits.device)
            if top_k is not None:
                top_k = top_k.type(torch.int32)
            else:
                top_k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
            q = generate_random_sequence(
                logits, sampling_metadata, metadata,
            ).type(torch.float32)
            res = torch_npu.npu_top_k_top_p_sample(logits, top_k, top_p, q)
            return res[0]
        else:
            return apply_top_k_top_p(logits, top_k, top_p)
    else:
        if do_sample:
            return logits.argmax(dim=-1)
        else:
            return logits


def simple_verify(
    metadata: SpecDecodeMetadata,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_sampled_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
) -> torch.Tensor:
    max_spec_len = metadata.max_spec_len
    batch_size = len(cu_num_draft_tokens)
    minus_one_tensor = -torch.ones(1, 1, device=draft_token_ids.device, dtype=draft_token_ids.dtype)
    all_sampled_tokens = torch.empty(
        (draft_token_ids.numel() + bonus_token_ids.numel(),),
        device=draft_token_ids.device,
        dtype=draft_token_ids.dtype)
    all_sampled_tokens[metadata.target_logits_indices] = target_sampled_tokens
    all_sampled_tokens[metadata.bonus_logits_indices] = bonus_token_ids.view(-1)[:batch_size]
    draft_sampled_tokens = torch.where(
        draft_token_ids == target_sampled_tokens,
        all_sampled_tokens[metadata.target_logits_indices + 1],
        minus_one_tensor)
    all_sampled_tokens[metadata.target_logits_indices + 1] = draft_sampled_tokens
    if (max_spec_len == torch.tensor(metadata.num_draft_tokens)).all():
        output_token_ids = all_sampled_tokens
    else:
        num_total_tokens = sum([i + 1 for i in metadata.num_draft_tokens])
        num_sample_tokens = metadata.cu_num_draft_tokens.clone()
        num_sample_tokens[1:] -= metadata.cu_num_draft_tokens[:-1]
        num_sample_tokens += 1

        output_token_ids = -torch.ones(batch_size * (max_spec_len + 1), device=draft_token_ids.device, dtype=draft_token_ids.dtype)
        indices = torch.arange(batch_size, device=draft_token_ids.device) * max_spec_len
        indices[1:] -= metadata.cu_num_draft_tokens[:-1]
        indices = indices.repeat_interleave(
            repeats=num_sample_tokens,
            dim=0,
            output_size=num_total_tokens,
        ) + torch.arange(num_total_tokens, device=draft_token_ids.device)
        output_token_ids[indices] = all_sampled_tokens
    return output_token_ids.view(batch_size, -1)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    replaced_x = torch.where(x == replace_from, replace_to, x)
    num_tokens_tensor = cu_num_tokens.clone()
    num_tokens_tensor[1:] -= cu_num_tokens[:-1]
    expanded_x = torch.repeat_interleave(
        replaced_x,
        num_tokens_tensor,
        output_size=num_tokens,
    )
    return expanded_x


def generate_random_sequence(
    probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    spec_metadata: Optional[SpecDecodeMetadata] = None,
):
    generators = sampling_metadata.generators
    batchsize = probs.shape[0] if spec_metadata is None else len(spec_metadata.num_draft_tokens)
    req_arange = list(range(batchsize + 1))
    if spec_metadata is not None:
        for i in range(batchsize):
            req_arange[i + 1] = req_arange[i] + spec_metadata.num_draft_tokens[i] + 1
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != batchsize:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[req_arange[i]:req_arange[i + 1]].exponential_(generator=generator)
    return q