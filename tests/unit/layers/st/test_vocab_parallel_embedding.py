# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist

from .distributed_test_common import distributed_worker_pool


def _logic_vocab_parallel_embedding_parallel(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.vocab_parallel_embedding import NPUVocabParallelEmbedding

    device = torch.device(f"npu:{device}")
    vocab_size = 8
    embedding_dim = 4
    local_batch = 2

    layer = NPUVocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        org_num_embeddings=vocab_size,
        padding_size=1,
        parallel_lmhead=True,
    ).to(device)

    full_weight = torch.arange(
        vocab_size * embedding_dim, device=device, dtype=dtype
    ).reshape(vocab_size, embedding_dim)
    dist.broadcast(full_weight, src=0)
    layer.weight_loader(layer.weight, full_weight)

    base = local_rank * local_batch
    input_local = torch.tensor(
        [base % vocab_size, (base + 3) % vocab_size],
        device=device,
        dtype=torch.long,
    )

    gathered = [torch.empty_like(input_local) for _ in range(world_size)]
    dist.all_gather(gathered, input_local)
    full_input = torch.cat(gathered, dim=0)

    output = layer.forward_vocab(input_local, reduce=0)
    expected = F.embedding(full_input, full_weight)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def _logic_parallel_lm_head_all_to_all(device, local_rank, world_size, dtype):
    from omni_npu.v1.layers.vocab_parallel_embedding import (
        NPUParallelLMHead,
        NPUVocabParallelEmbedding,
    )

    device = torch.device(f"npu:{device}")
    vocab_size = 8
    embedding_dim = 4
    local_batch = 2

    embed_tokens = NPUVocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        org_num_embeddings=vocab_size,
        padding_size=1,
        parallel_lmhead=True,
    ).to(device)
    lm_head = NPUParallelLMHead(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        org_num_embeddings=vocab_size,
        padding_size=1,
        parallel_lmhead=True,
    ).to(device)

    full_weight = torch.arange(
        vocab_size * embedding_dim, device=device, dtype=dtype
    ).reshape(vocab_size, embedding_dim)
    dist.broadcast(full_weight, src=0)
    embed_tokens.weight_loader(embed_tokens.weight, full_weight)
    lm_head.tie_weights(embed_tokens)

    hidden_local = torch.randn(local_batch, embedding_dim, device=device, dtype=dtype)

    gathered = [torch.empty_like(hidden_local) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_local)
    hidden_full = torch.cat(gathered, dim=0)
    shard_size = vocab_size // world_size
    start = local_rank * shard_size
    end = start + shard_size
    logits_shard = hidden_full @ full_weight[start:end].t()

    gathered_logits = [torch.empty_like(logits_shard) for _ in range(world_size)]
    dist.all_gather(gathered_logits, logits_shard)
    chunk = logits_shard.size(0) // world_size
    expected_chunks = [
        g[chunk * local_rank:chunk * (local_rank + 1), :]
        for g in gathered_logits
    ]
    expected = torch.cat(expected_chunks, dim=-1)

    output = lm_head.forward(hidden_local, embedding_bias=None)

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def _logic_vocab_parallel_embedding_reduce_scatter_padding(
    device, local_rank, world_size, dtype
):
    from omni_npu.v1.layers.vocab_parallel_embedding import NPUVocabParallelEmbedding

    device = torch.device(f"npu:{device}")
    vocab_size = 10
    org_vocab_size = 9
    embedding_dim = 4
    padding_size = 4

    layer = NPUVocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        org_num_embeddings=org_vocab_size,
        padding_size=padding_size,
        parallel_lmhead=False,
    ).to(device)

    base_weight = torch.arange(
        org_vocab_size * embedding_dim, device=device, dtype=dtype
    ).reshape(org_vocab_size, embedding_dim)
    added_weight = torch.arange(
        (vocab_size - org_vocab_size) * embedding_dim, device=device, dtype=dtype
    ).reshape(vocab_size - org_vocab_size, embedding_dim)
    dist.broadcast(base_weight, src=0)
    dist.broadcast(added_weight, src=0)

    local_weight = torch.zeros(
        layer.num_embeddings_per_partition, embedding_dim, device=device, dtype=dtype
    )
    num_org = layer.shard_indices.org_vocab_end_index - layer.shard_indices.org_vocab_start_index
    num_org_padded = layer.shard_indices.num_org_elements_padded
    num_added = layer.shard_indices.added_vocab_end_index - layer.shard_indices.added_vocab_start_index
    added_offset = num_org_padded

    if num_org > 0:
        local_weight[:num_org] = base_weight[
            layer.shard_indices.org_vocab_start_index:layer.shard_indices.org_vocab_end_index
        ]
    if num_added > 0:
        added_start = layer.shard_indices.added_vocab_start_index - org_vocab_size
        added_end = layer.shard_indices.added_vocab_end_index - org_vocab_size
        local_weight[added_offset:added_offset + num_added] = added_weight[
            added_start:added_end
        ]
    layer.weight.data.copy_(local_weight)

    input_full = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)

    output = layer.forward_vocab(input_full, reduce=1)

    full_vocab_weight = torch.cat([base_weight, added_weight], dim=0)
    full_out = F.embedding(input_full, full_vocab_weight)
    shard_size = input_full.shape[0] // world_size
    start = local_rank * shard_size
    end = start + shard_size
    expected = full_out[start:end]

    if local_rank == 0:
        print("reduce_scatter_padding debug:")
        print("  input_full:", input_full)
        print("  full_out.shape:", tuple(full_out.shape))
        print("  expected.shape:", tuple(expected.shape))
        print("  output.shape:", tuple(output.shape))
        print("  shard_indices:",
              layer.shard_indices.org_vocab_start_index,
              layer.shard_indices.org_vocab_end_index,
              layer.shard_indices.added_vocab_start_index,
              layer.shard_indices.added_vocab_end_index)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def get_test_config():
    return {"input_split": False}


@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_config())])
def test_vocab_parallel_embedding_parallel(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_vocab_parallel_embedding_parallel,
        dtype,
        config=config,
    )


@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_config())])
def test_parallel_lm_head_all_to_all(distributed_worker_pool, dtype, config):
    distributed_worker_pool(
        _logic_parallel_lm_head_all_to_all,
        dtype,
        config=config,
    )


@pytest.mark.parametrize("dtype, config", [(torch.float32, get_test_config())])
def test_vocab_parallel_embedding_reduce_scatter_padding(
    distributed_worker_pool, dtype, config
):
    distributed_worker_pool(
        _logic_vocab_parallel_embedding_reduce_scatter_padding,
        dtype,
        config=config,
    )
