# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.nn.functional as F


def _install_stubs(monkeypatch):
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_format_cast = lambda tensor, fmt: tensor
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name=None: None
    model_executor = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    vocab_mod = types.ModuleType(
        "vllm.model_executor.layers.vocab_parallel_embedding")
    quant_base = types.ModuleType(
        "vllm.model_executor.layers.quantization.base_config")
    utils_mod = types.ModuleType("vllm.model_executor.utils")
    dist_mod = types.ModuleType("vllm.distributed")

    def pad_vocab_size(size, padding_size):
        if padding_size <= 0:
            return size
        return ((size + padding_size - 1) // padding_size) * padding_size

    class FakeIndices:
        def __init__(self, org_start, org_end, org_padding, added_start,
                     added_end, total_padded):
            self.org_vocab_start_index = org_start
            self.org_vocab_end_index = org_end
            self.num_org_vocab_padding = org_padding
            self.added_vocab_start_index = added_start
            self.added_vocab_end_index = added_end
            self.num_elements_padded = total_padded

    class VocabParallelEmbedding(torch.nn.Module):
        @staticmethod
        def _get_indices(num_embeddings_padded, org_vocab_size_padded,
                         num_embeddings, org_vocab_size, tp_rank, tp_size):
            org_start = 0
            org_end = org_vocab_size
            org_padding = org_vocab_size_padded - org_vocab_size
            added_start = org_vocab_size_padded
            added_end = org_vocab_size_padded + (num_embeddings -
                                                 org_vocab_size)
            return FakeIndices(org_start, org_end, org_padding, added_start,
                               added_end, num_embeddings_padded)

        def weight_loader(self, param, loaded_weight):
            param.data.copy_(loaded_weight)

    class UnquantizedEmbeddingMethod:
        def create_weights(self,
                           layer,
                           embedding_dim,
                           shapes,
                           _output_dim,
                           _num_embeddings_padded,
                           params_dtype=None,
                           weight_loader=None):
            weight = torch.empty(shapes[0], embedding_dim, dtype=params_dtype)
            layer.weight = torch.nn.Parameter(weight)

        def embedding(self, layer, input_):
            return F.embedding(input_, layer.weight)

        def apply(self, layer, hidden_states, bias=None):
            logits = hidden_states @ layer.weight.t()
            if bias is not None:
                logits = logits + bias
            return logits

    class QuantizeMethodBase:
        pass

    class QuantizationConfig:
        def get_quant_method(self, _layer, prefix=""):
            return None

        def get_name(self):
            return "none"

    def method_has_implemented_embedding(_method_cls):
        return True

    utils_mod.set_weight_attrs = lambda *args, **kwargs: None

    dist_mod.divide = lambda value, divisor: value // divisor
    dist_mod.get_dp_group = lambda: None
    dist_mod.tensor_model_parallel_all_gather = lambda x: x
    dist_mod.get_tensor_model_parallel_rank = lambda: 0
    dist_mod.get_tensor_model_parallel_world_size = lambda: 1
    dist_mod.tensor_model_parallel_all_reduce = lambda x: x
    dist_mod.tensor_model_parallel_reduce_scatter = lambda x: x

    vocab_mod.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
    vocab_mod.pad_vocab_size = pad_vocab_size
    vocab_mod.VocabParallelEmbedding = VocabParallelEmbedding

    quant_base.QuantizationConfig = QuantizationConfig
    quant_base.QuantizeMethodBase = QuantizeMethodBase
    quant_base.method_has_implemented_embedding = method_has_implemented_embedding

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)
    monkeypatch.setitem(sys.modules,
                        "vllm.model_executor.layers.vocab_parallel_embedding",
                        vocab_mod)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization.base_config",
        quant_base)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist_mod)

    layers_pkg = types.ModuleType("omni_npu.v1.layers")
    repo_root = Path(__file__).resolve().parents[3]
    layers_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.layers", layers_pkg)

    parallel_state_mod = types.ModuleType(
        "omni_npu.v1.distributed.parallel_state_ext")

    class FakeGroup:
        local_rank = 0
        world_size = 1

        def all_gather(self, tensor, dim=0):
            return tensor

    parallel_state_mod.get_local_world_group = lambda: FakeGroup()
    parallel_state_mod.get_world_group = lambda: FakeGroup()
    monkeypatch.setitem(sys.modules,
                        "omni_npu.v1.distributed.parallel_state_ext",
                        parallel_state_mod)

    comm_mod = types.ModuleType("omni_npu.v1.distributed.communication_op_ext")
    comm_mod.all_gather_local = lambda x, dim=0: x
    comm_mod.all_to_all_local = lambda x, scatter_dim=0, gather_dim=-1: x
    comm_mod.reduce_scatter_local = lambda x: x
    monkeypatch.setitem(sys.modules,
                        "omni_npu.v1.distributed.communication_op_ext", comm_mod)


def _import_module(monkeypatch):
    _install_stubs(monkeypatch)
    monkeypatch.delitem(
        sys.modules, "omni_npu.v1.layers.vocab_parallel_embedding", raising=False
    )
    module = importlib.import_module(
        "omni_npu.v1.layers.vocab_parallel_embedding")
    return importlib.reload(module)


def test_get_masked_input_and_mask_no_added_vocab(monkeypatch):
    module = _import_module(monkeypatch)
    input_ids = torch.tensor([0, 3, 4, 7, 9])
    masked_input, mask = module.get_masked_input_and_mask(
        input_ids,
        org_vocab_start_index=0,
        org_vocab_end_index=5,
        num_org_vocab_padding=2,
        added_vocab_start_index=5,
        added_vocab_end_index=5,
    )

    assert torch.equal(masked_input, torch.tensor([0, 3, 4, 0, 0]))
    assert torch.equal(mask, torch.tensor([False, False, False, True, True]))


def test_get_masked_input_and_mask_with_added_vocab(monkeypatch):
    module = _import_module(monkeypatch)
    input_ids = torch.tensor([0, 2, 6, 7, 9])
    masked_input, mask = module.get_masked_input_and_mask(
        input_ids,
        org_vocab_start_index=0,
        org_vocab_end_index=4,
        num_org_vocab_padding=0,
        added_vocab_start_index=6,
        added_vocab_end_index=8,
    )

    assert torch.equal(masked_input, torch.tensor([0, 2, 4, 5, 0]))
    assert torch.equal(mask, torch.tensor([False, False, False, False, True]))


def test_forward_vocab_single_partition(monkeypatch):
    module = _import_module(monkeypatch)
    layer = module.NPUVocabParallelEmbedding(
        num_embeddings=8,
        embedding_dim=4,
        org_num_embeddings=8,
        padding_size=1,
    )
    weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    layer.weight.data.copy_(weight)

    input_ids = torch.tensor([1, 3], dtype=torch.int64)
    output = layer.forward_vocab(input_ids)

    expected = F.embedding(input_ids, weight)
    assert torch.equal(output, expected)


def test_parallel_lm_head_tie_weights(monkeypatch):
    module = _import_module(monkeypatch)
    embed_tokens = module.NPUVocabParallelEmbedding(
        num_embeddings=8,
        embedding_dim=4,
        org_num_embeddings=8,
        padding_size=1,
    )
    lm_head = module.NPUParallelLMHead(
        num_embeddings=8,
        embedding_dim=4,
        org_num_embeddings=8,
        padding_size=1,
    )

    returned = lm_head.tie_weights(embed_tokens)

    assert returned is lm_head
    assert lm_head.weight is embed_tokens.weight


def test_parallel_lm_head_tie_weights_gguf(monkeypatch):
    module = _import_module(monkeypatch)

    class DummyQuantConfig:
        def get_quant_method(self, _layer, prefix=""):
            return None

        def get_name(self):
            return "gguf"

    embed_tokens = module.NPUVocabParallelEmbedding(
        num_embeddings=8,
        embedding_dim=4,
        org_num_embeddings=8,
        padding_size=1,
        quant_config=DummyQuantConfig(),
    )
    lm_head = module.NPUParallelLMHead(
        num_embeddings=8,
        embedding_dim=4,
        org_num_embeddings=8,
        padding_size=1,
        quant_config=DummyQuantConfig(),
    )

    returned = lm_head.tie_weights(embed_tokens)

    assert returned is embed_tokens

