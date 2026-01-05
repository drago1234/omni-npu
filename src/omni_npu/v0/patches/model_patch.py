# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from omni_npu.v0.patches.thinking_tag_bug_fix_patch import patch_thinking_bug_fix

def patch_vllm_distributed():
    from vllm import distributed
    from omni_npu.v0.distributed.parallel_state import (
        initialize_model_parallel,
        GroupCoordinator
    )
    from omni_npu.distributed import communicator
    from omni_npu.v0.distributed.communicator import NPUCommunicator

    distributed.parallel_state.GroupCoordinator = GroupCoordinator
    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
    communicator.NPUCommunicator = NPUCommunicator
    print("++++++++++++++++++++++++patch_vllm_distributed++++++++++++++++++++++++++")
 
def patch_rope():
    from vllm.model_executor.layers import rotary_embedding
 
    from omni_npu.v0.layers.rotary_embedding import get_rope
    rotary_embedding.get_rope = get_rope
    print("+++++++++++++++++++++++patch_rope+++++++++++++++++++++++++++")
 
def patch_embedding():
    from vllm.model_executor.layers import vocab_parallel_embedding
    from omni_npu.v0.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
    vocab_parallel_embedding.ParallelLMHead = ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding.forward = VocabParallelEmbedding.forward_vocab
    print("++++++++++++++++++++++patch_embedding++++++++++++++++++++++++++++")

def patch_sampler():
    from omni_npu.v0.layers.sampler import AscendSampler
    from vllm.model_executor.layers import sampler
    sampler.Sampler = AscendSampler
    from vllm.model_executor.layers import rejection_sampler
    from omni_npu.v0.layers.sampler import RejectionSampler, _multinomial
    rejection_sampler.RejectionSampler = RejectionSampler
    rejection_sampler._multinomial = _multinomial
    print("++++++++++++++++++++++patch_sampler++++++++++++++++++++++++++++")

def patch_compilation():
    from omni_npu.v0.compilation.decorators import _support_torch_compile
    from vllm.compilation import decorators
    decorators._support_torch_compile = _support_torch_compile
    print("+++++++++++++++++++++++patch_compilation+++++++++++++++++++++++++++")

def patch_linear():
    from vllm.model_executor.layers import linear
    from omni_npu.v0.layers.linear import AscendUnquantizedLinearMethod
    linear.UnquantizedLinearMethod = AscendUnquantizedLinearMethod
    print("++++++++++++++++++++++patch_linear++++++++++++++++++++++++++++")

def patch_update_xgrammar_graph():
    exit_code = os.system(f"bash {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'update_xgrammar_graph.sh')}")
    if exit_code == 0:
        print("+++++++++++++++++++++++patch_update_xgrammar_graph success+++++++++++++++++++++++++++")
    else:
        print("+++++++++++++++++++++++patch_update_xgrammar_graph failed+++++++++++++++++++++++++++")

_patch_done = False

def patch_all():
    global _patch_done
    if _patch_done:
        return
    patch_vllm_distributed()
    patch_rope()
    patch_embedding()
    patch_linear()
    patch_thinking_bug_fix()
    _patch_done = True
