# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

__all__ = [
    "NPUCompressedTensorsConfig",
    "NPUUnquantizedFusedMoEMethod", 
    "NPUFusedMoE",
    "NPUMultiHeadLatentAttentionWrapper",
]

def __dir__():
    return __all__

def _load_compressed_tensors_config():
    from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import NPUCompressedTensorsConfig
    return NPUCompressedTensorsConfig

def _load_fused_moe(name):
    from omni_npu.layers.fused_moe.layer import NPUUnquantizedFusedMoEMethod, NPUFusedMoE
    return locals()[name]

def _load_mla_wrapper():
    from omni_npu.layers.attention.npu_mla_wrapper import NPUMultiHeadLatentAttentionWrapper
    return NPUMultiHeadLatentAttentionWrapper

_LAZY_LOADERS = {
    "NPUCompressedTensorsConfig": _load_compressed_tensors_config,
    "NPUUnquantizedFusedMoEMethod": lambda: _load_fused_moe("NPUUnquantizedFusedMoEMethod"),
    "NPUFusedMoE": lambda: _load_fused_moe("NPUFusedMoE"),
    "NPUMultiHeadLatentAttentionWrapper": _load_mla_wrapper,
}

def __getattr__(name):
    loader = _LAZY_LOADERS.get(name)
    if loader is not None:
        return loader()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
