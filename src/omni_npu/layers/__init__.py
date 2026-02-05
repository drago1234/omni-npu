# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import NPUCompressedTensorsConfig
from omni_npu.layers.fused_moe.layer import NPUUnquantizedFusedMoEMethod, NPUFusedMoE
from omni_npu.layers.attention.npu_mla_wrapper import NPUMultiHeadLatentAttentionWrapper
from omni_npu.layers.npu_rms_norm import NPURMSNorm
from omni_npu.layers.activation import NPUSiluAndMul
from omni_npu.layers.rotary_embedding.rotary_embedding_torch_npu import NPURotaryEmbedding
from omni_npu.layers.rotary_embedding.linear_scaling_rope import NPULinearScalingRotaryEmbedding
from omni_npu.layers.rotary_embedding.llama3_rope import NPULlama3RotaryEmbedding
from omni_npu.layers.rotary_embedding.deepseek_scaling_rope import NPUDeepseekScalingRotaryEmbedding
from omni_npu.layers.rotary_embedding.yarn_scaling_rope import NPUYaRNScalingRotaryEmbedding
