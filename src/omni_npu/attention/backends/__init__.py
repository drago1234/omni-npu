# SPDX-License-Identifier: Apache-2.0
# NPU attention backend shims for vLLM
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl, 
    NPUMetadata, 
    NPUAttentionBackend, 
    NPUAttentionMetadataBuilder,
)

__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
]
