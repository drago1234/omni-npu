# SPDX-License-Identifier: Apache-2.0
# NPU attention backend shims for vLLM
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)
from omni_npu.attention.backends.pangu_hybrid import NPUPanguMomeBackend

__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
    "NPUPanguMomeBackend",
]
