# SPDX-License-Identifier: Apache-2.0
# NPU attention backend shims for vLLM
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)
from omni_npu.attention.backends.mome import NPUPanguMomeBackend
from omni_npu.attention.backends.dsa import NPUDSABackend
from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.attention import NPUAttentionBackend

from omni_npu.attention.backends.utils import load_plugin_backends
load_plugin_backends()

__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
    "NPUPanguMomeBackend",
    "NPUDSABackend",
]
