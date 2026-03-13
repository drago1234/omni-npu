# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.logger import init_logger

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}


def register_attention_backend(backend: str):

    def decorator(cls: type) -> type:
        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend,
                     attn_module)
        NPU_ATTENTION_BACKEND[backend] = attn_module
        return cls

    return decorator
