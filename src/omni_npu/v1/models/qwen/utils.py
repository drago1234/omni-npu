# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.online_quantization import (
    support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import supports_any_eagle
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_cuda_view_from_cpu_tensor,
)
from vllm.model_executor.models.utils import (
    _flatten_embeddings, 
    _embedding_count_expression,
    split_list_into_ranges
)

logger = init_logger(__name__)


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)

        indices = is_multimodal.nonzero(as_tuple=True)
        # print(f"DEBUG: indices shape = {indices.shape}")
        # print(f"inputs_embeds shape: {inputs_embeds.shape}")
        # print(f"is_multimodal shape: {is_multimodal.shape}")
        # print(f"mm_embeds_flat shape: {mm_embeds_flat.shape}")

        inputs_embeds.index_put_(
        indices,  # 批次索引和序列位置索引
        mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds