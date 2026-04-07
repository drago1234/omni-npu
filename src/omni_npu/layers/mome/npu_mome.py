# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch.nn.functional as F
from omni_npu.v1.utils import on_ascend950

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.linear import (
    LinearBase, 
    BasevLLMParameter, 
    ModelWeightParameter
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import init_logger

logger = init_logger(__name__)
try:
    import omni_custom_ops
except ImportError as e:
    logger.warning(f"Failed to import omni_custom_ops: {e}")
except Exception as e:
    logger.warning(f"Error occurred while importing omni_custom_ops: {e}")

# ============================================================
# reference implementation of causal_conv1d begins

def causal_conv1d_fn(
    x_tnd: torch.Tensor,  
    weight: torch.Tensor,  
    conv_states: torch.Tensor,  
    query_start_loc: torch.Tensor, 
    cache_indices: torch.Tensor,  
    has_initial_state: torch.Tensor | None = None, 
):
    """    
    x_tnd: (cu_seq_len, dim)
        cu_seq_len = total number of tokens in all seqs

    weight: (width, dim)

    conv_states: (num_cache_lines, *, dim)

    query_start_loc: (batch + 1, ) int32
        The first element is zero, followed by the indices of the beginning token of each sequence. 

    cache_indices: (batch, )  int32
        indicates the corresponding state index, like so: conv_state = conv_states[cache_indices[batch_id]]

    has_initial_state: (batch, ) bool
        If has_initial_state[batch_idx] == Fasle, 
        then cache is not needed, and the first (width-1) tokens of the convolution output are set to zero. 
        If None, then this is equivalent to torch.zeros((batch, ), dtype=torch.bool)

    out: same shape as `x_tnd`
    """

    cu_seq_len, dim = x_tnd.shape
    batch_size = query_start_loc.shape[0] - 1  # Batch size inferred from query locations

    kernel_width = weight.size(0)
    state_len = kernel_width - 1

    # Create the output tensor
    out = torch.zeros_like(x_tnd) # Shape: (cu_seq_len, dim)

    for batch_idx in range(batch_size):
        # Extract the start and end indices for the current sequence
        start_idx = query_start_loc[batch_idx].item()
        end_idx = query_start_loc[batch_idx + 1].item()
        seq_len = end_idx - start_idx

        # Extract the sub-tensor for the current sequence
        seq_x = x_tnd[start_idx:end_idx]  # Shape: (seq_len, dim)

        # Initialize the state for the sequence
        if has_initial_state is not None and has_initial_state[batch_idx]:
            cached_state = conv_states[cache_indices[batch_idx]][:state_len] # Shape: (state_len, dim)
        else:
            cached_state = torch.zeros((state_len, dim), device=x_tnd.device, dtype=x_tnd.dtype)

        # Concatenate the cached state and the input sequence to create a larger receptive field
        padded_input = torch.cat([cached_state, seq_x], dim=0)  # Shape: (state_len + seq_len, dim)

        # Perform the convolution
        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0),  # Add batch dimension: (1, dim, state_len + seq_len)
            weight.transpose(0, 1).unsqueeze(1),        # Reshape for torch: (dim, 1, kernel_width)
            bias=None,                  # Optional bias
            stride=1,
            padding=0,                  # Padding already handled via cached states
            groups=dim                  # Grouped convolution to process each feature independently
        ) # Output shape: (1, dim, seq_len)
        result = result.squeeze(0).transpose(0, 1) # Shape: (seq_len, dim)

        # Reset the elements affected by zero padding
        if has_initial_state is None or not has_initial_state[batch_idx]:
            assert seq_len >= kernel_width - 1
            result[:kernel_width-1] = 0

        # Remove unnecessary dimensions and store to the output tensor
        out[start_idx:end_idx] = result  # Shape: (seq_len, dim)

        # Update the cached state for the current sequence
        conv_states[cache_indices[batch_idx]][:state_len] = padded_input[-state_len:] # (state_len, dim)

    return out + x_tnd # With residual connection, shape: (cu_seq_len, dim)


def causal_conv1d_update(
    x: torch.Tensor, 
    weight: torch.Tensor,  
    conv_state: torch.Tensor, 
    conv_state_indices: torch.Tensor, 
    query_start_loc: torch.Tensor = None, 
    residual: bool = True, 
    num_accepted_tokens: torch.Tensor = None, 
    pad_slot_id: int = -1,  # Padding identifier
):
    """
    x: 
        Either `[batch, seq_len, dim]`, where typically seq_len = 1 + m and m is the number of MTP heads. 
        Or `[cu_seq_len, dim]`, and in this case `query_start_loc` must be specified.  

    weight: (width, dim)

    conv_state: (-1, state_len, dim)
        state_len must be greater than the maximum of width-1+seq_len-1 for all batches. 
    
    conv_state_indices: (batch,), dtype int32  
        indicates the corresponding state index, like so: conv_state = conv_states[conv_state_indices[batch_id]]
        
    query_start_loc: (batch + 1, ) int32
        The first element is zero, followed by the indices of the beginning token of each sequence. 

    residual: bool

    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each sequence in the batch.
        If None, then speculative decoding is disabled, and is equivalent to torch.ones((batch, ), dtype=torch.int32).

    pad_slot_idx: int
        If conv_state_indices[batch_idx] == pad_slot_idx, 
        then batch_idx is a padding batch which will not be calculated. 

    out: same shape as `x`
    """

    if x.ndim == 3:
        batch, _, dim = x.shape
    else:
        assert x.ndim == 2 and query_start_loc is not None
        batch = query_start_loc.size(0) - 1
        dim = x.size(1)

    width = weight.size(0)
    state_len = conv_state.size(1)

    out = torch.empty_like(x)

    for batch_idx in range(batch):
        # Skip the padding batches in graph mode
        if conv_state_indices[batch_idx] == pad_slot_id:
            continue

        # Extract the batch's specific portion of `x` and `conv_state`
        if x.ndim == 2:
            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()
            seq_len = end_idx - start_idx
            x_seq = x[start_idx:end_idx] # Shape: (seq_len, dim)
        else:
            x_seq = x[batch_idx]  # Shape: (seq_len, dim)
            seq_len = x.size(1)

        conv_state_idx = conv_state_indices[batch_idx]
        current_conv_state = conv_state[conv_state_idx]  # Shape: (state_len, dim)

        # Rolling behavior for speculative decoding
        if num_accepted_tokens is not None:
            accepted_tokens = num_accepted_tokens[batch_idx].item()
            assert 1 <= accepted_tokens <= seq_len
            offset = accepted_tokens - 1  # Offset for speculative decoding
        else:
            offset = 0

        # Combine conv_state and x_seq for convolution
        padded_input = torch.cat(
            (current_conv_state[offset: offset + width - 1], x_seq), 
            dim=0) # (width-1 + seq_len, dim)

        # Perform causal convolution
        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0), # Shape: (1, dim, width-1 + seq_len)
            weight.transpose(0, 1).unsqueeze(1),       # Shape: (dim, 1, width)
            bias=None,
            stride=1,
            padding=0,
            groups=dim,
        )  # Shape: (1，dim, seq_len)
        result = result.squeeze(0).transpose(0, 1) # (seq_len, dim)

        # Store the result in the output tensor
        if x.ndim == 2:
            out[start_idx:end_idx] = result
        else:
            out[batch_idx] = result

        # Update the cache
        cache_len = min(padded_input.size(0) - 1, state_len)
        conv_state[conv_state_idx][:cache_len] = padded_input[-cache_len:]

    return out + x if residual else out

# reference implementation of causal_conv1d ends
# ============================================================

class ColumnParallelMOME(torch.nn.Module):
    """MOME layer with tensor (column) parallelism.

    An MOME layer is to perform causal convolution on the token dimension 
    for each feature channel independently. 
    The tensor parallelism will split the channel dimension. 

    Args:
        dim: Number of feature channels. 
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    def __init__(
        self,
        dim: int, 
        kernel_width: int = 3, 
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ):
        
        super().__init__()

        self.on_ascend950 = on_ascend950()

        # Keep input parameters
        self.dim = dim
        self.kernel_width = kernel_width
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix     
        self.disable_tp = disable_tp
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.layer_idx = extract_layer_index(self.prefix)

        assert kernel_width == 3, \
            "MOME supports kernel_width = 3 only. "
        assert quant_config is None, \
            "MOME does not support quantization currently. "

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(dim, self.tp_size)
        self.output_size_per_partition = self.input_size_per_partition
        self.output_partition_sizes = [self.output_size_per_partition]

        # if output_sizes is None:
        #     output_sizes = [output_size]
        
        # create the MOME weights
        # following UnquantizedLienarMethod.create_weights
        conv_weight = ModelWeightParameter(
            data=torch.empty(
                kernel_width,
                self.input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1, # ModelWeightParameter.load_column_parallel_weight will split the output_dim. 
            weight_loader=self.weight_loader_v2,
        )

        self.register_parameter("weight", conv_weight)

        self.update_param_tp_status()

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):        
        param.load_column_parallel_weight(loaded_weight=loaded_weight.squeeze(1).transpose(0, 1))

    def forward(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError("Please use forward_prefill or forward_decode. ")

    def forward_prefill(
        self,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.ndim == 2
        if not self.on_ascend950:
            assert conv_states.size(1) == self.kernel_width - 1
            return torch.ops.custom.npu_ai_infra_causal_conv1d_fn_add(
                x,
                self.weight,
                conv_states,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                initial_state_mode=has_initial_state,
            )
        else:
            return causal_conv1d_fn(
                x,
                self.weight,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
            )

    def forward_decode(
        self,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        pad_slot_id: int = -1,
    ) -> torch.Tensor:
        if not self.on_ascend950:
            return torch.ops.custom.npu_ai_infra_causal_conv1d_update_add(
                x,
                self.weight,
                conv_states,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                pad_slot_id=pad_slot_id,
            )
        else:
            return causal_conv1d_update(
                x,
                self.weight,
                conv_states,
                cache_indices,
                query_start_loc,
                True,
                num_accepted_tokens,
                pad_slot_id,
            )
    
    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    def extra_repr(self) -> str:
        s = f"dim={self.dim}"
        s += f", tp_size={self.tp_size}"
        return s