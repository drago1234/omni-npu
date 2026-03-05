import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp

from omni_models.models.pangu.openpangu import AggregateConv
import omni_training_custom_ops


@AggregateConv.register_oot
class NPUAggregateConv(AggregateConv):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # V1 profile run
            return hidden_states
        attn_metadata = attn_metadata[self.attn_prefix]
        cache_slot_id = forward_context.cache_slot_id
        batch_descriptor = forward_context.batch_descriptor
        if attn_metadata.num_prefills > 0 or batch_descriptor.num_tokens > attn_metadata.num_decodes:
            batch_size = len(attn_metadata.query_start_loc) - 1
            conv_output_list = []
            for i in range(batch_size):
                s = attn_metadata.query_start_loc[i]
                e = attn_metadata.query_start_loc[i + 1]
                local_input = hidden_states[s: e]
                conv_input = torch.cat([self.cache_states[cache_slot_id[i]].contiguous(), local_input], dim=0)
                conv_input_transpose = conv_input.unsqueeze(dim=1)
                weight = self.merge_conv.weight.squeeze(1).transpose(0, 1)
                conv_output = torch.ops.custom.npu_aggregate_hidden(
                                conv_input_transpose, weight).reshape(conv_input.shape)
                conv_output = conv_output[self.cache_length:]
                if not self.padding and cache_slot_id[i] == 0:
                    conv_output[:self.cache_length] = 0
                conv_output_list.append(conv_output)
                self.cache_states[i + 1] = conv_input[-self.cache_length:, :]
            if e < hidden_states.shape[0]:
                conv_output_list.append(hidden_states[e:])
            conv_output =  torch.cat(conv_output_list, dim=0)
        else:
            batch_size = len(attn_metadata.query_start_loc) - 1
            num_tokens = hidden_states.shape[0]
            conv_input = torch.cat([self.cache_states[cache_slot_id[:num_tokens], ...], hidden_states.unsqueeze(1)], dim=1)
            if batch_size<=8:
                conv_input_transpose = conv_input.permute(1, 0, 2)
                weight = self.merge_conv.weight.squeeze(1).transpose(0, 1)
                conv_output = torch.ops.custom.npu_aggregate_hidden(
                                conv_input_transpose, weight)
                conv_output = conv_output[self.cache_length:].view(-1, self.hidden_size)
            else:
                conv_input_transpose = conv_input.permute(0, 2, 1)
                conv_output = self.merge_conv(conv_input_transpose).permute(0, 2, 1).view(-1, self.hidden_size)
            # idx 0 for new requests padding 0
            self.cache_states[1: num_tokens + 1, :, :] = conv_input[:, -self.cache_length:, :]
        return conv_output