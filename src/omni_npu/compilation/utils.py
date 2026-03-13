# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from omni_npu.compilation.acl_graph import (
    get_graph_params,
    update_graph_params_workspaces,
    weak_ref_tensors,
)

import torch
import torch_npu

def adjust_fia_graph_params_ref(const_args):
    adjust_list = (
        "query",
        "key",
        "value",
        "query_rope",
        "key_rope",
        "block_table",
        "atten_mask",
    )

    for adjust_item in adjust_list:
        if adjust_item in const_args and const_args[adjust_item] is not None:
            const_args[adjust_item] = weak_ref_tensors(const_args[adjust_item])

def capture_multi_fia_graph_size(
    attn_output,
    softmax_lse,
    num_tokens,
    const_args,
):
    graph_params = get_graph_params()
    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    graph_params.events[num_tokens].append(event)

    workspace = graph_params.workspaces.get(num_tokens)
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            **const_args
        )
        update_graph_params_workspaces(num_tokens, workspace)
            
    local_const_args = const_args.copy()
    local_const_args.update({
        "op_name": "npu_fused_infer_attention_score",
        "attn_output": weak_ref_tensors(attn_output),
        "softmax_lse": weak_ref_tensors(softmax_lse),
    })
    adjust_fia_graph_params_ref(local_const_args)
    graph_params.attn_params[num_tokens].append(
        local_const_args
    )

    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score.out(
        **const_args, workspace=workspace, out=[attn_output, softmax_lse]
    )
    handle = torch.npu.graph_task_group_end(stream)
    graph_params.handles[num_tokens].append(handle)

def capture_multi_fia_v2_graph_size(
    attn_output,
    softmax_lse,
    num_tokens,
    const_args,
):
    graph_params = get_graph_params()
    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    graph_params.events[num_tokens].append(event)

    workspace = graph_params.workspaces.get(num_tokens)
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            **const_args
        )
        update_graph_params_workspaces(num_tokens, workspace)
            
    local_const_args = const_args.copy()
    local_const_args.update({
        "op_name": "npu_fused_infer_attention_score_v2",
        "attn_output": weak_ref_tensors(attn_output),
        "softmax_lse": weak_ref_tensors(softmax_lse),
    })
    
    adjust_fia_graph_params_ref(local_const_args)
    graph_params.attn_params[num_tokens].append(
        local_const_args
    )

    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        **const_args, workspace=workspace, out=[attn_output, softmax_lse]
    )
    handle = torch.npu.graph_task_group_end(stream)
    graph_params.handles[num_tokens].append(handle)

def capture_multi_fia_sink_graph_size(
    attn_output,
    softmax_lse,
    num_tokens,
    const_args,
):
    graph_params = get_graph_params()
    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    graph_params.events[num_tokens].append(event)

    workspace = graph_params.workspaces.get(num_tokens)
    if workspace is None:
        workspace = torch.ops.custom._npu_fused_infer_attention_sink_get_max_workspace(
            **const_args
        )
        update_graph_params_workspaces(num_tokens, workspace)
            
    local_const_args = const_args.copy()
    local_const_args.update({
        "op_name": "npu_fused_infer_attention_sink",
        "attn_output": weak_ref_tensors(attn_output),
        "softmax_lse": weak_ref_tensors(softmax_lse),
    })
    
    adjust_fia_graph_params_ref(local_const_args)
    graph_params.attn_params[num_tokens].append(
        local_const_args
    )

    torch.npu.graph_task_group_begin(stream)
    torch.ops.custom.npu_fused_infer_attention_sink.out(
        **const_args, workspace=workspace, out=[attn_output, softmax_lse]
    )
    handle = torch.npu.graph_task_group_end(stream)
    graph_params.handles[num_tokens].append(handle)
