# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import dataclasses
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import numpy as np
import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if hasattr(torch.ops._C_ascend, "weak_ref_tensor") and isinstance(tensor, torch.Tensor):
        return torch.ops._C_ascend.weak_ref_tensor(tensor)
    else:
        return tensor

def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.

    This function should be used in the following scenario:
    When a tensor is created during graph capture, and it's held by a method
    that's not part of the graph, we don't really need to store it, but we
    **do need** its buffer pointer. If we don't handle this, it cannot
    be garbage collected, leading to a memory leak. To avoid this,
    we should create a weak reference to the tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None,
                 update_stream: torch.npu.Stream = None,
                 attn_layer_names: list[str] = [],
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.update_stream = update_stream

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        if self.graph_pool is None:
            self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}
        self.attn_layer_names = attn_layer_names

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        logger.debug("<<< ACLGraphWrapper is being called.")
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        attn_metadata =  get_forward_context().attn_metadata
        asl = None
        aslkv = None
        if attn_metadata is not None:
            for _, metadata in attn_metadata.items():
                if hasattr(metadata, "num_accepted_tokens"):
                    # NPUMomeAttentionMetadata
                    continue
                elif not hasattr(metadata, "decode"):
                    # GQA
                    if hasattr(metadata, "query_start_loc"):
                        asl = metadata.query_start_loc[1:]
                        aslkv = metadata.seq_lens
                        break
                elif metadata.decode:
                    # MLA
                    if isinstance(metadata.decode.query_cumlens, torch.Tensor):
                        continue
                    asl = metadata.decode.query_cumlens
                    aslkv = metadata.decode.seq_lens
                    break

        if (
            aclgraph_runtime_mode == CUDAGraphMode.NONE
            or aclgraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                logger.debug(f"<<< {asl=}, {aslkv=}")
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if self.aclgraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            weak_ref_workspaces(_graph_params)
            
            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}")

        logger.debug(f"<<< Replaying aclgraph, {batch_descriptor=}, {aslkv=}")

        entry.aclgraph.replay()
        # Updates after replay to improve performance, temporal ordering ensured via inter-stream events.
        if self.vllm_config.model_config.use_mla:
            self._update_mla_attn_params(self.update_stream, forward_context,
                                   batch_descriptor.num_tokens)
        else:
            self._update_attn_fia_params(self.update_stream, forward_context,
                                    batch_descriptor.num_tokens)
        return entry.output

    def _pad_list(self, lst, n, v=None):
        if not lst or len(lst) == n:
            return copy.deepcopy(lst)
        if v is None:
            v = lst[-1]
        return (lst + [v] * (n - len(lst))) if len(lst) < n else lst[:n]

    def _update_fia_params(self, forward_context, key):
        sink_len = None
        is_gqa = False
        asl, aslkv = None, None
        metadata = forward_context.attn_metadata[key]
        if not hasattr(metadata, "decode"):
            is_gqa = True
            if hasattr(metadata, "query_start_loc"):
                asl = metadata.query_start_loc[1:]
                aslkv = metadata.seq_lens
            
        else:
            asl = metadata.decode.query_cumlens
            aslkv = metadata.decode.seq_lens
            sink_len = getattr(metadata.decode, "sink_len", None)
        
        if aslkv is None:
            return asl, aslkv
        
        if isinstance(aslkv, torch.Tensor):
            return asl, aslkv
        
        batch_descriptor = forward_context.batch_descriptor
        padding_lens = batch_descriptor.num_reqs
        if padding_lens is None:
            padding_lens = min(self.vllm_config.scheduler_config.max_num_seqs, batch_descriptor.num_tokens)
        asl = self._pad_list(asl, padding_lens) # padding asl to match gear
        aslkv = self._pad_list(aslkv, padding_lens, 0) # padding aslkv to match gear
        if sink_len is None and (aslkv[-1] != 0 or (not is_gqa)):
            aslkv[-1] += batch_descriptor.num_tokens - asl[-1]
        asl[-1] = batch_descriptor.num_tokens
        return asl, aslkv

    def _update_attn_fia_params(self, update_stream, forward_context, runtime_shape):
        if forward_context.attn_metadata is None:
            raise RuntimeError(f"attn_metadata is None")
        graph_params = get_graph_params()
        attn_metadata = forward_context.attn_metadata
        attn_keys = list(attn_metadata.keys())
        num_layers = len(attn_keys)
        if num_layers == 0:
            return
        with torch.npu.stream(update_stream):
            for key in attn_keys:
                if key not in self.attn_layer_names:
                    continue
                param = graph_params.attn_params[runtime_shape][key]
                handle = graph_params.handles[runtime_shape][key]
                event = graph_params.events[runtime_shape][key]
                op_name = param["op_name"]
                if op_name == "npu_fused_infer_attention_score":
                    query = param["query"]
                    key_cache = param["key"]
                    value = param["value"]
                    num_heads = param["num_heads"]
                    num_kv_heads = param["num_key_value_heads"]
                    input_layout = param["input_layout"]
                    scale = param["scale"]
                    block_tables = param["block_table"]
                    block_size = param["block_size"]
                    sparse_mode = param["sparse_mode"]
                    attn_mask = param["atten_mask"]
                    attn_output = param["attn_output"]
                    softmax_lse = param["softmax_lse"]
                    actual_seq_len, actual_seq_len_kv = self._update_fia_params(forward_context, key)
                    if actual_seq_len_kv is None:
                        raise RuntimeError(f"kv length is None. {(forward_context.attn_metadata[key] is None)=}")

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_len,
                        actual_seq_lengths_kv=actual_seq_len_kv,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        workspace=graph_params.workspaces.get(runtime_shape),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                elif op_name == "npu_fused_infer_attention_score_v2":
                    query = param["query"]
                    key_cache = param["key"]
                    value = param["value"]
                    num_heads = param["num_query_heads"]
                    num_kv_heads = param["num_key_value_heads"]
                    input_layout = param["input_layout"]
                    softmax_scale = param["softmax_scale"]
                    block_tables = param["block_table"]
                    block_size = param["block_size"]
                    sparse_mode = param["sparse_mode"]
                    attn_mask = param["atten_mask"]
                    attn_output = param["attn_output"]
                    softmax_lse = param["softmax_lse"]
                    pre_tokens = param.get("pre_tokens", 2147483647)
                    next_tokens = param.get("next_tokens", 2147483647)
                    learnable_sink = param.get("learnable_sink", None)
                    actual_seq_len, actual_seq_len_kv = self._update_fia_params(forward_context, key)
                    if actual_seq_len_kv is None:
                        raise RuntimeError(f"kv length is None. {(forward_context.attn_metadata[key] is None)=}")

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_len,
                        actual_seq_kvlen=actual_seq_len_kv,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        softmax_scale=softmax_scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        learnable_sink=learnable_sink,
                        workspace=graph_params.workspaces.get(runtime_shape),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                elif op_name == "npu_fused_infer_attention_sink":
                    query = param["query"]
                    key_cache = param["key"]
                    value = param["value"]
                    num_heads = param["num_query_heads"]
                    num_kv_heads = param["num_key_value_heads"]
                    input_layout = param["input_layout"]
                    softmax_scale = param["softmax_scale"]
                    sink_number = param["sink_number"]
                    pre_tokens = param["pre_tokens"]
                    next_tokens = param["next_tokens"]
                    attn_output = param["attn_output"]
                    softmax_lse = param["softmax_lse"]
                    sparse_mode = param["sparse_mode"]
                    attn_mask = param["atten_mask"]
                    block_tables = param["block_table"]
                    block_size = param["block_size"]
                    actual_seq_len, actual_seq_len_kv = self._update_fia_params(forward_context, key)
                    if actual_seq_len_kv is None:
                        raise RuntimeError(f"kv length is None. {(forward_context.attn_metadata[key] is None)=}")

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch.ops.custom.npu_fused_infer_attention_sink.out(
                        query,
                        key_cache,
                        value,
                        sink_number=sink_number,
                        input_layout=input_layout,
                        actual_seq_qlen=actual_seq_len,
                        actual_seq_kvlen=actual_seq_len_kv,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        softmax_scale=softmax_scale,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        sparse_mode=sparse_mode,
                        atten_mask=attn_mask,
                        block_table=block_tables,
                        block_size=block_size,
                        workspace=graph_params.workspaces.get(runtime_shape),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                else:
                    raise RuntimeError(f"Unsupported op name for attention: {(op_name)=}")

    def _update_mla_attn_params(self, update_stream, forward_context, runtime_shape):
        if forward_context.attn_metadata is None:
            raise RuntimeError(f"attn_metadata is None")
        attn_metadata = {
            key: metadata
            for key, metadata in forward_context.attn_metadata.items()
            if hasattr(metadata, "decode") and metadata.decode \
                and hasattr(metadata.decode, "seq_lens") and isinstance(metadata.decode.seq_lens, list)
        }
        graph_params = get_graph_params()
        with torch.npu.stream(update_stream):
            for key in attn_metadata:
                if key not in self.attn_layer_names:
                    continue
                param = graph_params.attn_params[runtime_shape][key]
                handle = graph_params.handles[runtime_shape][key]
                event = graph_params.events[runtime_shape][key]
                op_name = param["op_name"]
                if op_name == "npu_fused_infer_attention_score":
                    query = param["query"]
                    key_cache = param["key"]
                    value = param["value"]
                    query_rope = param["query_rope"]
                    key_rope = param["key_rope"]
                    num_heads = param["num_heads"]
                    num_kv_heads = param["num_key_value_heads"]
                    input_layout = param["input_layout"]
                    scale = param["scale"]
                    block_tables = param["block_table"]
                    block_size = param["block_size"]
                    sparse_mode = param["sparse_mode"]
                    attn_mask = param["atten_mask"]
                    attn_output = param["attn_output"]
                    softmax_lse = param["softmax_lse"]
                    pre_tokens = param.get("pre_tokens", 2147483647)
                    next_tokens = param.get("next_tokens", 2147483647)
                    antiquant_mode = param.get("antiquant_mode", 0)
                    antiquant_scale = param.get("antiquant_scale", None)
                    softmax_lse_flag = param.get("softmax_lse_flag", False)
                    actual_seq_len, actual_seq_len_kv = self._update_fia_params(forward_context, key)
                    if actual_seq_len_kv is None:
                        raise RuntimeError(f"kv length is None. {(forward_context.attn_metadata[key] is None)=}")

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query,
                        key_cache,
                        value,
                        query_rope=query_rope,
                        key_rope=key_rope,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_len,
                        actual_seq_lengths_kv=actual_seq_len_kv,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        antiquant_mode=antiquant_mode,
                        antiquant_scale=antiquant_scale,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        softmax_lse_flag=softmax_lse_flag,
                        workspace=graph_params.workspaces.get(runtime_shape),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                elif op_name == "npu_fused_infer_attention_sink":
                    query = param["query"]
                    key_cache = param["key"]
                    value = param["value"]
                    query_rope = param["query_rope"]
                    key_rope = param["key_rope"]
                    num_heads = param["num_query_heads"]
                    num_kv_heads = param["num_key_value_heads"]
                    input_layout = param["input_layout"]
                    softmax_scale = param["softmax_scale"]
                    pre_tokens = param["pre_tokens"]
                    next_tokens = param["next_tokens"]
                    attn_output = param["attn_output"]
                    softmax_lse = param["softmax_lse"]
                    sparse_mode = param["sparse_mode"]
                    attn_mask = param["atten_mask"]
                    block_tables = param["block_table"]
                    block_size = param["block_size"]

                    sink_kwargs = {}
                    if "sink_number" in param:
                        sink_kwargs["sink_number"] = param["sink_number"]
                    if "key_sink" in param:
                        sink_kwargs["key_sink"] = param["key_sink"]
                        sink_kwargs["value_sink"] = param["value_sink"]
                        sink_kwargs["key_rope_sink"] = param["key_rope_sink"]

                    actual_seq_len, actual_seq_len_kv = self._update_fia_params(forward_context, key)
                    if actual_seq_len_kv is None:
                        raise RuntimeError(f"kv length is None. {(forward_context.attn_metadata[key] is None)=}")

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch.ops.custom.npu_fused_infer_attention_sink.out(
                        query,
                        key_cache,
                        value,
                        query_rope=query_rope,
                        key_rope=key_rope,
                        input_layout=input_layout,
                        actual_seq_qlen=actual_seq_len,
                        actual_seq_kvlen=actual_seq_len_kv,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        softmax_scale=softmax_scale,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        sparse_mode=sparse_mode,
                        atten_mask=attn_mask,
                        block_table=block_tables,
                        block_size=block_size,
                        workspace=graph_params.workspaces.get(runtime_shape),
                        out=[attn_output, softmax_lse],
                        **sink_kwargs, 
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                else:
                    raise RuntimeError(f"Unsupported op name for mla: {(op_name)=}")


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])

@dataclass
class GraphParams:
    events: dict[int, dict[str, torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, dict[str, torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, dict[str, dict]]

_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: set[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: dict()
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: dict()
         for size in aclgraph_capture_sizes},
        {size: dict()
         for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = weak_ref_tensors(workspace)


def get_graph_params():
    return _graph_params
