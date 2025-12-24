from abc import abstractmethod
import types
from typing import Union

import torch
import torch.nn as nn
import torchair
from vllm.config import VllmConfig
from vllm.logger import init_logger

from omni_npu.compilation.ge_compile_config import get_torchair_config

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import get_forward_context

logger = init_logger(__name__)
torch._dynamo.config.inline_inbuilt_nn_modules=False

def GE_graph_padding(
    graph_pad_size: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
):
    attn_metadata_dict = get_forward_context().attn_metadata

    if graph_pad_size < 0:
        return input_ids, positions, attn_metadata_dict

    if not attn_metadata_dict:
        return input_ids, positions, attn_metadata_dict

    first_key = next(iter(attn_metadata_dict))
    attn_metadata = attn_metadata_dict[first_key]

    if attn_metadata is None or getattr(attn_metadata, "decode", None) is None:
        return input_ids, positions, attn_metadata_dict

    slot_mapping = attn_metadata.slot_mapping
    block_table = attn_metadata.decode.block_table

    assert block_table.shape[0] == slot_mapping.shape[0], (
        f"block_table.shape[0] ({block_table.shape[0]}) != "
        f"slot_mapping.shape[0] ({slot_mapping.shape[0]})"
    )

    # padding tensors
    pad_slots = torch.full(
        (graph_pad_size,),
        PAD_SLOT_ID,
        dtype=slot_mapping.dtype,
        device=slot_mapping.device,
    )
    pad_positions = torch.zeros(
        (graph_pad_size,),
        dtype=positions.dtype,
        device=positions.device,
    )
    pad_input_ids = torch.zeros(
        (graph_pad_size,),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    # concat 1D vectors
    slot_mapping = torch.cat([slot_mapping, pad_slots], dim=0)
    input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
    positions = torch.cat([positions, pad_positions], dim=0)

    # pad block_table on dim 0
    pad_block_table = torch.zeros(
        (graph_pad_size,) + block_table.shape[1:],
        dtype=block_table.dtype,
        device=block_table.device,
    )
    block_table = torch.cat([block_table, pad_block_table], dim=0)

    # write back
    attn_metadata.slot_mapping = slot_mapping
    attn_metadata.decode.block_table = block_table

    attn_metadata.decode.seq_lens += [1] * graph_pad_size

    return input_ids, positions, attn_metadata_dict


class TorchNpuCompilerWrapperWithCustomDispatcher:

    def __init__(self, vllm_config: VllmConfig, dynamic_arg_dims: dict[str, Union[int, list[int]]]):
        self.compiled_model = None
        self.cached_compiled_models = {}
        self.vllm_config = vllm_config
        self.dynamic_arg_dims = dynamic_arg_dims
        self.do_not_compile = not vllm_config.npu_compilation_config.use_gegraph
        if self.do_not_compile:
            return
        self.compile_dispatcher()

    def compile_dispatcher(self):
        backend = self.vllm_config.npu_compilation_config.init_backend(self.vllm_config)
        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug("not use ge cache graph")
            self.compiled_model = torch.compile(
                self.forward,
                dynamic=False,
                fullgraph=True,
                backend=backend)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug("use ge cache graph")
            for gear_size in self.vllm_config.npu_compilation_config.decode_gear_list:
                new_forward_proxy_name = f"{self.__class__.__name__}_forward_with_gear_size_{gear_size}"
                code = self.forward.__code__
                new_code = code.replace(co_name=new_forward_proxy_name, )
                new_func = types.FunctionType(new_code, self.forward.__globals__,
                                              name=new_forward_proxy_name,
                                              argdefs=self.forward.__defaults__)
                self.__dict__[new_forward_proxy_name] = new_func.__get__(self, nn.Module)
                config = get_torchair_config(self.vllm_config)
                self.cached_compiled_models[gear_size] = torchair.inference.cache_compile(
                    self.__dict__[new_forward_proxy_name],
                    config=config,
                    dynamic=False,
                    ge_cache=True,
                    fullgraph=True,
                )
                logger.debug(f"[use cache npu graph], the method name = {new_forward_proxy_name}")

    def call_dispatcher(self, *args, **kwargs):
        use_eager_model = self.should_use_eager_mode(*args, **kwargs)
        if self.do_not_compile or use_eager_model:
            return self.forward(*args, **kwargs)

        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            return self.compiled_model(*args, **kwargs)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            if len(args) == 0 and len(kwargs) == 0:
                raise ValueError(
                    "If you use the compile cache function, you must input a tensor or directly input the gear size")

            inputs = 0
            if len(args) > 0:
                inputs = args[0]
                positions = args[1]
                args = args[2:]
            elif len(kwargs) > 0:
                if kwargs["input_ids"] is not None:
                    inputs = kwargs["input_ids"]
                else:
                    inputs = kwargs["inputs_embeds"]
                positions = kwargs["positions"]

            gear_size = inputs.shape[0]
            graph_pad_size = self.vllm_config.npu_compilation_config.decode_gear_list[-1]-gear_size
            inputs, positions, attn_metadata_dict = GE_graph_padding(graph_pad_size, inputs, positions)
            gear_size = inputs.shape[0]
            kwargs["attn_metadata"] = attn_metadata_dict
            kwargs["input_ids"] = inputs
            kwargs["positions"] = positions

            return self.cached_compiled_models[gear_size](*args, **kwargs)


        logger.error(f"encountered a missed scene")
        return None

    def __call__(self, *args, **kwargs):
        return self.call_dispatcher(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def should_use_eager_mode(self, *args, **kwargs):
        ...