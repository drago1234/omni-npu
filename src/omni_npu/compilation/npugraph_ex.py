from collections.abc import Callable
from typing import Any

import torch
import torch._inductor.compile_fx
import torch.fx as fx
import torchair

from vllm.compilation.counter import compilation_counter
from vllm.compilation.compiler_interface import CompilerInterface

class NpuGraphExAdaptor(CompilerInterface):
    name = "npugraph_ex"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:

        from torch._inductor.compile_fx import graph_returns_tuple
        fx_graph = graph.graph
        if not graph_returns_tuple(graph):
            output_node = fx_graph.output_node()
            return_value = output_node.args[0]
            with fx_graph.inserting_before(output_node):
                tuple_node = fx_graph.create_node("call_function", tuple, args=([return_value], ))
            output_node.args = (tuple_node, )
            graph.recompile()

        config = torchair.CompilerConfig()
        # use aclgraph mode, avoid the transformation from fx graph to Ascend IR.
        config.mode = "reduce-overhead"
        # execute FX graph in eager mode before graph mode to optimize FX graph.
        config.debug.run_eagerly = True
        # static kernel switch, suitable for static shapes or scenes with less shape changes.
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True

        npugraph_ex = torchair.get_npu_backend(compiler_config=config)
        compile_graph = npugraph_ex(graph, example_inputs)
        return compile_graph, None