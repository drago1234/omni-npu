# SPDX-License-Identifier: Apache-2.0
# Minimal HCCL-based NPU communicator for vLLM

from __future__ import annotations

from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

logger = init_logger(__name__)


class NPUCommunicator(CudaCommunicator):
    """
    Device communicator for NPU using torch.distributed with HCCL backend.
    This MVP implementation delegates collectives to torch.distributed and
    follows the same semantics as CpuCommunicator where possible.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: Optional[torch.device] = None,
        device_group: Optional[ProcessGroup] = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.dist_module = torch.distributed

        # Validate platform to help early diagnostics
        if not hasattr(torch, "npu"):
            raise RuntimeError(
                "NPUCommunicator requires torch.npu to be available. "
                "Please ensure torch_npu is properly installed."
            )

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module) -> None:
        if not self.is_ep_communicator:
            return

        moe_modules = [
            module for module in model.modules()
            if module.__class__.__name__ in ["FusedMoE", "SharedFusedMoE", "NPUFusedMoE", "NPUSharedFusedMoE"]
        ]
        for module in moe_modules:
            if hasattr(module, "quant_method") and hasattr(module.quant_method, "init_prepare_finalize"):
                module.quant_method.init_prepare_finalize(module)

    # Collectives
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.dist_module.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:  # type: ignore[override]
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        self.dist_module.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:  # type: ignore[override]
        world_size = self.world_size
        if world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_.movedim(0, dim).contiguous()
        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        self.dist_module.reduce_scatter_tensor(output_tensor, input_tensor, group=self.device_group)
        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: Optional[list[int]] = None
    ) -> torch.Tensor:  # type: ignore[override]
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()
        input_tensor = input_.movedim(0, dim).contiguous()
        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        # Fallback to standard reduce_scatter if variable sizes unsupported.
        if sizes is not None:
            self.dist_module.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        else:
            self.dist_module.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, list[torch.Tensor]],
        dim: int = 0,
        sizes: Optional[list[int]] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:  # type: ignore[override]
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        def _all_gather_single(inp: torch.Tensor, sizes: Optional[list[int]] = None):
            input_size = inp.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert inp.shape[dim] == sizes[self.rank_in_group]
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            out = torch.empty(output_size, dtype=inp.dtype, device=inp.device)
            # Torch.distributed lacks all_gatherv API; use concat gather if sizes provided
            if sizes is None:
                self.dist_module.all_gather_into_tensor(out, inp, group=self.device_group)
            else:
                # Emulate all_gatherv with a slow fallback
                tmp = []
                for r in range(world_size):
                    if self.rank_in_group == r:
                        tmp.append(inp)
                    else:
                        buf = torch.empty_like(inp)
                        self.dist_module.broadcast(buf, src=self.ranks[r], group=self.device_group)
                        tmp.append(buf)
                out.copy_(torch.cat(tmp, dim=0))
            return out

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        outputs: list[torch.Tensor] = []
        for t in input_:
            outputs.append(_all_gather_single(t, sizes=sizes))
        return outputs

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:  # type: ignore[override]
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim()
        if dim < 0:
            dim += input_.dim()
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        self.dist_module.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
        if self.rank_in_group == dst:
            return torch.cat(gather_list, dim=dim)  # type: ignore[arg-type]
        return None

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:  # type: ignore[override]
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self.dist_module.send(tensor, self.ranks[dst], self.device_group)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None) -> torch.Tensor:  # type: ignore[override]
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self.dist_module.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self) -> None:  # type: ignore[override]
        # Nothing special for HCCL torch.distributed
        return None

    def dispatch(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        return hidden_states, router_logits

    def combine(
            self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        return hidden_states
