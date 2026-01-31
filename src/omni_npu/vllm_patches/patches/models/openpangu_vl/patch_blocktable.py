import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("BlockTablePath", BlockTable)
class BlockTablePath(VLLMPatch):
    _attr_names_to_apply = ['__init__', 'compute_slot_mapping']

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,

        #####patch start: for pangu72B-VL
        sink_len: int = 0,
        #####patch end
    ):
        """"
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        #####patch start: for pangu72B-VL
        self.sink_block_len = sink_len // self.block_size
        self.max_num_blocks_per_req = self.max_num_blocks_per_req + self.sink_block_len
        #####patch end

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * total_cp_world_size

            #####patch start: for pangu72B-VL
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            ) + self.sink_block_len
            #####patch end

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:

            #####patch start: for pangu72B-VL
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            ) + self.sink_block_len
            #####patch end

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )
 

@register_patch("MultiGroupBlockTablePatch", MultiGroupBlockTable)
class MultiGroupBlockTablePatch(VLLMPatch):
    _attr_names_to_apply = ['__init__']

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        num_speculative_tokens: int = 0,

        #####patch start: for pangu72B-VL
        sink_len: int = 0,
        #####patch end

        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        # Note(hc): each dcp rank only store
        # (max_model_len//dcp_world_size) tokens in kvcache,
        # so the block_size which used for calc max_num_blocks_per_req
        # must be multiplied by dcp_world_size.
        try:
            pcp_world_size = get_pcp_group().world_size
        except AssertionError:
            # PCP might not be initialized in testing
            pcp_world_size = 1
        try:
            dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            # DCP might not be initialized in testing
            dcp_world_size = 1

        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        total_cp_world_size = dcp_world_size * pcp_world_size

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max(
                    cdiv(max_model_len, block_size * total_cp_world_size),
                    1 + num_speculative_tokens,
                ),
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,

                #####patch start: for pangu72B-VL
                sink_len=sink_len,
                #####patch end
            )
            for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]
