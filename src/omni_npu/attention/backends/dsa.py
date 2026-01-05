"""
Minimal, self-contained NPU MLA attention with indexer backend for omni_npu.

This implementation currently delegates to the standard NPU attention
backend to remain fully self-contained and avoid external dependencies.
It satisfies vLLM's backend interface so the platform selector can
import and use it. We can iterate later with true MLA specialization.
"""
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple
import math

import torch
import torch_npu

from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonPrefillMetadata,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonBaseImpl,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class NPUDSABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "NPUDSA"

    @staticmethod
    def get_metadata_cls() -> type["NPUDSAMetadata"]:
        return NPUDSAMetadata

    @staticmethod
    def get_builder_cls() -> type["NPUDSAMetadataBuilder"]:
        return NPUDSAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUDSAImpl"]:
        return NPUDSAImpl

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, ...]:
        raw_tensor = raw_tensor.view(dtype=dtype)
        shapes = [(num_blocks, block_size, 1, 512), (num_blocks, block_size, 1, 64), (num_blocks, block_size, 1, 128)]
        sizes = [math.prod(shape) for shape in shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape) for t, shape in zip(tensors, shapes))


@dataclass
class NPUDSAPrefillMetadata(MLACommonPrefillMetadata):
    query_cumlens: torch.Tensor = None
    seq_lens: torch.Tensor = None


@dataclass
class NPUDSADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor


@dataclass
class NPUDSAMetadata(MLACommonMetadata[NPUDSADecodeMetadata]):
    pass


class NPUDSAMetadataBuilder(MLACommonMetadataBuilder[NPUDSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_uniform_spec_as_decode: ClassVar[bool] = True
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, NPUDSAMetadata
        )
        self.prefill_metadata_cls = NPUDSAPrefillMetadata
        if self._use_fi_prefill:
            raise ValueError("Flashinfer should not be enabled.")
        if self._use_cudnn_prefill:
            raise ValueError("CUDNN should not be enabled.")
        if self.dcp_world_size > 1:
            raise ValueError("DCP should not be enabled.")
        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> NPUDSADecodeMetadata:
        return NPUDSADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_cumlens=query_start_loc_device[1:],
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> NPUDSAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.prefill is not None:
            metadata.prefill.query_cumlens = metadata.prefill.query_start_loc[1:] - metadata.prefill.query_start_loc[:-1]
            metadata.prefill.seq_lens = metadata.prefill.query_cumlens
            metadata.prefill.query_start_loc = metadata.prefill.query_start_loc.tolist()
        if metadata.prefill is not None and metadata.prefill.chunked_context is not None:
            raise RuntimeError(f"Chunked prefill is not enabled yet.")
        return metadata


class NPUDSAImpl(MLACommonBaseImpl[NPUDSAMetadata]):
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        self.chunked_prefill_workspace_size = (
            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                get_current_vllm_config()
            )
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "NPUDSAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "NPUDSAImpl"
            )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        x = x.transpose(0, 1)
        x = x.view(self.num_heads, -1, self.kv_lora_rank)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        out2 = torch.bmm(x, self.W_UV)
        out_new = out2.transpose(0, 1).contiguous().view(-1, self.num_heads * self.v_head_dim)
        out.copy_(out_new)  # Copy result

    def _absorb_prolog(
        self,
        q: torch.Tensor,
    ):
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        N, B, P = q_nope.shape
        _, _, L = self.W_UK_T.shape
        ql_nope = q_nope.new_empty((N, B, L))

        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        torch.bmm(q_nope, self.W_UK_T, out=ql_nope)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)
        return ql_nope, q_pe
        
    def _apply_sparse_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attn_metadata: NPUDSAMetadata,
    ):
        if attn_metadata.prefill is not None:
            metadata = attn_metadata.prefill
        else:
            metadata = attn_metadata.decode

        actual_seq_lens_query = metadata.query_cumlens.to(torch.int32)
        actual_seq_lens_key = metadata.seq_lens.to(torch.int32)
        block_table = metadata.block_table

        bs = q_nope.shape[0]
        return torch.ops.custom.npu_sparse_flash_attention(
            query=q_nope,
            key=kv_cache[0],
            value=kv_cache[0],
            sparse_indices=self.indexer.topk_indices_buffer[:bs].view(bs, 1, self.indexer.topk_tokens),
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lens_query,
            actual_seq_lengths_kv=actual_seq_lens_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: NPUDSAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for NPUDSAImpl"
            )

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens
        assert isinstance(kv_cache, (list, tuple)), f"{type(kv_cache)=}."
        assert len(kv_cache) == 3, f"{len(kv_cache)=}."
        k_nope, k_rope, _ = kv_cache
        assert isinstance(k_nope, torch.Tensor) and isinstance(k_rope, torch.Tensor), \
            f"{type(k_nope)=}, {type(k_rope)=}."
        assert k_nope.numel() > 0 and k_rope.numel() > 0, \
            f"{k_nope.shape=}, {k_rope.shape=}"

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]

        # write the latent and rope to kv cache
        if k_nope.numel() > 0:
            slots = attn_metadata.slot_mapping.view(-1, 1)
            torch_npu.npu_scatter_nd_update_(k_nope.view(-1, k_nope.shape[-1]), slots, k_c_normed)
            torch_npu.npu_scatter_nd_update_(k_rope.view(-1, k_rope.shape[-1]), slots, k_pe.squeeze(1))

        if has_prefill:
            assert attn_metadata.prefill is not None
            # do attn absorb prolog
            q_nope, q_pe = self._absorb_prolog(prefill_q)

            # call prefill attn
            attn_out = self._apply_sparse_attention(
                q_nope, q_pe, kv_cache, attn_metadata
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output[num_decode_tokens:])

        if has_decode:
            assert attn_metadata.decode is not None
            # do attn absorb prolog
            q_nope, q_pe = self._absorb_prolog(decode_q)

            # call decode attn
            attn_out = self._apply_sparse_attention(
                q_nope, q_pe, kv_cache, attn_metadata
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded
