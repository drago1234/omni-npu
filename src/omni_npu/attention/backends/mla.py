"""
Minimal, self-contained NPU MLA attention backend for omni_npu.

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
from vllm.forward_context import get_forward_context
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonBaseImpl,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.platforms import current_platform
from omni_npu.attention import ops


logger = init_logger(__name__)


class NPUMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "NPUMLA"

    @staticmethod
    def get_metadata_cls() -> type["NPUMLAMetadata"]:
        return NPUMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["NPUMLAMetadataBuilder"]:
        return NPUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUMLAImpl"]:
        return NPUMLAImpl

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
        shapes = [(num_blocks, block_size, 512), (num_blocks, block_size, 64)]
        sizes = [math.prod(shape) for shape in shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape) for t, shape in zip(tensors, shapes))


@dataclass
class NPUMLADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor
    mc2_mask: torch.Tensor = None

@dataclass
class NPUMLAMetadata(MLACommonMetadata[NPUMLADecodeMetadata]):
    pass


class NPUMLAMetadataBuilder(MLACommonMetadataBuilder[NPUMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, NPUMLAMetadata
        )

        if self._use_fi_prefill:
            raise ValueError("Flashinfer should not be enabled.")
        if self._use_cudnn_prefill:
            raise ValueError("CUDNN should not be enabled.")
        if self.dcp_world_size > 1:
            raise ValueError("DCP should not be enabled.")
        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")

        uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )
        max_decode_tokens = self.vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
        self.mc2_mask = torch.zeros(max_decode_tokens, dtype=torch.bool, device=current_platform.device_type)

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> NPUMLADecodeMetadata:
        return NPUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_cpu.tolist(),
            query_cumlens=query_start_loc_cpu[1:].tolist(),
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> NPUMLAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.decode is not None and self.vllm_config.kv_transfer_config is not None:
            # for pd-mixed, TP is used, no need to use mc2_mask
            metadata.decode.mc2_mask = self.generate_activate_mask(common_attn_metadata.num_actual_tokens)
        if metadata.prefill is not None:
            metadata.prefill.query_start_loc = metadata.prefill.query_start_loc.tolist()
        return metadata

    def generate_activate_mask(self, actual_seqs_num):
        self.mc2_mask.fill_(False)
        self.mc2_mask[:actual_seqs_num].fill_(True)
        return self.mc2_mask


class NPUMLAImpl(MLACommonBaseImpl[NPUMLAMetadata]):
    can_return_lse_for_decode: bool = False
    SHARE_MASK_TRIL_SPARSE = None

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
                "NPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "NPUMLAImpl"
            )

        if NPUMLAImpl.SHARE_MASK_TRIL_SPARSE is None:
            NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )
            NPUMLAImpl.DECORE_ATTN_MASK = NPUMLAImpl.SHARE_MASK_TRIL_SPARSE.to(torch.uint8)

    def _v_up_proj(self, x: torch.Tensor):
        x = x.view(self.num_heads, -1, self.kv_lora_rank)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        out2 = torch.bmm(x, self.W_UV)
        out_new = out2.transpose(0, 1).contiguous().view(-1, self.num_heads * self.v_head_dim)
        return out_new

    def _compute_prefill_context(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.gather_and_maybe_dequant_cache(
                src_cache=(kv_c_cache, k_pe_cache),
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            attn_output, attn_softmax_lse = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_pe,
                key_rope=k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1),
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                input_layout="TND",
                atten_mask=None,
                sparse_mode=0,  # for prefix, no mask on attention matrix
                actual_seq_lengths=prefill_metadata.query_start_loc[1:],
                actual_seq_lengths_kv=prefill_metadata.chunked_context.cu_seq_lens[i, 1:],
                scale=self.scale,
                next_tokens=0,
                softmax_lse_flag=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                ops.merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        has_context = attn_metadata.prefill.chunked_context is not None
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        tnd_cumlens = attn_metadata.prefill.query_start_loc[1:]
        output, output_lse = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            v,
            query_rope=q_pe,
            key_rope=k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1),
            num_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            input_layout="TND",
            atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
            sparse_mode=3,
            actual_seq_lengths=tnd_cumlens,
            actual_seq_lengths_kv=tnd_cumlens,
            scale=self.scale,
            next_tokens=0,
            softmax_lse_flag=has_context,
        )

        if has_context:
            context_output, context_lse = self._compute_prefill_context(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c_cache=kv_cache[0],
                k_pe_cache=kv_cache[1],
                attn_metadata=attn_metadata,
                k_scale=k_scale,
            )
            merged_output = torch.empty_like(output)
            ops.merge_attn_states(
                output=merged_output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=output,
                suffix_lse=output_lse,
            )
            output = merged_output

        return output.flatten(start_dim=-2)

    def _forward_decode(
        self,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert attn_metadata.decode is not None

        ## TODO: TND_NTD layout bug. Currently graph mode only support BSND.
        # output shape: (N, T, D)
        batch_descriptor = get_forward_context().batch_descriptor
        if batch_descriptor is not None and batch_descriptor.uniform:
            batch_size = attn_metadata.decode.block_table.shape[0]
            T, N, D = decode_ql_nope.shape
            S = attn_metadata.num_decode_tokens // attn_metadata.num_decodes
            if T > batch_size:
                attn_mask = NPUMLAImpl.DECORE_ATTN_MASK
                sparse_mode = 3
            else:
                attn_mask = None
                sparse_mode = 0

            o = torch.ops.npu.npu_fused_infer_attention_score(
                decode_ql_nope.view(-1, S, N, D), kv_cache[0], kv_cache[0],
                query_rope=decode_q_pe.view(-1, S, N, self.qk_rope_head_dim),
                key_rope=kv_cache[1],
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="BSND",
                scale=self.scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=attn_metadata.decode.block_table,
                block_size=128,
                # actual_seq_lengths=attn_metadata.decode.query_cumlens,
                actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                atten_mask=attn_mask,
                sparse_mode=sparse_mode,
            )[0]
            o = o.view(-1, N, D).transpose(0, 1).contiguous()
        else:
            num_tokens = attn_metadata.decode.query_cumlens[-1]
            decode_ql_nope = decode_ql_nope[:num_tokens]
            decode_q_pe = decode_q_pe[:num_tokens]
            o = torch.ops.npu.npu_fused_infer_attention_score(
                decode_ql_nope, kv_cache[0], kv_cache[0],
                query_rope=decode_q_pe,
                key_rope=kv_cache[1],
                num_heads=self.num_heads,
                num_key_value_heads=1,
                input_layout="TND_NTD",
                scale=self.scale,
                antiquant_mode=0,
                antiquant_scale=None,
                block_table=attn_metadata.decode.block_table,
                block_size=128,
                actual_seq_lengths=attn_metadata.decode.query_cumlens,
                actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
            )[0]

        return o

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for NPUMLAImpl"
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
        assert len(kv_cache) == 2, f"{len(kv_cache)=}."
        k_nope, k_rope = kv_cache
        assert isinstance(k_nope, torch.Tensor) and isinstance(k_rope, torch.Tensor), \
            f"{type(k_nope)=}, {type(k_rope)=}."
        assert k_nope.numel() > 0 and k_rope.numel() > 0, \
            f"{k_nope.shape=}, {k_rope.shape=}"

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        # logger.debug(f"<<< {output.shape=}, {q.shape=}, {k_c_normed.shape=}, {k_pe.shape=}")
        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        
        decode_q = q

        # write the latent and rope to kv cache
        if k_nope.numel() > 0:
            slots = attn_metadata.slot_mapping.view(-1, 1)
            torch_npu.npu_scatter_nd_update_(k_nope.view(-1, k_nope.shape[-1]), slots, k_c_normed)
            torch_npu.npu_scatter_nd_update_(k_rope.view(-1, k_rope.shape[-1]), slots, k_pe.squeeze(1))

        if has_prefill:
            output = output[:num_actual_toks, ...]
            q = q[:num_actual_toks, ...]
            k_c_normed = k_c_normed[:num_actual_toks, ...]
            k_pe = k_pe[:num_actual_toks, ...]
            prefill_q = q[num_decode_tokens:]
            prefill_k_pe = k_pe[num_decode_tokens:]
            prefill_k_c_normed = k_c_normed[num_decode_tokens:]
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
            )

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            # call decode attn
            attn_out = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer
            )

            # v_up projection
            out_proj = self._v_up_proj(attn_out)
            output[:out_proj.shape[0]].copy_(out_proj)
        return output_padded
