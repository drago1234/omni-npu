"""
Minimal, self-contained Ascend MLA attention backend for omni_npu.

This implementation currently delegates to the standard Ascend attention
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
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonBaseImpl,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger("vllm.omni_npu.attention.backends.mla")


class AscendMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "AscendMLA"

    @staticmethod
    def get_metadata_cls() -> type["AscendMLAMetadata"]:
        return AscendMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendMLAMetadataBuilder"]:
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["AscendMLAImpl"]:
        return AscendMLAImpl

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
class AscendMLADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor


@dataclass
class AscendMLAMetadata(MLACommonMetadata[AscendMLADecodeMetadata]):
    pass


class AscendMLAMetadataBuilder(MLACommonMetadataBuilder[AscendMLAMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AscendMLAMetadata
        )

        if self._use_fi_prefill:
            raise ValueError("Flashinfer should not be enabled.")
        if self._use_cudnn_prefill:
            raise ValueError("CUDNN should not be enabled.")
        if self.dcp_world_size > 1:
            raise ValueError("DCP should not be enabled.")
        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")
        if self.reorder_batch_threshold != 1:
            raise ValueError(f"reorder_batch_threshold should be 1, but got {self.reorder_batch_threshold}.")

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
    ) -> AscendMLADecodeMetadata:
        return AscendMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_cumlens=query_start_loc_device[1:],
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMLAMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.prefill is not None and metadata.prefill.chunked_context is not None:
            raise RuntimeError(f"Chunked prefill is not enabled yet.")
        return metadata


class AscendMLAImpl(MLACommonBaseImpl[AscendMLAMetadata]):
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
                "AscendMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "AscendMLAImpl"
            )

        if AscendMLAImpl.SHARE_MASK_TRIL_SPARSE is None:
            AscendMLAImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        x = x.view(self.num_heads, -1, self.kv_lora_rank)
        # Convert from (B, N * V) to (N, B, V)
        out = out.view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        torch.bmm(x, self.W_UV, out=out)  # Reuse "out" to make it "hot"

        # Convert from (N, B, V) to (B, N * V)
        out_new = out.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

        # Adjust output buffer shape back to the original (B, N * V)
        N, B, V = out.shape
        out.resize_((B, N * V))
        out.copy_(out_new)  # Copy result

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AscendMLAMetadata,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        tnd_cumlens = attn_metadata.prefill.query_start_loc[1:]
        o = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            v,
            query_rope=q_pe,
            key_rope=k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1),
            num_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            input_layout="TND",
            atten_mask=AscendMLAImpl.SHARE_MASK_TRIL_SPARSE,
            sparse_mode=3,
            actual_seq_lengths=tnd_cumlens,
            actual_seq_lengths_kv=tnd_cumlens,
            scale=self.scale,
            next_tokens=0
        )[0]
        return o.flatten(start_dim=-2)

    def _forward_decode(
        self,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AscendMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert attn_metadata.decode is not None

        # output shape: (N, B, L)
        o = torch.ops.npu.npu_fused_infer_attention_score(
            decode_ql_nope, kv_cache[0], kv_cache[0], query_rope=decode_q_pe, key_rope=kv_cache[1],
            num_heads=self.num_heads,
            num_key_value_heads=1, input_layout="TND_NTD",
            scale=self.scale,
            antiquant_mode=0, antiquant_scale=None,
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
        attn_metadata: AscendMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendMLAImpl"
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
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

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
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # write the latent and rope to kv cache
        if k_nope.numel() > 0:
            slots = attn_metadata.slot_mapping.view(-1, 1)
            torch_npu.npu_scatter_nd_update_(k_nope.view(-1, k_nope.shape[-1]), slots, k_c_normed)
            torch_npu.npu_scatter_nd_update_(k_rope.view(-1, k_rope.shape[-1]), slots, k_pe.squeeze(1))

        if has_prefill:
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
            decode_ql_nope = decode_q_nope.new_empty((N, B, L))

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            # call decode attn
            attn_out = self._forward_decode(
                decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded
