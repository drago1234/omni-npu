# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from omni_models.models.pangu.openpangu import mHCModule
import omni_training_custom_ops


@mHCModule.register_oot
class NPUmHCModule(mHCModule):
    def __init__(
        self,
        config: PretrainedConfig,
        merge_layer_only_pre = False,
        prefix: str = "",
    ):
        super().__init__(config, merge_layer_only_pre, prefix)
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre

        # modify alpha beta gamma and phi to float32 for mhc pre operator
        if not self.merge_layer_only_pre:
            phi_output_hidden_size = (self.num_stream + 2) * self.num_stream

            self.branch_alpha_post = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.branch_alpha_res = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.branch_beta_post = nn.Parameter(torch.empty(self.num_stream, dtype=torch.float32))
            self.branch_beta_res = nn.Parameter(
                torch.empty(self.num_stream * self.num_stream, dtype=torch.float32)
            )
        else:
            phi_output_hidden_size = self.num_stream

        self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.float32))

        self.phi = ReplicatedLinear(
            self.hidden_size * self.num_stream,
            phi_output_hidden_size,
            bias=False,
            prefix=f"{prefix}.phi",
            params_dtype=torch.float32,
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.float32))

        # These buffers are materialized after weight loading.
        # Keep None at init so we do not allocate placeholder tensors and
        # can detect uninitialized state explicitly.
        self.register_buffer('_branch_alpha', None)
        self.register_buffer('_branch_beta', None)
        self._concat_tensors_initialized = False
        self._concat_tensors_dirty = True
        self._register_branch_weight_loaders()

    def _register_branch_weight_loaders(self) -> None:
        branch_params = [
            self.branch_alpha_pre,
            self.branch_beta_pre,
        ]
        if not self.merge_layer_only_pre:
            branch_params.extend(
                [
                    self.branch_alpha_post,
                    self.branch_alpha_res,
                    self.branch_beta_post,
                    self.branch_beta_res,
                ]
            )
        for param in branch_params:
            set_weight_attrs(param, {"weight_loader": self._branch_weight_loader})

    def _branch_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        self._concat_tensors_dirty = True

    def _update_concatenated_tensors(self) -> None:
        """Update the pre-computed concatenated alpha and beta tensors.

        This should be called after any of the branch parameters are loaded.
        For merge_layer_only_pre=True mode, only alpha_pre and beta_pre are concatenated.
        """
        if self.merge_layer_only_pre:
            # Pre-only mode: use single element tensors directly
            self._branch_alpha = self.branch_alpha_pre.clone()
            self._branch_beta = self.branch_beta_pre.clone()
        else:
            # Full mode: concatenate all three components
            self._branch_alpha = torch.cat([
                self.branch_alpha_pre,
                self.branch_alpha_post,
                self.branch_alpha_res
            ])
            self._branch_beta = torch.cat([
                self.branch_beta_pre,
                self.branch_beta_post,
                self.branch_beta_res
            ])
        self._concat_tensors_initialized = True
        self._concat_tensors_dirty = False

    def post_weight_load(self) -> None:
        """Process concatenated tensors after weight loading.

        Provides a manual interface to trigger tensor concatenation. Can be called
        after model weight loading. Works with lazy initialization to ensure
        concatenated tensors are always up-to-date.
        """
        self._update_concatenated_tensors()

    def hc_pre(self, x: torch.Tensor):
        if self.merge_layer_only_pre:
            dtype = x.dtype
            x = x.float()
            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            if self.mhc_use_gamma:
                weight = self.phi(x*rsqrt*self.norm_gamma.unsqueeze(0))[0]
            else:
                weight = self.phi(x)[0] * rsqrt
            h_pre, h_post, h_res = self.hc_split_sinkhorn_torch(weight)
            y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)), dim=1).squeeze(1).to(dtype)
        else:
            # Lazy initialization: ensure concatenated tensors are up-to-date
            if (not self._concat_tensors_initialized) or self._concat_tensors_dirty:
                self._update_concatenated_tensors()

            # Use pre-computed concatenated tensors
            y, h_post, h_res, _, _, _ = torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
                x.view(-1, self.num_stream, self.hidden_size), self.phi.weight,
                self._branch_alpha, self._branch_beta,
                gamma=self.norm_gamma.view(self.num_stream, -1),
                norm_eps=self.norm_eps, hc_eps=self.hc_eps
            )
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(h_res, num_iters=self.mhc_recur_norm, eps=self.hc_eps)
        return y, h_post, h_res
    
    def hc_split_sinkhorn_torch(self, weight):
        if not self.merge_layer_only_pre:
            h_pre, h_post, h_res = weight.split([self.num_stream, self.num_stream, self.num_stream * self.num_stream], dim=-1)
            h_post = 2 * torch.sigmoid(h_post * self.branch_alpha_post + self.branch_beta_post)
            h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream))
            h_res = h_res * self.branch_alpha_res + \
                    self.branch_beta_res.view(self.num_stream, self.num_stream)
            h_res, _, _ = torch.ops.custom.npu_sinkhorn(h_res, num_iters=self.mhc_recur_norm, eps=self.hc_eps)
        else:
            h_pre = weight
            h_post = None
            h_res = None
        h_pre = torch.sigmoid(h_pre * self.branch_alpha_pre + self.branch_beta_pre) + self.hc_eps
        return h_pre, h_post, h_res

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor):
        if self.merge_layer_only_pre:
            return x
        else:
            y = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
                residual.unflatten(dim=-1, sizes=(self.num_stream, -1)), h_res, x, h_post
            ).view(-1, self.num_stream * self.hidden_size)
            return y
