import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

import omni_npu.v1.models.deepseek.deepseek_mtp as deepseek_mtp_mod


# ==============================================================================
# Lightweight fakes for init/forward coverage
# ==============================================================================

class _FakeRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x


class _FakeRowParallelFlashCommLinear(nn.Module):
    """Return tensor directly (not tuple) because code uses it as a module output."""
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def forward(self, x):
        return torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)


class _FakeRotaryEmb:
    def __init__(self):
        self.calls = 0

    def get_cos_sin(self, positions: torch.Tensor):
        self.calls += 1
        bsz = positions.numel()
        cos = torch.zeros((bsz, 1, 1, 4), dtype=torch.float32, device=positions.device)
        sin = torch.zeros((bsz, 1, 1, 4), dtype=torch.float32, device=positions.device)
        return cos, sin


class _FakeSelfAttn:
    def __init__(self):
        self.rotary_emb = _FakeRotaryEmb()


class _FakeParallelLMHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _FakeVocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.calls = 0

    def forward(self, input_ids: torch.Tensor):
        self.calls += 1
        return torch.zeros((input_ids.shape[0], self.hidden_size),
                           dtype=torch.float32, device=input_ids.device)


class _FakeLogitsProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, head, hidden_states):
        self.calls.append({"head": head, "hidden_states": hidden_states})
        return torch.zeros((hidden_states.shape[0], self.vocab_size),
                           dtype=torch.float32, device=hidden_states.device)


class _FakeDecoderLayer(nn.Module):
    """Used for DeepSeekMultiTokenPredictorLayer and DeepSeekMultiTokenPredictor."""
    def __init__(self, mlp=None):
        super().__init__()
        self.self_attn = _FakeSelfAttn()
        self.mlp = mlp if mlp is not None else nn.Linear(1, 1)

    def forward(self, hidden_states, cos, sin, residual=None):
        out = hidden_states + 2.0
        residual_out = torch.ones_like(out)
        return out, residual_out


class _FakeMoE(nn.Module):
    """Used to hit isinstance(layer.mlp, DeepseekV2MoE) branch."""
    def __init__(self):
        super().__init__()
        self.experts = ["e0", "e1"]


class _FakeModelReturn(nn.Module):
    """nn.Module callable stub to replace mtp.model (cannot use MagicMock)."""
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.out = out
        self.calls = 0
        self.last_args = None
        self.last_kwargs = None

    def forward(self, *args, **kwargs):
        self.calls += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.out


# ==============================================================================
# Existing helpers for load_weights tests
# ==============================================================================

class _FakeParam:
    """A minimal param-like object that supports weight_loader."""
    def __init__(self, name: str):
        self.name = name
        self.load_calls = []

        def _loader(param, weight, *args, **kwargs):
            self.load_calls.append(
                {"param": param, "weight": weight, "args": args, "kwargs": kwargs}
            )
            if kwargs.get("return_success", False):
                return True
            return None

        self.weight_loader = _loader


def _make_mtp_obj_for_load_weights():
    """Create DeepSeekMTP object without running heavy __init__."""
    m = deepseek_mtp_mod.DeepSeekMTP.__new__(deepseek_mtp_mod.DeepSeekMTP)
    m.config = SimpleNamespace(
        num_nextn_predict_layers=2,
        n_group=1,
        n_routed_experts=2,
        n_shared_experts=2,
    )
    m.model = SimpleNamespace(mtp_start_layer_idx=10)
    return m


# ==============================================================================
# Tests: _rewrite_spec_layer_name
# ==============================================================================

class TestRewriteSpecLayerName(unittest.TestCase):
    def test_rewrite_transformer_block_adds_mtp_block(self):
        m = _make_mtp_obj_for_load_weights()
        out = m._rewrite_spec_layer_name(10, "model.layers.10.self_attn.q_proj.weight")
        self.assertEqual(out, "model.layers.10.mtp_block.self_attn.q_proj.weight")

    def test_rewrite_spec_layer_weight_keeps_no_mtp_block(self):
        m = _make_mtp_obj_for_load_weights()
        out = m._rewrite_spec_layer_name(10, "model.layers.10.enorm.weight")
        self.assertEqual(out, "model.layers.10.enorm.weight")

    def test_rewrite_shared_weight_moves_to_top_level(self):
        m = _make_mtp_obj_for_load_weights()
        out = m._rewrite_spec_layer_name(10, "model.layers.10.embed_tokens.weight")
        self.assertEqual(out, "model.embed_tokens.weight")

    def test_rewrite_non_spec_layer_weight_transformer_block(self):
        m = _make_mtp_obj_for_load_weights()
        out = m._rewrite_spec_layer_name(11, "model.layers.11.mlp.down_proj.weight")
        self.assertEqual(out, "model.layers.11.mtp_block.mlp.down_proj.weight")


# ==============================================================================
# Tests: load_weights
# ==============================================================================

class TestLoadWeights(unittest.TestCase):
    def _install_named_params(self, m, names):
        params = [(n, _FakeParam(n)) for n in names]

        def _named_parameters():
            return params

        m.named_parameters = _named_parameters
        return {n: p for n, p in params}

    @patch.object(deepseek_mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    @patch.object(deepseek_mtp_mod, "default_weight_loader", autospec=True)
    @patch.object(deepseek_mtp_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(deepseek_mtp_mod.SharedFusedMoE, "make_expert_params_mapping", autospec=True)
    @patch.object(deepseek_mtp_mod.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", autospec=True)
    def test_load_weights_skips_rotary_and_non_spec(
        self,
        mock_fusion_enabled,
        mock_make_expert_map,
        mock_get_spec_layer,
        mock_default_loader,
        mock_remap,
    ):
        m = _make_mtp_obj_for_load_weights()
        self._install_named_params(m, ["model.embed_tokens.weight"])

        mock_fusion_enabled.return_value = False
        mock_make_expert_map.return_value = []
        mock_get_spec_layer.side_effect = lambda cfg, name: None

        weights = [
            ("model.layers.10.rotary_emb.inv_freq", torch.ones(1)),
            ("model.layers.10.self_attn.q_proj.weight", torch.ones(1)),
        ]
        loaded = m.load_weights(weights)
        self.assertEqual(len(loaded), 0)

    @patch.object(deepseek_mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    @patch.object(deepseek_mtp_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(deepseek_mtp_mod.SharedFusedMoE, "make_expert_params_mapping", autospec=True)
    @patch.object(deepseek_mtp_mod.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", autospec=True)
    def test_load_weights_transformer_block_path_uses_stacked_mapping(
        self,
        mock_fusion_enabled,
        mock_make_expert_map,
        mock_get_spec_layer,
        mock_remap,
    ):
        m = _make_mtp_obj_for_load_weights()
        target_param_name = "model.layers.10.mtp_block.mlp.gate_up_proj.weight"
        params_dict = self._install_named_params(m, [target_param_name])
        target_param = params_dict[target_param_name]

        mock_fusion_enabled.return_value = False
        mock_make_expert_map.return_value = []
        mock_get_spec_layer.side_effect = lambda cfg, name: 10

        weights = [("model.layers.10.mlp.gate_proj.weight", torch.randn(4, 4))]
        loaded = m.load_weights(weights)

        self.assertIn(target_param_name, loaded)
        self.assertEqual(len(target_param.load_calls), 1)
        self.assertEqual(target_param.load_calls[0]["args"][0], 0)

    @patch.object(deepseek_mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    @patch.object(deepseek_mtp_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(deepseek_mtp_mod.SharedFusedMoE, "make_expert_params_mapping", autospec=True)
    @patch.object(deepseek_mtp_mod.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", autospec=True)
    def test_load_weights_shared_embed_tokens_only_loaded_for_first_spec_layer(
        self,
        mock_fusion_enabled,
        mock_make_expert_map,
        mock_get_spec_layer,
        mock_remap,
    ):
        m = _make_mtp_obj_for_load_weights()
        m.model.mtp_start_layer_idx = 10

        param_name = "model.embed_tokens.weight"
        params_dict = self._install_named_params(m, [param_name])
        p = params_dict[param_name]

        mock_fusion_enabled.return_value = False
        mock_make_expert_map.return_value = []

        def _spec_layer(cfg, name):
            if ".10." in name:
                return 10
            if ".11." in name:
                return 11
            return None

        mock_get_spec_layer.side_effect = _spec_layer

        weights = [
            ("model.layers.10.embed_tokens.weight", torch.randn(8, 8)),
            ("model.layers.11.embed_tokens.weight", torch.randn(8, 8)),
        ]
        loaded = m.load_weights(weights)

        self.assertIn(param_name, loaded)
        self.assertEqual(len(p.load_calls), 1)

    @patch.object(deepseek_mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    @patch.object(deepseek_mtp_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(deepseek_mtp_mod.SharedFusedMoE, "make_expert_params_mapping", autospec=True)
    @patch.object(deepseek_mtp_mod.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", autospec=True)
    def test_load_weights_expert_weight_maps_and_loads(
        self,
        mock_fusion_enabled,
        mock_make_expert_map,
        mock_get_spec_layer,
        mock_remap,
    ):
        m = _make_mtp_obj_for_load_weights()

        # IMPORTANT: in some branches, name might stay as gate_proj
        # (depending on how the code breaks / continues). Provide both.
        gate_proj_name = "model.layers.10.mtp_block.mlp.experts.0.gate_proj.weight"
        gate_up_name = "model.layers.10.mtp_block.mlp.experts.0.gate_up_proj.weight"
        params = self._install_named_params(m, [gate_proj_name, gate_up_name])

        mock_fusion_enabled.return_value = False
        mock_make_expert_map.return_value = []
        mock_get_spec_layer.side_effect = lambda cfg, name: 10

        weights = [
            ("model.layers.10.mlp.experts.0.gate_proj.weight", torch.randn(4, 4)),
        ]
        loaded = m.load_weights(weights)

        self.assertTrue(len(loaded) >= 1)
        self.assertTrue(
            (len(params[gate_proj_name].load_calls) + len(params[gate_up_name].load_calls)) >= 1
        )

    @patch.object(deepseek_mtp_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    @patch.object(deepseek_mtp_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(deepseek_mtp_mod.SharedFusedMoE, "make_expert_params_mapping", autospec=True)
    @patch.object(deepseek_mtp_mod.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", autospec=True)
    def test_load_weights_fusion_shared_experts_splits_and_loads_chunks(
        self,
        mock_fusion_enabled,
        mock_make_expert_map,
        mock_get_spec_layer,
        mock_remap,
    ):
        m = _make_mtp_obj_for_load_weights()
        m.model.mtp_start_layer_idx = 10
        m.config.n_routed_experts = 2
        m.config.n_shared_experts = 2

        mock_fusion_enabled.return_value = True

        expert_map = [
            ("gate_up_proj", "gate_proj", 2, 0),
            ("gate_up_proj", "gate_proj", 3, 0),
        ]
        mock_make_expert_map.return_value = expert_map
        mock_get_spec_layer.side_effect = lambda cfg, name: 10

        p2_name = "model.layers.10.mtp_block.mlp.experts.2.gate_up_proj.weight"
        p3_name = "model.layers.10.mtp_block.mlp.experts.3.gate_up_proj.weight"
        params_dict = self._install_named_params(m, [p2_name, p3_name])
        p2 = params_dict[p2_name]
        p3 = params_dict[p3_name]

        loaded_weight = torch.randn(10, 4)  # split into 5 and 5
        weights = [("model.layers.10.mlp.shared_experts.gate_proj.weight", loaded_weight)]
        loaded = m.load_weights(weights)

        self.assertEqual(len(p2.load_calls), 1)
        self.assertEqual(len(p3.load_calls), 1)
        self.assertEqual(p2.load_calls[0]["weight"].shape[0], 5)
        self.assertEqual(p3.load_calls[0]["weight"].shape[0], 5)
        self.assertIn(p2_name, loaded)
        self.assertIn(p3_name, loaded)


# ==============================================================================
# NEW: Coverage for DeepSeekMultiTokenPredictorLayer / Predictor / MTP
# ==============================================================================

def _make_fake_vllm_config():
    hf = SimpleNamespace(
        hidden_size=8,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_hidden_layers=10,
        num_nextn_predict_layers=2,
        n_group=1,
        n_routed_experts=2,
        n_shared_experts=2,
    )

    # support_torch_compile wrapper touches:
    #   self.vllm_config.compilation_config
    #   self.compilation_config.mode
    compilation_config = SimpleNamespace(mode="DISABLED")

    vllm = SimpleNamespace(
        quant_config=None,
        compilation_config=compilation_config,
        model_config=SimpleNamespace(hf_config=hf),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        ),
    )
    return vllm


class TestDeepSeekMultiTokenPredictorLayerInitForward(unittest.TestCase):
    @patch.object(deepseek_mtp_mod, "ParallelLMHead", _FakeParallelLMHead)
    @patch.object(deepseek_mtp_mod, "RMSNorm", _FakeRMSNorm)
    @patch.object(deepseek_mtp_mod, "RowParallelFlashCommLinear", _FakeRowParallelFlashCommLinear)
    def test_layer_init_and_forward(self):
        vllm_config = _make_fake_vllm_config()

        def _decoder_ctor(*args, **kwargs):
            return _FakeDecoderLayer(mlp=nn.Linear(1, 1))

        with patch.object(deepseek_mtp_mod, "DeepseekV2DecoderLayer", side_effect=_decoder_ctor), \
             patch.object(deepseek_mtp_mod, "maybe_prefix", side_effect=lambda p, s: f"{p}.{s}"):
            layer = deepseek_mtp_mod.DeepSeekMultiTokenPredictorLayer(
                vllm_config=vllm_config,
                prefix="model.layers.10"
            )

        self.assertTrue(hasattr(layer, "enorm"))
        self.assertTrue(hasattr(layer, "hnorm"))
        self.assertTrue(hasattr(layer, "eh_proj"))
        self.assertTrue(hasattr(layer, "shared_head"))
        self.assertTrue(hasattr(layer, "mtp_block"))

        B = 3
        input_ids = torch.zeros((B,), dtype=torch.int64)
        positions = torch.arange(B, dtype=torch.int64)
        prev_h = torch.randn((B, vllm_config.model_config.hf_config.hidden_size), dtype=torch.float32)
        embeds = torch.randn((B, vllm_config.model_config.hf_config.hidden_size), dtype=torch.float32)

        out = layer.forward(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=prev_h,
            inputs_embeds=embeds,
            spec_step_index=0,
        )
        self.assertEqual(tuple(out.shape), (B, vllm_config.model_config.hf_config.hidden_size))


class TestDeepSeekMultiTokenPredictorInitForwardLogits(unittest.TestCase):
    @patch.object(deepseek_mtp_mod, "ParallelLMHead", _FakeParallelLMHead)
    @patch.object(deepseek_mtp_mod, "RMSNorm", _FakeRMSNorm)
    @patch.object(deepseek_mtp_mod, "RowParallelFlashCommLinear", _FakeRowParallelFlashCommLinear)
    @patch.object(deepseek_mtp_mod, "VocabParallelEmbedding", _FakeVocabParallelEmbedding)
    @patch.object(deepseek_mtp_mod, "LogitsProcessor", _FakeLogitsProcessor)
    def test_predictor_init_forward_and_compute_logits(self):
        vllm_config = _make_fake_vllm_config()

        def _decoder_ctor(*args, **kwargs):
            return _FakeDecoderLayer(mlp=nn.Linear(1, 1))

        with patch.object(deepseek_mtp_mod, "DeepseekV2DecoderLayer", side_effect=_decoder_ctor), \
             patch.object(deepseek_mtp_mod, "maybe_prefix", side_effect=lambda p, s: f"{p}.{s}"):
            pred = deepseek_mtp_mod.DeepSeekMultiTokenPredictor(
                vllm_config=vllm_config, prefix="model"
            )

        self.assertEqual(pred.mtp_start_layer_idx, vllm_config.model_config.hf_config.num_hidden_layers)
        self.assertEqual(pred.num_mtp_layers, vllm_config.model_config.hf_config.num_nextn_predict_layers)
        self.assertEqual(len(pred.layers), pred.num_mtp_layers)

        B = 4
        input_ids = torch.zeros((B,), dtype=torch.int64)
        positions = torch.arange(B, dtype=torch.int64)
        prev_h = torch.randn((B, vllm_config.model_config.hf_config.hidden_size), dtype=torch.float32)

        out = pred.forward(
            input_ids=input_ids,
            positions=positions,
            previous_hidden_states=prev_h,
            inputs_embeds=None,  # hit inputs_embeds is None branch
            spec_step_idx=3,     # hit modulo
        )
        self.assertEqual(tuple(out.shape), (B, vllm_config.model_config.hf_config.hidden_size))
        self.assertGreater(pred.embed_tokens.calls, 0)

        logits = pred.compute_logits(out, spec_step_idx=5)
        self.assertEqual(tuple(logits.shape), (B, vllm_config.model_config.hf_config.vocab_size))
        self.assertGreater(len(pred.logits_processor.calls), 0)


class TestDeepSeekMTPInitSetMoEForward(unittest.TestCase):
    @patch.object(deepseek_mtp_mod, "ParallelLMHead", _FakeParallelLMHead)
    @patch.object(deepseek_mtp_mod, "RMSNorm", _FakeRMSNorm)
    @patch.object(deepseek_mtp_mod, "RowParallelFlashCommLinear", _FakeRowParallelFlashCommLinear)
    @patch.object(deepseek_mtp_mod, "VocabParallelEmbedding", _FakeVocabParallelEmbedding)
    @patch.object(deepseek_mtp_mod, "LogitsProcessor", _FakeLogitsProcessor)
    def test_mtp_init_set_moe_and_forward(self):
        vllm_config = _make_fake_vllm_config()

        # Need real "type" for isinstance checks
        class _PatchedDeepseekV2MoE(_FakeMoE):
            pass

        class _PatchedDeepseekV2DecoderLayer(nn.Module):
            _count = 0

            def __init__(self, *args, **kwargs):
                super().__init__()
                self.self_attn = _FakeSelfAttn()
                if _PatchedDeepseekV2DecoderLayer._count == 0:
                    self.mlp = _PatchedDeepseekV2MoE()  # hit MoE branch
                else:
                    self.mlp = nn.Linear(1, 1)          # non-MoE branch
                _PatchedDeepseekV2DecoderLayer._count += 1

            def forward(self, hidden_states, cos, sin, residual=None):
                out = hidden_states + 2.0
                residual_out = torch.ones_like(out)
                return out, residual_out

        with patch.object(deepseek_mtp_mod, "DeepseekV2MoE", _PatchedDeepseekV2MoE), \
             patch.object(deepseek_mtp_mod, "DeepseekV2DecoderLayer", _PatchedDeepseekV2DecoderLayer), \
             patch.object(deepseek_mtp_mod, "maybe_prefix", side_effect=lambda p, s: f"{p}.{s}"), \
             patch.object(deepseek_mtp_mod.DeepSeekMTP, "extract_moe_parameters", MagicMock()) as mock_extract:

            mtp = deepseek_mtp_mod.DeepSeekMTP(vllm_config=vllm_config, prefix="")

            # set_moe_parameters executed in __init__
            self.assertTrue(hasattr(mtp, "moe_layers"))
            self.assertTrue(hasattr(mtp, "moe_mlp_layers"))
            self.assertGreaterEqual(len(mtp.moe_layers), 1)
            self.assertGreaterEqual(len(mtp.moe_mlp_layers), 1)
            self.assertTrue(mock_extract.called)

            # FIX: cannot assign MagicMock to nn.Module child module
            fake_out = torch.ones((2, vllm_config.model_config.hf_config.hidden_size), dtype=torch.float32)
            mtp.model = _FakeModelReturn(fake_out)

            input_ids = torch.zeros((2,), dtype=torch.int64)
            positions = torch.arange(2, dtype=torch.int64)
            hidden_states = torch.zeros((2, vllm_config.model_config.hf_config.hidden_size), dtype=torch.float32)

            out = mtp.forward(
                input_ids, positions, hidden_states,
                intermediate_tensors=None,
                inputs_embeds=None,
                spec_step_idx=0
            )

            self.assertEqual(tuple(out.shape), (2, vllm_config.model_config.hf_config.hidden_size))
            self.assertEqual(mtp.model.calls, 1)
            self.assertTrue(torch.equal(out, fake_out))


if __name__ == "__main__":
    unittest.main()