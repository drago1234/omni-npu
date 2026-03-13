import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import omni_npu.v1.models.deepseek.deepseek_v3 as deepseek_v3_mod


class _FakeParam:
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


class _FakeExperts:
    def __init__(self, is_internal_router=False, ret_shared=None):
        self.is_internal_router = is_internal_router
        self.ret_shared = ret_shared
        self.calls = []

    def __call__(self, hidden_states, router_logits):
        self.calls.append((hidden_states, router_logits))
        return self.ret_shared, hidden_states + 2.0

    def update_expert_map(self):
        self.updated = True


class _FakeGate:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden_states):
        self.calls += 1
        logits = torch.zeros((hidden_states.shape[0], 3), dtype=hidden_states.dtype)
        return logits, None


class _FakeNorm:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, residual=None):
        self.calls.append((hidden_states, residual))
        if residual is None:
            return hidden_states + 1.0
        return hidden_states + 1.0, residual + 1.0


class _FakeAttn:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, cos, sin):
        self.calls.append((hidden_states, cos, sin))
        return hidden_states + 3.0


class _FakeMLP:
    def __init__(self):
        self.calls = 0

    def __call__(self, hidden_states):
        self.calls += 1
        return hidden_states + 4.0


class TestSpecLayerUtils(unittest.TestCase):
    def test_get_spec_layer_idx_hit(self):
        cfg = SimpleNamespace(num_hidden_layers=10, num_nextn_predict_layers=2)
        got = deepseek_v3_mod.get_spec_layer_idx_from_weight_name(
            cfg, "model.layers.11.self_attn.q_proj.weight"
        )
        self.assertEqual(got, 11)

    def test_get_spec_layer_idx_miss_or_disabled(self):
        cfg = SimpleNamespace(num_hidden_layers=10, num_nextn_predict_layers=0)
        got = deepseek_v3_mod.get_spec_layer_idx_from_weight_name(
            cfg, "model.layers.10.self_attn.q_proj.weight"
        )
        self.assertIsNone(got)


class TestMoEMixins(unittest.TestCase):
    def test_extract_moe_parameters_none(self):
        m = deepseek_v3_mod.DeepseekV2MixtureOfExperts()
        with patch.object(deepseek_v3_mod.logger, "warning") as mock_warn:
            m.extract_moe_parameters(None)
        self.assertEqual(m.num_moe_layers, 0)
        self.assertEqual(m.num_redundant_experts, 0)
        self.assertTrue(mock_warn.called)

    def test_extract_and_update_physical_experts_metadata(self):
        m = deepseek_v3_mod.DeepseekV2MixtureOfExperts()
        ex = SimpleNamespace(
            n_logical_experts=8,
            n_physical_experts=10,
            n_local_physical_experts=5,
            n_routed_experts=8,
            n_shared_experts=2,
            n_redundant_experts=2,
        )
        m.extract_moe_parameters(ex)
        self.assertEqual(m.num_physical_experts, 10)
        self.assertEqual(m.num_local_physical_experts, 5)

        l1 = SimpleNamespace(
            n_local_physical_experts=5,
            n_physical_experts=10,
            n_redundant_experts=2,
            experts=_FakeExperts(),
        )
        l2 = SimpleNamespace(
            n_local_physical_experts=5,
            n_physical_experts=10,
            n_redundant_experts=2,
            experts=_FakeExperts(),
        )
        m.moe_mlp_layers = [l1, l2]

        m.update_physical_experts_metadata(12, 5)
        self.assertEqual(m.num_redundant_experts, 4)
        self.assertEqual(l1.n_physical_experts, 12)
        self.assertEqual(l2.n_redundant_experts, 4)
        self.assertTrue(getattr(l1.experts, "updated", False))
        self.assertTrue(getattr(l2.experts, "updated", False))


class TestDeepseekV2MoEMethods(unittest.TestCase):
    def test_bind_prefetch_moe_to_experts(self):
        m = deepseek_v3_mod.DeepseekV2MoE.__new__(deepseek_v3_mod.DeepseekV2MoE)
        m.prefetch_moe = deepseek_v3_mod.DeepseekV2MoE.prefetch_moe.__get__(
            m, deepseek_v3_mod.DeepseekV2MoE
        )
        m.experts = SimpleNamespace()

        with patch.object(
            deepseek_v3_mod.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(enable_prefetch=True),
        ):
            m._bind_prefetch_moe_to_experts()

        self.assertTrue(hasattr(m.experts, "prefetch_moe"))

    def test_prefetch_moe_calls_prefetch_weight(self):
        m = deepseek_v3_mod.DeepseekV2MoE.__new__(deepseek_v3_mod.DeepseekV2MoE)
        m.min_prefetch_size = 1
        m.prefetch_weight = MagicMock()
        m.w13_weight = torch.ones(1)
        m.w2_weight = torch.ones(1)
        m.shared_experts = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=torch.ones(1)),
            down_proj=SimpleNamespace(weight=torch.ones(1)),
        )
        trig = torch.ones(1)

        m.prefetch_moe(trig, prefetch_experts=True, prefetch_shared_experts=True)
        self.assertGreaterEqual(m.prefetch_weight.call_count, 4)

    def test_forward_with_and_without_sequence_parallel(self):
        m = deepseek_v3_mod.DeepseekV2MoE.__new__(deepseek_v3_mod.DeepseekV2MoE)
        m.is_sequence_parallel = False
        m.experts = _FakeExperts(is_internal_router=False, ret_shared=None)
        m.gate = _FakeGate()
        m.shared_experts = None
        x = torch.ones((4, 8))
        y = m.forward(x)
        self.assertEqual(tuple(y.shape), (4, 8))
        self.assertEqual(m.gate.calls, 1)

        m2 = deepseek_v3_mod.DeepseekV2MoE.__new__(deepseek_v3_mod.DeepseekV2MoE)
        m2.is_sequence_parallel = True
        m2.experts = _FakeExperts(is_internal_router=True, ret_shared=torch.ones((4, 8)))
        m2.gate = _FakeGate()
        m2.shared_experts = object()
        with patch.object(
            deepseek_v3_mod, "sequence_parallel_chunk", side_effect=lambda t: t
        ), patch.object(
            deepseek_v3_mod, "tensor_model_parallel_all_gather", side_effect=lambda t, _: t
        ):
            y2 = m2.forward(torch.ones((4, 8)))
        self.assertEqual(tuple(y2.shape), (4, 8))


class TestDecoderLayerForward(unittest.TestCase):
    def test_decoder_forward_residual_branches(self):
        d = deepseek_v3_mod.DeepseekV2DecoderLayer.__new__(deepseek_v3_mod.DeepseekV2DecoderLayer)
        d.input_layernorm = _FakeNorm()
        d.post_attention_layernorm = _FakeNorm()
        d.self_attn = _FakeAttn()
        d.mlp = _FakeMLP()

        h = torch.ones((2, 8))
        cos = torch.zeros((2, 1, 1, 4))
        sin = torch.zeros((2, 1, 1, 4))

        out1, res1 = d.forward(h, cos, sin, residual=None)
        self.assertEqual(tuple(out1.shape), (2, 8))
        self.assertEqual(tuple(res1.shape), (2, 8))

        out2, res2 = d.forward(h, cos, sin, residual=torch.zeros((2, 8)))
        self.assertEqual(tuple(out2.shape), (2, 8))
        self.assertEqual(tuple(res2.shape), (2, 8))


def _make_model_obj_for_load_weights():
    m = deepseek_v3_mod.DeepseekV2ForCausalLM.__new__(deepseek_v3_mod.DeepseekV2ForCausalLM)
    m.config = SimpleNamespace(
        n_routed_experts=2,
        n_shared_experts=2,
        num_hidden_layers=10,
        num_nextn_predict_layers=2,
    )
    m.num_redundant_experts = 0
    return m


class TestDeepseekV2ForCausalLMBasicMethods(unittest.TestCase):
    def test_embed_forward_compute_logits_get_mapping(self):
        m = _make_model_obj_for_load_weights()
        model_out = torch.randn((2, 8))

        m.model = MagicMock()
        m.model.embed_input_ids.return_value = torch.randn((2, 8))
        m.model.return_value = model_out
        m.lm_head = object()
        m.logits_processor = MagicMock(return_value=torch.randn((2, 16)))

        ids = torch.zeros((2,), dtype=torch.long)
        pos = torch.arange(2, dtype=torch.long)
        got_emb = m.embed_input_ids(ids)
        got_h = m.forward(ids, pos)
        got_logits = m.compute_logits(got_h)
        self.assertEqual(tuple(got_emb.shape), (2, 8))
        self.assertTrue(torch.equal(got_h, model_out))
        self.assertEqual(tuple(got_logits.shape), (2, 16))

        with patch.object(
            deepseek_v3_mod.SharedFusedMoE,
            "make_expert_params_mapping",
            return_value=[("a", "b", 0, 0)],
        ):
            mapping = m.get_expert_mapping()
        self.assertEqual(mapping, [("a", "b", 0, 0)])

    def test_set_moe_parameters_collects_layers(self):
        m = _make_model_obj_for_load_weights()
        m.config.n_group = 1

        class _FakePPMissing:
            pass

        class _FakeMoEType:
            def __init__(self):
                self.experts = object()

        class _FakeLayer:
            def __init__(self, mlp):
                self.mlp = mlp

        with patch.object(deepseek_v3_mod, "PPMissingLayer", _FakePPMissing), patch.object(
            deepseek_v3_mod, "DeepseekV2DecoderLayer", _FakeLayer
        ), patch.object(deepseek_v3_mod, "DeepseekV2MoE", _FakeMoEType):
            l0 = _FakePPMissing()
            l1 = _FakeLayer(mlp=_FakeMoEType())
            l2 = _FakeLayer(mlp=nn.Linear(1, 1))
            m.model = SimpleNamespace(layers=[l0, l1, l2])
            with patch.object(
                deepseek_v3_mod.DeepseekV2ForCausalLM,
                "extract_moe_parameters",
                autospec=True,
            ) as mock_extract:
                m.set_moe_parameters()

        self.assertEqual(m.num_expert_groups, 1)
        self.assertEqual(len(m.moe_layers), 1)
        self.assertEqual(len(m.moe_mlp_layers), 1)
        self.assertTrue(mock_extract.called)


class TestDeepseekV2ForCausalLMLoadWeights(unittest.TestCase):
    def _install_named_params(self, m, names):
        params = [(n, _FakeParam(n)) for n in names]

        def _named_parameters():
            return params

        m.named_parameters = _named_parameters
        return {n: p for n, p in params}

    @patch.object(deepseek_v3_mod, "is_pp_missing_parameter", return_value=False)
    @patch.object(
        deepseek_v3_mod.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        return_value=False,
    )
    @patch.object(deepseek_v3_mod, "get_spec_layer_idx_from_weight_name")
    @patch.object(
        deepseek_v3_mod.SharedFusedMoE,
        "make_expert_params_mapping",
        return_value=[],
    )
    def test_load_weights_skips_rotary_and_spec_layers(
        self, _m0, mock_get_spec, _m2, _m3
    ):
        m = _make_model_obj_for_load_weights()
        self._install_named_params(m, ["model.embed_tokens.weight"])

        def _spec(_cfg, name):
            if ".10." in name:
                return 10
            return None

        mock_get_spec.side_effect = _spec
        weights = [
            ("model.layers.1.rotary_emb.inv_freq", torch.ones(1)),
            ("model.layers.10.self_attn.q_proj.weight", torch.ones(1)),
        ]
        loaded = m.load_weights(weights)
        self.assertEqual(len(loaded), 0)

    @patch.object(deepseek_v3_mod, "is_pp_missing_parameter", return_value=False)
    @patch.object(
        deepseek_v3_mod.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        return_value=False,
    )
    @patch.object(deepseek_v3_mod, "get_spec_layer_idx_from_weight_name", return_value=None)
    @patch.object(
        deepseek_v3_mod.SharedFusedMoE,
        "make_expert_params_mapping",
        return_value=[],
    )
    def test_load_weights_stacked_mapping_path(self, _m0, _m1, _m2, _m3):
        m = _make_model_obj_for_load_weights()
        p_name = "model.layers.1.mlp.gate_up_proj.weight"
        p = self._install_named_params(m, [p_name])[p_name]

        loaded = m.load_weights([("model.layers.1.mlp.gate_proj.weight", torch.randn(4, 4))])
        self.assertIn(p_name, loaded)
        self.assertEqual(len(p.load_calls), 1)
        self.assertEqual(p.load_calls[0]["args"][0], 0)

    @patch.object(deepseek_v3_mod, "is_pp_missing_parameter", return_value=False)
    @patch.object(
        deepseek_v3_mod.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        return_value=True,
    )
    @patch.object(deepseek_v3_mod, "get_spec_layer_idx_from_weight_name", return_value=None)
    @patch.object(
        deepseek_v3_mod.SharedFusedMoE,
        "make_expert_params_mapping",
        return_value=[
            ("gate_up_proj", "gate_proj", 2, 0),
            ("gate_up_proj", "gate_proj", 3, 0),
        ],
    )
    def test_load_weights_fusion_shared_experts_split(self, _m0, _m1, _m2, _m3):
        m = _make_model_obj_for_load_weights()
        p2 = "model.layers.1.mlp.experts.2.gate_up_proj.weight"
        p3 = "model.layers.1.mlp.experts.3.gate_up_proj.weight"
        params = self._install_named_params(m, [p2, p3])

        w = torch.randn(10, 4)
        loaded = m.load_weights([("model.layers.1.mlp.shared_experts.gate_proj.weight", w)])
        self.assertIn(p2, loaded)
        self.assertIn(p3, loaded)
        self.assertEqual(len(params[p2].load_calls), 1)
        self.assertEqual(len(params[p3].load_calls), 1)
        self.assertEqual(params[p2].load_calls[0]["weight"].shape[0], 5)
        self.assertEqual(params[p3].load_calls[0]["weight"].shape[0], 5)

    @patch.object(deepseek_v3_mod, "is_pp_missing_parameter", return_value=False)
    @patch.object(
        deepseek_v3_mod.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        return_value=False,
    )
    @patch.object(deepseek_v3_mod, "get_spec_layer_idx_from_weight_name", return_value=None)
    @patch.object(
        deepseek_v3_mod.SharedFusedMoE,
        "make_expert_params_mapping",
        return_value=[],
    )
    @patch.object(deepseek_v3_mod, "default_weight_loader", autospec=True)
    @patch.object(deepseek_v3_mod, "maybe_remap_kv_scale_name", side_effect=lambda n, _: n)
    def test_load_weights_fallback_default_loader(
        self, _m0, mock_default_loader, _m2, _m3, _m4, _m5
    ):
        m = _make_model_obj_for_load_weights()
        param_name = "model.some.weight"
        p = self._install_named_params(m, [param_name])[param_name]
        delattr(p, "weight_loader")

        loaded = m.load_weights([(param_name, torch.randn(2, 2))])
        self.assertIn(param_name, loaded)
        self.assertEqual(mock_default_loader.call_count, 1)


class _FakeEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, quant_config=None, prefix=""):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.prefix = prefix

    def forward(self, input_ids):
        return torch.zeros((input_ids.shape[0], self.hidden_size))


class _FakeRotaryEmb:
    def get_cos_sin(self, positions):
        bsz = positions.numel()
        cos = torch.zeros((bsz, 1, 1, 4))
        sin = torch.zeros((bsz, 1, 1, 4))
        return cos, sin


class _FakeDecoderForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SimpleNamespace(rotary_emb=_FakeRotaryEmb())
        self.is_moe = False

    def forward(self, hidden_states, cos, sin, residual):
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states + 1.0, residual + 1.0


class _FakeMoEExpertsForPrefetch:
    def __init__(self):
        self.ep_size = 128
        self.quant_config = True
        self.shared_experts = object()


class _FakePPGroup:
    def __init__(self, is_first_rank, is_last_rank):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank


class TestInitHeavyPaths(unittest.TestCase):
    def test_moe_init_success_and_invalid_activation(self):
        class _FakeReplicatedLinear(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.e_score_correction_bias = None

        class _FakeDeepseekV2MLP(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.gate_up_proj = SimpleNamespace(weight=torch.ones(1))
                self.down_proj = SimpleNamespace(weight=torch.ones(1))

        class _FakeNPUFusedMoE(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.kwargs = kwargs

        fake_ep_group = SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 2), rank_in_group=0
        )
        cfg = SimpleNamespace(
            hidden_act="silu",
            n_routed_experts=8,
            n_shared_experts=2,
            hidden_size=16,
            moe_intermediate_size=32,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
        )
        parallel_cfg = SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        )

        with patch.object(deepseek_v3_mod, "ReplicatedLinear", _FakeReplicatedLinear), patch.object(
            deepseek_v3_mod, "DeepseekV2MLP", _FakeDeepseekV2MLP
        ), patch.object(
            deepseek_v3_mod, "NPUSharedFusedMoE", _FakeNPUFusedMoE
        ), patch.object(
            deepseek_v3_mod, "get_tensor_model_parallel_world_size", return_value=1
        ), patch.object(
            deepseek_v3_mod, "get_tensor_model_parallel_rank", return_value=0
        ), patch.object(
            deepseek_v3_mod, "get_ep_group", return_value=fake_ep_group
        ), patch.object(
            deepseek_v3_mod.rocm_aiter_ops, "is_fused_moe_enabled", return_value=False
        ), patch.object(
            deepseek_v3_mod.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            return_value=False,
        ), patch.object(
            deepseek_v3_mod.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(enable_prefetch=False),
        ):
            moe = deepseek_v3_mod.DeepseekV2MoE(
                config=cfg, parallel_config=parallel_cfg, quant_config=None, prefix="m"
            )
        self.assertIsNotNone(moe.shared_experts)
        self.assertEqual(moe.n_physical_experts, 8)

        bad_cfg = SimpleNamespace(
            hidden_act="gelu",
            n_routed_experts=8,
            n_shared_experts=2,
        )
        with patch.object(
            deepseek_v3_mod, "get_tensor_model_parallel_world_size", return_value=1
        ), patch.object(
            deepseek_v3_mod, "get_tensor_model_parallel_rank", return_value=0
        ), patch.object(
            deepseek_v3_mod, "get_ep_group", return_value=fake_ep_group
        ), self.assertRaises(ValueError):
            deepseek_v3_mod.DeepseekV2MoE(
                config=bad_cfg, parallel_config=parallel_cfg, quant_config=None, prefix="m"
            )

    def test_decoder_layer_init_sparse_and_dense(self):
        class _FakeSparseAttn:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

        class _FakeMlaAttn:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

        class _FakeMoEType(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class _FakeMLPType(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        vllm_cfg = SimpleNamespace(
            cache_config=object(),
            quant_config=None,
            parallel_config=SimpleNamespace(
                use_sequence_parallel_moe=False,
                enable_eplb=False,
                eplb_config=SimpleNamespace(num_redundant_experts=0),
            ),
            model_config=SimpleNamespace(hf_config=None),
        )
        cfg_sparse_moe = SimpleNamespace(
            hidden_size=8,
            max_position_embeddings=1024,
            moe_layer_freq=1,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            v_head_dim=1,
            kv_lora_rank=1,
            num_attention_heads=1,
            q_lora_rank=1,
            n_routed_experts=2,
            first_k_dense_replace=0,
            intermediate_size=16,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            index_topk=8,
        )
        cfg_dense = SimpleNamespace(
            hidden_size=8,
            max_position_embeddings=1024,
            moe_layer_freq=1,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            v_head_dim=1,
            kv_lora_rank=1,
            num_attention_heads=1,
            n_routed_experts=None,
            first_k_dense_replace=99,
            intermediate_size=16,
            hidden_act="silu",
            rms_norm_eps=1e-6,
        )

        with patch.object(deepseek_v3_mod, "NPUDeepseekSparseAttention", _FakeSparseAttn), patch.object(
            deepseek_v3_mod, "NPUDeepseekMLAAttention", _FakeMlaAttn
        ), patch.object(deepseek_v3_mod, "DeepseekV2MoE", _FakeMoEType), patch.object(
            deepseek_v3_mod, "DeepseekV2MLP", _FakeMLPType
        ), patch.object(deepseek_v3_mod, "RMSNorm", side_effect=lambda *a, **k: _FakeNorm()):
            d1 = deepseek_v3_mod.DeepseekV2DecoderLayer(
                vllm_config=vllm_cfg, prefix="model.layers.2", config=cfg_sparse_moe
            )
            d2 = deepseek_v3_mod.DeepseekV2DecoderLayer(
                vllm_config=vllm_cfg, prefix="model.layers.1", config=cfg_dense
            )
        self.assertTrue(d1.is_moe)
        self.assertFalse(d2.is_moe)

    def test_model_init_forward_and_prefetch(self):
        fake_cfg = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
        )
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=fake_cfg),
            quant_config=None,
            compilation_config=SimpleNamespace(mode="DISABLED"),
        )
        fake_layers = [_FakeDecoderForwardLayer(), _FakeDecoderForwardLayer()]

        with patch.object(
            deepseek_v3_mod, "get_pp_group", return_value=_FakePPGroup(True, True)
        ), patch.object(
            deepseek_v3_mod, "VocabParallelEmbedding", _FakeEmbedding
        ), patch.object(
            deepseek_v3_mod, "make_layers", return_value=(0, 2, fake_layers)
        ), patch.object(
            deepseek_v3_mod, "RMSNorm", side_effect=lambda *a, **k: _FakeNorm()
        ), patch.object(
            deepseek_v3_mod, "make_empty_intermediate_tensors_factory", return_value=lambda: {}
        ):
            m = deepseek_v3_mod.DeepseekV2Model(vllm_config=vllm_cfg, prefix="m")

        ids = torch.zeros((3,), dtype=torch.long)
        pos = torch.arange(3, dtype=torch.long)
        with patch.object(
            deepseek_v3_mod, "get_pp_group", return_value=_FakePPGroup(True, True)
        ):
            out = m.forward(ids, pos, intermediate_tensors=None, inputs_embeds=None)
        self.assertEqual(tuple(out.shape), (3, 8))

        selected = _FakeMoEExpertsForPrefetch()
        selected.prefetch_tensors_map = {}
        layer0 = SimpleNamespace(is_moe=True, mlp=SimpleNamespace(experts=selected))
        next_attn = SimpleNamespace(
            q_a_proj=SimpleNamespace(weight=torch.ones(1)),
            q_b_proj=SimpleNamespace(weight=torch.ones(1)),
            attn=SimpleNamespace(impl=SimpleNamespace(W_UK_T=torch.ones(1))),
            kv_a_proj_with_mqa=SimpleNamespace(weight=torch.ones(1)),
        )
        layer1 = SimpleNamespace(is_moe=False, self_attn=next_attn)
        m.start_layer = 0
        m.end_layer = 2
        m.layers = [layer0, layer1]
        with patch.object(
            deepseek_v3_mod.model_extra_config,
            "operator_opt_config",
            SimpleNamespace(
                enable_prefetch=True,
                attn_prefetch=10,
                expert_down_prefetch=11,
                expert_gate_up_prefetch=12,
                shared_expert_down_prefetch=13,
                shared_expert_gate_up_prefetch=14,
            ),
        ):
            m.prefetch_post_load()
        self.assertIn("q_a_proj_weight", selected.prefetch_tensors_map)

    def test_model_forward_non_first_rank_returns_intermediate(self):
        m = deepseek_v3_mod.DeepseekV2Model.__new__(deepseek_v3_mod.DeepseekV2Model)
        m.start_layer = 0
        m.end_layer = 1
        m.layers = [_FakeDecoderForwardLayer()]
        m.norm = _FakeNorm()

        inter = {"hidden_states": torch.ones((2, 8)), "residual": torch.zeros((2, 8))}
        with patch.object(
            deepseek_v3_mod, "get_pp_group", return_value=_FakePPGroup(False, False)
        ), patch.object(
            deepseek_v3_mod, "IntermediateTensors", side_effect=lambda d: d
        ):
            out = m.forward(
                input_ids=torch.zeros((2,), dtype=torch.long),
                positions=torch.arange(2, dtype=torch.long),
                intermediate_tensors=inter,
                inputs_embeds=None,
            )
        self.assertIn("hidden_states", out)
        self.assertIn("residual", out)

    def test_for_causal_lm_init_last_rank_and_not_last_rank(self):
        class _FakeModelCls(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.layers = []
                self.make_empty_intermediate_tensors = lambda: {}

            def forward(self, *args, **kwargs):
                return torch.ones((2, 8))

            def embed_input_ids(self, ids):
                return torch.ones((ids.shape[0], 8))

        class _FakeLMHead(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class _FakeLogitsProc:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def __call__(self, lm_head, hidden_states):
                return torch.zeros((hidden_states.shape[0], self.vocab_size))

        cfg = SimpleNamespace(
            num_hidden_layers=4,
            first_k_dense_replace=1,
            vocab_size=64,
            hidden_size=8,
        )
        vllm_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=cfg), quant_config=None
        )

        with patch.object(deepseek_v3_mod.DeepseekV2ForCausalLM, "model_cls", _FakeModelCls), patch.object(
            deepseek_v3_mod, "maybe_prefix", side_effect=lambda p, s: s
        ), patch.object(
            deepseek_v3_mod, "ParallelLMHead", _FakeLMHead
        ), patch.object(
            deepseek_v3_mod, "LogitsProcessor", _FakeLogitsProc
        ), patch.object(
            deepseek_v3_mod, "get_pp_group", return_value=_FakePPGroup(True, True)
        ), patch.object(
            deepseek_v3_mod.DeepseekV2ForCausalLM, "set_moe_parameters", autospec=True
        ):
            lm_last = deepseek_v3_mod.DeepseekV2ForCausalLM(vllm_config=vllm_cfg, prefix="")
        self.assertEqual(lm_last.num_moe_layers, 3)

        with patch.object(deepseek_v3_mod.DeepseekV2ForCausalLM, "model_cls", _FakeModelCls), patch.object(
            deepseek_v3_mod, "maybe_prefix", side_effect=lambda p, s: s
        ), patch.object(
            deepseek_v3_mod, "ParallelLMHead", _FakeLMHead
        ), patch.object(
            deepseek_v3_mod, "LogitsProcessor", _FakeLogitsProc
        ), patch.object(
            deepseek_v3_mod, "get_pp_group", return_value=_FakePPGroup(True, False)
        ), patch.object(
            deepseek_v3_mod.DeepseekV2ForCausalLM, "set_moe_parameters", autospec=True
        ):
            lm_non_last = deepseek_v3_mod.DeepseekV2ForCausalLM(vllm_config=vllm_cfg, prefix="")
        self.assertIsInstance(lm_non_last.lm_head, deepseek_v3_mod.PPMissingLayer)


if __name__ == "__main__":
    unittest.main()
