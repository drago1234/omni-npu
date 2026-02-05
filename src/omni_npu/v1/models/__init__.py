from vllm import ModelRegistry


def register_models():
    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni_npu.v1.models.deepseek.deepseek_mtp:DeepSeekMTP")
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "omni_npu.v1.models.qwen.qwen3:Qwen3ForCausalLM")
    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "omni_npu.v1.models.qwen.qwen3_vl:Qwen3VLForConditionalGeneration")
    ModelRegistry.register_model(
        "PanguUltraMoEForCausalLM",
        "omni_npu.v1.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM")
    ModelRegistry.register_model(
        "OpenPanguMTPModel",
        "omni_npu.v1.models.pangu.pangu_ultra_moe_mtp:OpenPanguMTP")
    ModelRegistry.register_model(
        "PanguProMoEV2ForCausalLM",
        "omni_npu.v1.models.pangu.pangu_pro_moe:PanguProMoEV2ForCausalLM")
    import os

    if (
        int(os.getenv("RANDOM_MODE", default='0'))
        or int(os.getenv("CAPTURE_MODE", default='0'))
        or int(os.getenv("REPLAY_MODE", default='0'))
    ):
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        from omni_npu.v1.models.mock.mock import mock_model_class_factory
        
        ModelRegistry.register_model(
            "Qwen2ForCausalLM", 
            mock_model_class_factory(Qwen2ForCausalLM))
            