from vllm import ModelRegistry


def register_models():
    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni_npu.v1.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")