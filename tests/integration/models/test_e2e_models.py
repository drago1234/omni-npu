
import pytest
import os
import json
import time

from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


models_path = os.path.join(os.path.dirname(__file__), "model_configs") 
models = [os.path.join(models_path, model) for model in os.listdir(models_path)]

@pytest.fixture(scope="module")
def init_env():
    os.environ["VLLM_PLUGINS"]="omni-npu,omni_pangu_models,omni_npu_patches"
    os.environ["OMNI_NPU_PATCHES_DIR"]="pangu_sink_swa_mla"
    os.environ["OMNI_NPU_VLLM_PATCHES"]="ALL"

@pytest.mark.parametrize('model', models)
@pytest.mark.parametrize('compilation', [True, False])
def test_e2e_models(init_env, model: str, compilation: bool, max_tokens: int=20):
    
    if compilation:
        config_file = os.path.join(model,"config.json")
        compilation_config = json.load(open(config_file)).get("compilation_config")

    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
  
    llm = LLM(
            model=model,
            trust_remote_code=True,
            load_format="dummy",
            skip_tokenizer_init=True,
            enforce_eager=False if compilation else True,
            compilation_config=compilation_config  if compilation else False
            )
    
    prompts = TokensPrompt(prompt_token_ids=[148899, 14518, 11, 1678, 1181, 405 ])
    outputs = llm.generate(prompts, sampling_params)
    
    assert outputs is not None
    for output in outputs:
        output_token_ids = output.outputs[0].token_ids
        assert len(output_token_ids) == max_tokens