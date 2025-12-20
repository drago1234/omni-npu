# omni-npu (vLLM NPU Plugin)

A vLLM (0.12.0) out-of-tree platform plugin that enables running vLLM on NPU (Ascend/torch_npu).

- Loaded via vLLM plugin entry points (no code changes to vLLM required).
- Provides a minimal NPU Platform, Worker, and a standalone NPU ModelRunner adapter.
- Uses vLLM's existing serving APIs unchanged.

## Requirements

- Python >= 3.12
- vLLM == 0.12.0
- torch and torch_npu for your platform (vendor-specific install)

## Install (order matters)

```bash
# 1) Install vLLM first (pin to 0.12.0 for compatibility)
pip install vllm==0.12.0

# 2) Install vendor runtime (example: torch_npu for Ascend)
# Follow your vendor instructions; example:
# pip install torch==<compatible> torch_npu==<compatible>

# 3) Install omni-npu plugin (this project)
pip install .
# or
pip install -e .
```

## How it works

- The plugin registers under the vLLM entry point group `vllm.platform_plugins`.
- vLLM discovers `omni_npu.platform.NPUPlatform` when `torch_npu` is available.
- The platform sets `device_type=npu`, configures worker class, and uses HCCL.
- The NPU worker constructs a standalone `NPUModelRunner` that adapts vLLM's model runner logic to NPU without subclassing GPUModelRunner.

## Usage (serve via vLLM API)

- OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model your/model \
  --device npu \
  --port 8000 \
  --trust-remote-code
```

- Python API:

```python
from vllm import LLM
llm = LLM(model="your/model", device="npu")
print(llm.generate(["hello world"]))
```

Notes:
- Keep using vLLM’s parameters; only change `--device npu`.
- Ensure `torch_npu` is installed and NPUs are visible (`ASCEND_RT_VISIBLE_DEVICES`).

## Troubleshooting

- Plugin not detected: ensure `pip show vllm omni-npu` lists both and `torch_npu` is importable.
- Distributed backend: HCCL is used; configure env (MASTER_ADDR/PORT) per your cluster.
- Memory issues: adjust `--gpu-memory-utilization` or `--kv-cache-memory`.

## License

MIT
