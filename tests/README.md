# omni-npu Test Suite

This directory contains comprehensive tests for the omni-npu package, separated into unit tests and integration tests.

**Current Status:** NPUCommunicator tests implemented and serving as template for other components.

## Test Structure

Tests are organized to mirror the source code structure in `src/omni_npu/`.

### Unit Tests (`tests/unit/`)
**Do NOT require NPU hardware** - use mocking to verify logic and API contracts.

```
tests/unit/
├── __init__.py
├── distributed/
│   ├── __init__.py
│   └── test_communicator.py    # ✅ NPUCommunicator tests
├── attention/                   # 🔲 TODO
│   └── backends/
├── v1/                          # 🔲 TODO
│   ├── sample/
│   └── worker/
└── test_platform.py             # 🔲 TODO: NPUPlatform tests
```

**Currently Implemented:**
- ✅ NPUCommunicator: Initialization, collective ops, point-to-point ops, edge cases
- Uses mocks to avoid requiring actual NPU hardware

**To Be Implemented:**
- 🔲 NPUPlatform: Device management, memory operations
- 🔲 Attention backends: Attention mechanisms, MLA
- 🔲 NPU Worker & Model Runner: Batch processing, model execution
- 🔲 Sampler: Sampling strategies

### Integration Tests (`tests/integration/`)
**REQUIRE NPU hardware** - verify end-to-end functionality with real devices.

```
tests/integration/
├── __init__.py
├── distributed/
│   ├── __init__.py
│   └── test_communicator.py    # ✅ NPUCommunicator integration tests
├── attention/                   # 🔲 TODO
│   └── backends/
└── v1/                          # 🔲 TODO
    └── worker/
```

**Currently Implemented:**
- ✅ NPUCommunicator: Device operations, multi-device communication (with torchrun)
- Automatically skipped if NPU hardware is not available

**To Be Implemented:**
- 🔲 Attention backends: End-to-end attention with real NPU
- 🔲 NPU Worker: End-to-end model inference workflows

## Running Tests

### Install test dependencies

```bash
pip install -e ".[test]"
```

### Run unit tests only (no NPU required)

```bash
pytest tests/unit/
```

### Run integration tests (requires NPU hardware)

```bash
pytest tests/integration/
```

### Run all tests

```bash
pytest tests/
```

### Run with coverage

```bash
pytest --cov=omni_npu --cov-report=html --cov-report=term tests/
```

### Run specific test file

```bash
pytest tests/unit/distributed/test_communicator.py
```

### Run specific test

```bash
pytest tests/unit/distributed/test_communicator.py::TestNPUCommunicatorUnit::test_init_with_torch_npu_available
```

### Run multi-device integration tests (requires 2+ NPUs)

```bash
torchrun --nproc_per_node=2 -m pytest tests/integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice
```

## CI/CD Integration

For CI/CD pipelines without NPU hardware:
```bash
# Run only unit tests
pytest tests/unit/ -v
```

For systems with NPU hardware:
```bash
# Run all tests including integration
pytest tests/ -v
```

## Notes

- **Unit tests** use mocking and can run anywhere
- **Integration tests** are automatically skipped if NPU is not available
- Multi-device tests require `torchrun` and multiple NPU devices
- Tests verify correct delegation to torch.distributed APIs
