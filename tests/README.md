# NPU Communicator Tests

This directory contains tests for the omni-npu package, separated into unit tests and integration tests.

## Test Structure

Tests are organized to mirror the source code structure in `src/omni_npu/`.

### Unit Tests (`tests/unit/`)
**Do NOT require NPU hardware** - use mocking to verify logic and API contracts.

```
tests/unit/
├── __init__.py
└── distributed/
    ├── __init__.py
    └── test_communicator.py    # Unit tests for NPUCommunicator
```

- Tests initialization with and without torch.npu
- Tests all collective operations (all_reduce, all_gather, reduce_scatter, etc.)
- Tests point-to-point operations (send, recv)
- Tests edge cases and error handling
- Uses mocks to avoid requiring actual NPU hardware

### Integration Tests (`tests/integration/`)
**REQUIRE NPU hardware** - verify end-to-end functionality with real devices.

```
tests/integration/
├── __init__.py
└── distributed/
    ├── __init__.py
    └── test_communicator.py    # Integration tests for NPUCommunicator
```

- Tests NPU device availability and basic operations
- Tests communicator initialization with real NPU
- Tests multi-device distributed operations (requires torchrun)
- Automatically skipped if NPU hardware is not available

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
