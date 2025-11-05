# Test Suite Summary for NPUCommunicator

## Overview

Comprehensive test suite created for `NPUCommunicator` with clear separation between unit tests (no hardware required) and integration tests (NPU hardware required).

## Files Created

### Test Files
```
tests/
├── __init__.py
├── README.md                                    # Test documentation
├── TESTING.md                                   # Detailed testing guide
├── unit/
│   ├── __init__.py
│   └── test_npu_communicator_unit.py           # 20+ unit tests (no NPU required)
└── integration/
    ├── __init__.py
    └── test_npu_communicator_integration.py    # 8+ integration tests (NPU required)
```

### Configuration Files
- `pytest.ini` - Pytest configuration with markers and coverage settings
- `pyproject.toml` - Updated with test dependencies and pytest config
- `run_tests.sh` - Shell script to run different test suites

## Test Coverage

### Unit Tests (tests/unit/)
**No NPU hardware required** - Uses mocking

✅ **Initialization Tests**
- Test with torch.npu available
- Test without torch.npu (raises RuntimeError)

✅ **Collective Operations**
- `all_reduce()` - Delegation to torch.distributed
- `all_gather()` - Shape transformation logic
- `reduce_scatter()` - World size handling, delegation
- `reduce_scatterv()` - Variable sizes support
- `all_gatherv()` - NotImplementedError for dim != 0
- `gather()` - Destination vs non-destination rank behavior

✅ **Point-to-Point Operations**
- `send()` - Explicit and default destination
- `recv()` - Tensor creation with correct shape/dtype

✅ **Edge Cases**
- World size = 1 optimization
- Negative dimension handling
- Destroy method

### Integration Tests (tests/integration/)
**Requires NPU hardware** - Real device testing

✅ **Single Device Tests**
- NPU device availability
- Tensor creation on NPU
- Memory allocation/deallocation
- Basic NPU operations (add, matmul)
- Communicator initialization with real NPU

✅ **Multi-Device Tests** (requires torchrun)
- Real all_reduce with multiple NPUs
- Real all_gather with multiple NPUs
- Real send/recv between ranks

## Running Tests

### Quick Commands

```bash
# Unit tests only (no NPU)
./run_tests.sh unit

# Integration tests only (requires NPU)
./run_tests.sh integration

# All tests
./run_tests.sh all
```

### Using pytest directly

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run with markers
pytest -m unit -v                    # Only unit tests
pytest -m integration -v             # Only integration tests
pytest -m "not multi_device" -v      # Skip multi-device tests

# Run with coverage
pytest tests/unit/ --cov=omni_npu --cov-report=html
```

### Multi-device tests

```bash
torchrun --nproc_per_node=2 -m pytest tests/integration/ -v
```

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.unit` - Unit tests (no hardware)
- `@pytest.mark.integration` - Integration tests (requires NPU)
- `@pytest.mark.multi_device` - Requires multiple NPUs
- `@pytest.mark.slow` - Long-running tests

## CI/CD Integration

### Without NPU Hardware (GitHub Actions, etc.)
```yaml
- run: pip install -e ".[test]"
- run: pytest tests/unit/ -v --cov=omni_npu
```

### With NPU Hardware (On-premise CI)
```yaml
- run: pip install -e ".[test,npu]"
- run: pytest tests/ -v --cov=omni_npu
```

## Key Features

✅ **Separation of Concerns**
- Unit tests don't require NPU hardware
- Integration tests automatically skip if NPU unavailable

✅ **Comprehensive Coverage**
- All NPUCommunicator methods tested
- Edge cases and error handling covered
- Real hardware validation included

✅ **Easy to Run**
- Simple shell script interface
- Pytest markers for filtering
- Clear documentation

✅ **CI/CD Ready**
- Can run unit tests anywhere
- Integration tests skip gracefully without NPU
- Coverage reporting included

## Test Statistics

- **Total test cases**: 28+
- **Unit tests**: 20+ (no hardware required)
- **Integration tests**: 8+ (NPU required)
- **Methods covered**: All NPUCommunicator public methods
- **Target coverage**: >80%

## Next Steps

1. Run unit tests to verify setup:
   ```bash
   ./run_tests.sh unit
   ```

2. On NPU hardware, run integration tests:
   ```bash
   ./run_tests.sh integration
   ```

3. Add more tests as new features are implemented

4. Integrate into CI/CD pipeline
