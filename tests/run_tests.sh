#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Script to run tests for omni-npu

set -e

# Parse command line arguments
TEST_TYPE="${1:-all}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install -e ".[test]"
fi

# Check if pytest-cov is installed
if python3 -c "import pytest_cov" 2>/dev/null; then
    HAS_COV=true
else
    HAS_COV=false
    echo "Note: pytest-cov not installed. Running without coverage."
    echo "To enable coverage, run: pip install -e \".[test]\""
    echo ""
fi

case "$TEST_TYPE" in
    unit)
        echo "Running unit tests (no NPU required)..."
        if [ "$HAS_COV" = true ]; then
            pytest unit/ \
                --cov=omni_npu \
                --cov-report=term-missing \
                --cov-report=html \
                --cov-config=./.coveragerc \
                -v
        else
            pytest unit/ -v
        fi
        ;;
    integration)
        echo "Running integration tests (requires NPU hardware)..."
        echo "  - Single-device tests with pytest"
        pytest integration/ -v -k "not TestNPUCommunicatorMultiDevice"
        echo ""
        echo "  - Multi-device tests with torchrun (2 NPUs)"
        torchrun --nproc_per_node=2 -m pytest integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice -v
        ;;
    all)
        echo "Running all tests..."
        if [ "$HAS_COV" = true ]; then
            pytest unit/ \
                --cov=omni_npu \
                --cov-report=term-missing \
                --cov-report=html \
                -v
        else
            pytest unit/ -v
        fi
        echo ""
        echo "Running integration tests (requires NPU hardware)..."
        echo "  - Single-device tests with pytest"
        pytest integration/distributed/test_communicator.py::TestNPUCommunicatorIntegration -v
        echo ""
        echo "  - Multi-device tests with torchrun (2 NPUs)"
        torchrun --nproc_per_node=2 -m pytest integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice -v
        ;;
    *)
        echo "Usage: $0 [unit|integration|all]"
        echo ""
        echo "  unit        - Run unit tests only (no NPU required)"
        echo "  integration - Run integration tests only (requires NPU)"
        echo "  all         - Run all tests (default)"
        exit 1
        ;;
esac

echo ""
if [ "$HAS_COV" = true ] && ([ "$TEST_TYPE" = "unit" ] || [ "$TEST_TYPE" = "all" ]); then
    echo "Coverage report saved to htmlcov/index.html"
fi
echo "Tests completed!"
