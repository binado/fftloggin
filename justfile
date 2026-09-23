mod docs

default:
    @just --list

lint:
    uv run ruff check .

test:
    JAX_ENABLE_X64=1 uv run pytest

typecheck:
    uv run ty check

fmt:
    uv run ruff format .

test-benchmark:
    JAX_ENABLE_X64=1 uv run pytest tests/test_benchmark.py --run-benchmarks
