"""Pytest configuration for fftloggin tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for benchmark tests."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="Run benchmark tests (requires gfortran)",
    )
    parser.addoption(
        "--generate-benchmarks",
        action="store_true",
        default=False,
        help="Generate benchmark files before running tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: marks tests as benchmark tests (deselect with '-m \"not benchmark\"')",
    )


def pytest_collection_modifyitems(config, items):
    """Skip benchmark tests by default unless --run-benchmarks is specified."""
    if config.getoption("--run-benchmarks"):
        # Run benchmarks
        return

    # Skip benchmark tests by default
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture(scope="session", autouse=True)
def generate_benchmarks_fixture(request):
    """Generate benchmarks if requested via --generate-benchmarks flag."""
    if not request.config.getoption("--generate-benchmarks"):
        return

    # Import and call the generation function directly (same directory)
    from generate_benchmarks import main as generate_benchmarks_main

    print("\nGenerating benchmarks...")

    try:
        generate_benchmarks_main()
    except SystemExit as e:
        if e.code != 0:
            pytest.exit(f"Benchmark generation failed with exit code {e.code}")
