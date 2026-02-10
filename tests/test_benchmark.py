"""
Test FFTLog implementation against Fortran benchmark results.
"""

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from fftloggin import FFTLog, NumPyFFTBackend, SciPyFFTBackend
from fftloggin.kernels import BesselJKernel

# Check if benchmarks exist
BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
BENCHMARKS_EXIST = BENCHMARK_DIR.exists() and list(
    BENCHMARK_DIR.glob("benchmark_*.txt")
)

# Skip all benchmark tests if benchmarks don't exist
pytestmark = pytest.mark.skipif(
    not BENCHMARKS_EXIST,
    reason="Benchmark files not found. Run: python scripts/generate_benchmarks.py",
)


def f(x: npt.NDArray, mu: float):
    """
    Analytical function: f(r) = r^(mu+1) * exp(-r^2/2)

    This is the test function used to generate the Fortran benchmarks.
    """
    return x ** (mu + 1) * np.exp(-(x**2) / 2)


def parse_benchmark_filename(filename):
    """
    Parse benchmark filename to extract parameters.

    Format: benchmark_log10rmin={}_log10rmax={}_n={}_mu={}_q={}_kr={}_lowring={}.txt
    """
    pattern = r"benchmark_log10rmin=([-\d.]+)_log10rmax=([-\d.]+)_n=(\d+)_mu=(\d+)_q=([-\d.]+)_kr=([-\d.]+)_lowring=([yn])\.txt"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")

    log10rmin, log10rmax, n, mu, q, kr, lowring = match.groups()
    return {
        "log10rmin": float(log10rmin),
        "log10rmax": float(log10rmax),
        "n": int(n),
        "mu": int(mu),
        "q": float(q),
        "kr": float(kr),
        "lowring": lowring == "y",
    }


def get_benchmark_files() -> list[Path]:
    """Get all benchmark files in the benchmarks directory."""
    if not BENCHMARKS_EXIST:
        return []
    return sorted(BENCHMARK_DIR.glob("benchmark_*.txt"))


def get_fft_backends():
    """Get all available FFT backend instances."""
    backends = [
        pytest.param(SciPyFFTBackend(), id="scipy"),
        pytest.param(NumPyFFTBackend(), id="numpy"),
    ]
    try:
        from fftloggin import PyFFTWBackend

        backends.append(pytest.param(PyFFTWBackend(), id="pyfftw"))
    except ImportError:
        pass
    return backends


@pytest.mark.benchmark
@pytest.mark.parametrize("rtol", [1e-5])
@pytest.mark.parametrize("benchmark_file", get_benchmark_files(), ids=lambda f: f.name)
@pytest.mark.parametrize("fft_backend", get_fft_backends())
def test_benchmark(benchmark_file: Path, rtol: float, fft_backend):
    """
    Test FFTLog against a single benchmark file.

    This test:
    1. Loads benchmark data from the Fortran executable output
    2. Extracts parameters from the filename
    3. Constructs the r array from log10rmin, log10rmax, n
    4. Creates an FFTLog object with the specified parameters
    5. Creates a Grid from the FFTLog
    6. Evaluates the analytical function on the r grid
    7. Performs the forward FFTLog transform
    8. Evaluates the analytical solution on the k grid
    9. Compares the results against the benchmark values
    """
    params = parse_benchmark_filename(benchmark_file.name)

    # Load benchmark data: k, a_fftlog, a_analytical
    data = np.loadtxt(benchmark_file, skiprows=1)
    k_expected = data[:, 0]
    a_fftlog_expected = data[:, 1]
    a_analytical_expected = data[:, 2]

    # Extract parameters
    log10rmin = params["log10rmin"]
    log10rmax = params["log10rmax"]
    n = params["n"]
    mu = params["mu"]
    q = params["q"]  # bias parameter
    kr = params["kr"]
    lowring = params["lowring"]

    # Create r array using logspace
    r = np.logspace(log10rmin, log10rmax, n)

    # Create FFTLog object with specified parameters
    kernel = BesselJKernel(mu)
    dlog = (log10rmax - log10rmin) / (n - 1) * np.log(10)

    fftlog = FFTLog(
        kernel=kernel,
        n=n,
        dlog=dlog,
        bias=q,
        kr=kr,
        lowring=lowring,
        backend=fft_backend,
    )

    # Create grid
    grid = fftlog.create_grid(r=r)

    # Evaluate analytical function on r grid
    fr = f(r, mu)

    # Perform forward transform
    ak = fftlog.forward(fr)

    # Evaluate analytical solution on k grid
    fk = f(grid.k, mu)

    # Compare results
    # k values should match very closely
    np.testing.assert_allclose(
        grid.k,
        k_expected,
        err_msg=f"k mismatch in {benchmark_file.name}",
    )

    # FFTLog transform should match benchmark results
    np.testing.assert_allclose(
        ak,
        a_fftlog_expected,
        rtol=rtol,
        err_msg=f"FFTLog transform mismatch in {benchmark_file.name}",
    )

    # Analytical solution should match
    np.testing.assert_allclose(
        fk,
        a_analytical_expected,
        err_msg=f"Analytical solution mismatch in {benchmark_file.name}",
    )
