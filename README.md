# fftloggin

Vectorized FFTLog in pure python

## Installation

```bash
# Install with uv
uv add fftloggin

# Or with pip
pip install fftloggin
```

### Optional PyFFTW backend

If you want FFTW-based plans and buffer reuse, install the optional dependency
and use the PyFFTW backend.

```bash
# With uv
uv add "fftloggin[fftw]"

# Or with pip
pip install "fftloggin[fftw]"
```

```python
from fftloggin import FFTLog, PyFFTWBackend
from fftloggin.kernels import BesselJKernel

fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05, backend=PyFFTWBackend())
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/binado/fftloggin.git
cd fftloggin

# Install dependencies
uv sync --all-groups
```

### Running Tests

```bash
# Run standard tests (fast)
uv run pytest

# Run all tests excluding benchmarks
uv run pytest -m "not benchmark"

# Run with verbose output
uv run pytest -v
```

### Benchmark Tests

Benchmark tests compare the Python implementation against the original Fortran FFTLog code. These tests are optional and require a Fortran compiler.

#### Prerequisites

- Have `gfortran` installed on your PATH

#### Running Benchmarks

```bash
# Generate benchmark reference files
python scripts/generate_benchmarks.py

# Run benchmark tests
uv run pytest --run-benchmarks

# Or run only benchmark tests
uv run pytest tests/test_benchmark.py --run-benchmarks -v

# Generate and run benchmarks in one command
uv run pytest --generate-benchmarks --run-benchmarks
```

#### Regenerating Benchmarks

If you need to regenerate the benchmark files:

```bash
# Remove old benchmarks
rm -rf tests/benchmarks/*.txt

# Generate fresh benchmarks
python scripts/generate_benchmarks.py
```

### Linting

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .
```

## References

- Hamilton, A. J. S. "Uncorrelated modes of the non-linear power spectrum." Monthly Notices of the Royal Astronomical Society 312.2 (2000): 257-284. [[astro-ph/9905191]](https://arxiv.org/abs/astro-ph/9905191)
- Assassi, Valentin, Marko Simonović, and Matias Zaldarriaga. "Efficient Evaluation of Cosmological Angular Statistics." arXiv preprint arXiv:1705.05022 (2017). [[1705.05022]](https://arxiv.org/abs/1705.05022)
- Schöneberg, Nils, et al. "Beyond the traditional Line-of-Sight approach of cosmological angular statistics." Journal of Cosmology and Astroparticle Physics 2018.10 (2018): 047. [[1807.09540]](https://arxiv.org/abs/1807.09540)
- Fang, Xiao, et al. "Beyond Limber: Efficient computation of angular power spectra for galaxy clustering and weak lensing." Journal of Cosmology and Astroparticle Physics 2020.05 (2020): 010. [[1911.11947]](https://arxiv.org/abs/1911.11947)
