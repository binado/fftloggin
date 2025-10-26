# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fftloggin` is a high-performance vectorized FFTLog implementation for fast Hankel transforms in pure Python with optional numexpr acceleration. FFTLog implements the fast Hankel transform algorithm from Hamilton (2000) for efficient computation of integral transforms commonly used in cosmology and astrophysics.

## Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with numexpr support (requires numexpr to be installed)
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_fftlog.py
uv run pytest tests/test_fftlog_class.py
uv run pytest tests/test_kernels.py

# Run tests with verbose output
uv run pytest -v
```

### Dependencies
```bash
# Install core dependencies (numpy, scipy)
uv sync

# Install with development dependencies (includes pytest, ruff, pre-commit)
uv sync --group dev

# Install with numexpr acceleration (optional)
uv sync --group numexpr

# Install with tutorial dependencies (CAMB, matplotlib, jupyter)
uv sync --group tutorial
```

### Linting
```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .
```

## Architecture

### Core Modules

**fftlog.py** - Core FFTLog transform algorithm
- `FFTLog` class: Pure transform algorithm with configurable parameters (kernel, n, dlog, bias, logc)
- Focus on computation only - does NOT manage coordinates
- Key methods: `forward()`, `inverse()`, `compute_kernel_coefficients()`, `optimal_logcenter()`
- Cached properties for performance: `logc`, `kernel_coefficients`
- Low-level functions: `_forward_hankel_transform()`, `_inverse_hankel_transform()`

**kernels.py** - Mellin transform kernels for integral transforms
- `Kernel` base class: Defines interface for all kernels with `forward()` method and `strip` property
- `BesselJKernel`: Standard Hankel transform with Bessel function J_μ
- `Derivative`: Wrapper for computing derivatives of transforms via the Mellin transform property
- All kernels have a "strip of convergence" in the complex plane where the transform is valid
- Kernels support vectorization (e.g., multiple μ values for batch transforms)

**grids.py** - High-level workspace API with coordinate management
- `Grid` class: Combines coordinates (r, k), transform algorithm (FFTLog), and data storage (ar, ak)
- Stateful API: stores input/output data in `.ar` and `.ak` properties
- Factory methods: `from_r()`, `from_k()`, `from_fftlog()`
- Helper functions: `infer_dlog()`, `infer_logc()`, `get_other_array()`
- Follows scipy convention: y = exp(logc) / x[::-1]

### Design Philosophy

The library follows a **decoupled architecture**:
1. **FFTLog**: Pure algorithm - handles the math but not coordinates
2. **Grid**: Convenience wrapper - manages both coordinates and data
3. **Kernel**: Transform kernels - encapsulates integral kernel properties

Users can choose between:
- **High-level API**: Use `Grid` for convenience (recommended)
- **Low-level API**: Use `FFTLog` directly when you need fine control

### Key Concepts

**Log-center parameter (logc)**:
- Controls the relationship between input (r) and output (k) grids
- Can be "snapped" to minimize ringing artifacts via `minimize_ringing=True`
- Computed optimally via `optimal_logcenter()` method

**Bias parameter**:
- Power-law bias exponent for improved numerical stability
- Applied as (r/r_c)^(-bias) before transform

**Strip of convergence**:
- Each kernel defines a range in complex s-plane where Mellin transform is valid
- Kernels validate input automatically in `forward()` method

**Vectorization**:
- Kernels support batch transforms (e.g., multiple μ values)
- Grid batching: kernel parameters can be arrays with shape (*batch_shape,)

## Testing Strategy

Tests are organized by module:
- `test_fftlog.py` - Low-level FFTLog algorithm tests
- `test_fftlog_class.py` - FFTLog class API tests (currently references old API)
- `test_kernels.py` - Kernel implementations

Note: Some test files may reference an older API (e.g., `fht`, `ifht` functions, `offset` parameter). The current API uses `FFTLog` class with `logc` parameter instead.

## Common Patterns

### Creating a Grid and performing transforms
```python
from fftloggin import Grid
from fftloggin.kernels import BesselJKernel
import numpy as np

# From r coordinates
r = np.logspace(-2, 2, 128)
grid = Grid.from_r(r, kernel=BesselJKernel(0))

# Transform
a = np.exp(-(grid.r/1.0)**2)
A = grid.forward(a)

# Access results
print(grid.k)  # Output coordinates
print(grid.ak)  # Transformed data
```

### Using FFTLog directly (low-level)
```python
from fftloggin import FFTLog
from fftloggin.kernels import BesselJKernel

fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)
A = fftlog.forward(a)  # You manage coordinates separately
```

### Computing derivatives
```python
from fftloggin.kernels import BesselJKernel

kernel = BesselJKernel(0)
d_kernel = kernel.derive(1)  # First derivative
d2_kernel = kernel.derive(2)  # Second derivative
```

## References

The implementation follows these papers:
- Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191) - Original FFTLog algorithm
- Assassi et al., 2017 (1705.05022) - Efficient evaluation of cosmological angular statistics
- Schöneberg et al., 2018 (1807.09540) - Beyond traditional line-of-sight approach
- Fang et al., 2020 (1911.11947) - Beyond Limber approximation

## File Locations

- Source code: `src/fftloggin/`
- Tests: `tests/`
- Notebooks: `notebooks/` (if present)
- Benchmarks: `benchmarks/` (if present)
