# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fftloggin` is a high-performance vectorized FFTLog implementation for fast Hankel transforms in pure Python. FFTLog implements the fast Hankel transform algorithm from Hamilton (2000) for efficient computation of integral transforms commonly used in cosmology and astrophysics.

## Development Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v
```

### Dependencies
```bash
# Install core dependencies (numpy, scipy)
uv sync

# Install with development dependencies (includes pytest, ruff, pre-commit)
uv sync --group dev

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
- `FFTLog` class: Pure transform algorithm with configurable parameters (kernel, n, dlog, bias, kr, lowring)
- Focus on computation only - does NOT manage coordinates
- Parameters `dlog`, `bias`, `kr` can be scalars or arrays with shape `(*batch_shape, 1)` for batch transforms
- Key methods: `forward()`, `inverse()`, `optimal_logcenter()`
- Cached properties for performance: `kr`, `logc`, `kernel_coefficients`
- Low-level functions: `_forward_hankel_transform()`, `_inverse_hankel_transform()`

**kernels.py** - Mellin transform kernels for integral transforms
- `Kernel` base class: Defines interface for all kernels with `forward()` method (computation) and `__call__()` method (with bounds checking) and `strip` property
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

**Log-center parameter (kr)**:
- The product k*r at the geometric center of the grid (public user-facing parameter)
- Internally represented as `logc = np.log(kr)` for mathematical computations
- Can be "snapped" to minimize ringing artifacts via `lowring=True`
- Both `kr` and `logc` properties are available for convenience

**Bias parameter**:
- Power-law bias exponent for improved numerical stability
- Applied as (r/r_c)^(-bias) before transform

**Strip of convergence**:
- Each kernel defines a range in complex s-plane where Mellin transform is valid
- Kernels validate input automatically in `forward()` method

**Vectorization and Batching**:
- Kernels support batch transforms (e.g., multiple μ values)
- Parameters `dlog`, `bias`, `kr` can be arrays with shape `(*batch_shape, 1)` for batch transforms
- Use `prepare_batch_params()` helper to prepare parameters for batching
- Result arrays have shape `(*batch_shape, n)` when using batched parameters

## Testing Strategy

Tests are organized by module:
- `test_fftlog.py` - Low-level FFTLog algorithm tests
- `test_grid.py` - Grid API tests
- `test_kernels.py` - Kernel implementations
- `test_utils.py` - Utility function tests
- `test_benchmark.py` - Benchmark tests comparing against Fortran reference implementation (optional)

## Common Patterns

### Creating a Grid and performing transforms
```python
from fftloggin import FFTLog, Grid
from fftloggin.kernels import BesselJKernel
import numpy as np

# Create FFTLog from r coordinates
r = np.logspace(-2, 2, 128)
fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0), kr=1.0)

# Create grid
grid = fftlog.create_grid(r=r)

# Transform
a = np.exp(-(grid.r/1.0)**2)
A = fftlog.forward(a)

# Access results
print(grid.k)  # Output coordinates
print(A)  # Transformed data
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

### Batching transforms with array parameters
```python
from fftloggin import FFTLog, prepare_batch_params
from fftloggin.kernels import BesselJKernel
import numpy as np

# Setup input data and coordinates
r = np.logspace(-2, 2, 128)
a = np.exp(-(r/1.0)**2)

# Method 1: Using prepare_batch_params helper (recommended)
# Automatically handles reshaping to (*batch_shape, 1)
dlog, bias, kr = prepare_batch_params(
    dlog=0.05,           # scalar
    bias=0.0,            # scalar
    kr=[0.5, 1.0, 2.0]  # batch over 3 kr values
)

fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=dlog, bias=bias, kr=kr)
result = fftlog.forward(a)
print(result.shape)  # (3, 128)

# Method 2: Manual reshaping
# Must add trailing singleton dimension for proper broadcasting
kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)  # shape (3, 1)
fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05, kr=kr)
result = fftlog.forward(a)  # shape (3, 128)

# Batch over multiple parameters
dlog, bias, kr = prepare_batch_params(
    dlog=[0.04, 0.05],   # 2 values
    bias=[0.0, 0.1],     # 2 values
    kr=[1.0, 2.0]        # 2 values
)
# After prepare_batch_params, all have compatible shapes (2, 1)
fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=dlog, bias=bias, kr=kr)
result = fftlog.forward(a)  # shape (2, 128)

# Combining kernel batching with parameter batching
# Requires manual shape management for outer product broadcasting
mu_values = np.array([0, 1, 2])  # 3 different orders
kr_values = np.array([0.5, 1.0])  # 2 different kr values

# For outer product: need shapes (3, 1, 1) and (1, 2, 1)
mu_batch = mu_values.reshape(-1, 1, 1)  # Remove last singleton for kernel
kr_batch = kr_values.reshape(1, -1, 1)

kernel = BesselJKernel(mu_batch.squeeze(-1))  # shape (3, 1) for kernel
fftlog = FFTLog(kernel=kernel, n=128, dlog=0.05, kr=kr_batch)
result = fftlog.forward(a)  # shape (3, 2, 128)
```

**Key principles for batching**:
1. **Pre-shaped arrays**: Parameters must arrive with shape `(*batch_shape, 1)` for proper broadcasting
2. **Use `prepare_batch_params`**: Handles common reshaping and validates compatibility
3. **Manual control**: For complex batch shapes, reshape arrays explicitly before passing to FFTLog
4. **Broadcasting rules**: Standard NumPy broadcasting applies - shapes must be compatible

**Shape requirements**:
- Input data `a`: shape `(n,)` (no batch dimension)
- Parameters `dlog`, `bias`, `kr`: scalar or shape `(*batch_shape, 1)`
- Kernel coefficients: shape `(*batch_shape, ns)` where `ns = n//2 + 1`
- Result: shape `(*batch_shape, n)`

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
