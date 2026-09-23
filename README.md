# fftloggin

Differentiable FFTLog transforms built on JAX. The core API transforms one
real, one-dimensional sample array at a time. Use JAX transformations for
compilation, batching, and differentiation.

Full user guides and the API reference are available on
[Read the Docs](https://fftloggin.readthedocs.io/en/latest/).

## Installation

```bash
uv add fftloggin
# or: pip install fftloggin
```

For the numerical accuracy expected by the Fortran reference cases, enable
JAX 64-bit values **before importing JAX or fftloggin**:

```bash
export JAX_ENABLE_X64=1
```

The library does not change process-wide JAX configuration on import.

## Transform a logarithmic grid

```python
import jax
import jax.numpy as jnp
from fftloggin import BesselJKernel, forward, get_paired_grids, infer_dlog

r = jnp.geomspace(1e-2, 1e2, 128)
a = r * jnp.exp(-(r**2) / 2)
dlog = infer_dlog(r)  # eager spacing validation
kernel = BesselJKernel(mu=0.0)
log_kr = 0.0

r, k = get_paired_grids(r=r, log_kr=log_kr)
A = jax.jit(forward)(a, kernel, dlog=dlog, log_kr=log_kr)
```

`log_kr` is the logarithm of the product of the input and output grid centers.
`get_paired_grids` accepts exactly one of `r` or `k` and always returns the
pair in `(r, k)` order.
To request the traditional low-ringing snap, calculate it explicitly and use
the returned value for both the transform and the paired grid:

```python
from fftloggin import lowring_log_kr

log_kr = lowring_log_kr(kernel, dlog=dlog, log_kr=0.0)
r, k = get_paired_grids(r=r, log_kr=log_kr)
A = forward(a, kernel, dlog=dlog, log_kr=log_kr)
```

The snap is piecewise constant in the requested `log_kr`. For fitting that
parameter, use `forward` without snapping.

## Batch and differentiate

```python
mus = jnp.array([0.0, 1.0, 2.0])
batched = jax.jit(jax.vmap(lambda mu: forward(a, BesselJKernel(mu), dlog=dlog)))(mus)


def loss(mu):
    prediction = forward(a, BesselJKernel(mu), dlog=dlog)
    return jnp.sum((prediction - A) ** 2)


gradient = jax.grad(loss)(0.5)
```

The built-in kernels are JAX pytrees. `forward` and `inverse` require scalar
`dlog`, `bias`, and `log_kr`; map over any of them with `jax.vmap`. Call
`validate_parameters(kernel, dlog=..., bias=..., log_kr=...)` outside JAX
transformations for eager value and Mellin-domain checks.

## Development and reference comparison

```bash
uv sync --all-groups
uv run ruff check .
JAX_ENABLE_X64=1 uv run pytest tests/test_benchmark.py --run-benchmarks
```

For opt-in runtime checks of the jaxtyping annotations during development,
install the development dependencies, then run tests with
`FFTLOGGIN_RUNTIME_TYPECHECK=1 uv run pytest`. This uses beartype to check
shapes and dtypes while JAX traces functions; the checks are absent from
normal imports and compiled execution. The flag only enables checks when its
value is exactly `1`.

The benchmark compares with 216 generated reference outputs from the original
Fortran FFTLog program. The files are ignored by Git. To generate them, install
`gfortran` and run `uv run python scripts/generate_benchmarks.py`. The previous
API's tests live in `tests/legacy/` for the separate test-suite redesign.

## References

- Hamilton, A. J. S. (2000), *Uncorrelated modes of the non-linear power spectrum*, [astro-ph/9905191](https://arxiv.org/abs/astro-ph/9905191).
- Assassi, V., Simonović, M., and Zaldarriaga, A. (2017), *Efficient Evaluation of Cosmological Angular Statistics*, [arXiv:1705.05022](https://arxiv.org/abs/1705.05022).
