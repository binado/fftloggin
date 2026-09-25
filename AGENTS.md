# Repository guidance

`fftloggin` implements one-dimensional FFTLog transforms using JAX. Public
entry points are `forward`, `inverse`, `plan`, `product_plan`,
`lowring_log_kr`, the coordinate functions in `src/fftloggin/grids.py`, and the
unequal-time (double spherical Bessel) plan constructor
`double_spherical_bessel_plan` in `src/fftloggin/cosmology.py`. Kernel
instances are frozen JAX pytrees with scalar data leaves. A `Plan` is a frozen
pytree holding precomputed coefficients and their grid parameters, with the
sample count `n` as static metadata; `forward`/`inverse` accept either a
kernel with grid keywords or a plan without them.

Set `JAX_ENABLE_X64=1` for numerical reference comparisons. The package must
not change process-wide JAX configuration on import. Keep traced code free of
Python conversions of parameter values; `validate_parameters` and
`infer_dlog` are deliberately eager helpers.

The test suite lives in `tests/`; `tests/test_benchmark.py` additionally
compares against the Fortran FFTLog and runs only with `--run-benchmarks`
(requires gfortran). The previous API's tests are quarantined in
`tests/legacy/`.

## Development

- Use the conventional commits format for commit messages
- Use `uv` for running python and devtools:

```bash
uv sync --all-groups
uv run ruff check .
JAX_ENABLE_X64=1 uv run pytest
JAX_ENABLE_X64=1 uv run pytest tests/test_benchmark.py --run-benchmarks
```

## Code styleguide

### jax

- Always assert operations are compatible with `jit`, `vmap` and `grad`
- Annotate inputs arrays with `jax.typing.ArrayLike`
and output arrays with `jax.Array`

### Tests
- Use pytest fixtures,
- Use `pytest.mark.parametrize` decorator
- Don't test internal behavior or private methods
