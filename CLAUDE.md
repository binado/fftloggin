# Repository guidance

`fftloggin` implements one-dimensional FFTLog transforms using JAX. Public
entry points are `forward`, `inverse`, `lowring_log_kr`, and the coordinate
functions in `src/fftloggin/grids.py`. Kernel instances are frozen JAX pytrees
with scalar data leaves. There is no transform object, FFT backend abstraction,
or implicit parameter broadcasting.

Set `JAX_ENABLE_X64=1` for numerical reference comparisons. The package must
not change process-wide JAX configuration on import. Keep traced code free of
Python conversions of parameter values; `validate_parameters` and
`infer_dlog` are deliberately eager helpers.

The only active test in this API cutover is the 216-case Fortran benchmark in
`tests/test_benchmark.py`. The previous API's tests are quarantined in
`tests/legacy/`; a replacement suite belongs in a separate commit.

```bash
uv sync --all-groups
uv run ruff check .
JAX_ENABLE_X64=1 uv run pytest tests/test_benchmark.py --run-benchmarks
```
