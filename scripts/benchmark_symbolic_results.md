# Mellin expression benchmark

These are historical results from the removed custom engine. The current
`benchmark_symbolic.py` compares generated kernels with direct JAX formulas.

Measured 2026-10-09 on macOS arm64 CPU, JAX 0.10.2, SymPy 1.14.0, with `JAX_ENABLE_X64=1`.

The custom engine and SymPy generate broadly similar steady-state JIT performance. SymPy reduces eager work and compilation cost for redundant expressions. These CPU measurements do not establish GPU performance.

## Method

The same analytic Mellin formulas are composed using the public custom expression API or parsed with `sympy.sympify` and emitted using `lambdify(..., modules=[{"loggamma": _loggamma}, "jax"])`, with and without CSE. Both use the package's complex log-gamma primitive and its custom JVP. This isolates expression representation; it does not benchmark SymPy deriving a Mellin transform from a real-space Bessel expression.

All four scalar parameters remain dynamic JAX inputs: `(mu, weight, power, scale) = (4.5, 2, 0.2, 1.7)`. Samples are 1,024 complex points with real part 1.1 and imaginary parts 0.1–80. Five warmup calls precede eleven randomized timing rounds of fifty calls per implementation. Every invocation synchronizes using `block_until_ready()`, following the [JAX timing guidance](https://docs.jax.dev/en/latest/async_dispatch.html). CSE is the optional common-subexpression elimination provided by [SymPy lambdify](https://docs.sympy.org/latest/modules/utilities/lambdify.html).

Let `K(t) = J_mu(t)` and `W(t) = weight * t**power * J_mu(scale*t)`. The expressions are `K`, `W - diff(K,t)`, `diff(W,t,2)`, and a sum of eight identical `W` terms. SymPy automatically combines the last expression into `8*W`, including without CSE.

## Steady-state JIT

| Expression | Custom (µs) | SymPy (µs) | SymPy + CSE (µs) |
|---|---:|---:|---:|
| base | 108.59 | 101.39 | 102.78 |
| nested | 178.79 | 176.68 | 176.74 |
| second_derivative | 101.65 | 100.82 | 101.84 |
| repeated_sum | 102.76 | 103.36 | 104.48 |

Differences are about 0–7% in this run, with overlapping timing ranges. Earlier runs varied in which implementation was fastest. These results show no consistent substantial JIT advantage. Additional float64 runs covered 1 and 16,384 samples and gave the same general outcome.

## Compilation and eager evaluation

Compilation is one fresh measurement per variant after clearing in-memory caches and disabling persistent compilation caching. It excludes tracing/lowering. Eager times are medians of three rounds of three calls, after warming primitive caches; these estimates are less precise than the JIT timings. The custom eager column uses an expression constructed once before timing, so it excludes expression rebuilding.

| Expression | Custom compile (ms) | SymPy compile (ms) | Custom eager (µs) | SymPy eager (µs) |
|---|---:|---:|---:|---:|
| base | 149.50 | 157.01 | 266.86 | 262.01 |
| nested | 347.67 | 350.40 | 576.75 | 452.35 |
| second_derivative | 169.93 | 169.05 | 403.24 | 344.47 |
| repeated_sum | 248.09 | 167.67 | 1561.83 | 303.49 |

The eight-term sum compiled about 32% faster with SymPy and evaluated about 5.1× faster eagerly. Its steady-state JIT execution stayed essentially equal, consistent with JAX eliminating redundant computation during compilation. Simpler expressions compiled in similar times.

Setup costs are recorded in the JSON output but are single first-use measurements. They include per-expression cold caches and are not robust estimates of repeated construction or parsing. Symbolic Mellin derivation and imports are excluded.

## Validation and reproduction

All expressions passed JIT value comparisons, JIT parameter-gradient comparisons, independent finite-difference checks, and JIT/vmap comparisons in float64 and float32. Maximum relative value difference in the final float64 run was 1.2e-13. Float32 was a correctness smoke run, not a performance result. `uv run ruff check .` passed.

The library and project dependencies were unchanged. SymPy is an ephemeral dependency for the benchmark command.

```bash
JAX_ENABLE_X64=1 uv run --with sympy python scripts/benchmark_symbolic.py \
  --n 1024 --rounds 11 --loops 50 --output /tmp/fftloggin-symbolic-final.json

# Broader sample-size comparison
JAX_ENABLE_X64=1 uv run --with sympy python scripts/benchmark_symbolic.py \
  --output /tmp/fftloggin-symbolic-x64.json

# Float32 correctness smoke run
uv run --with sympy python scripts/benchmark_symbolic.py \
  --n 8 --rounds 2 --loops 3 --output /tmp/fftloggin-symbolic-f32.json
```
