#!/usr/bin/env python3
"""Compare generated kernels with independent direct JAX formulas.

Run: JAX_ENABLE_X64=1 uv run python scripts/benchmark_symbolic.py
Historical custom-engine measurements remain in benchmark_symbolic_results.md.
"""

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax.typing import ArrayLike

from fftloggin import BesselJKernel
from fftloggin.symbolic import from_mellin

CASES = ("base", "nested", "second_derivative", "repeated_sum")


def direct_function(case):
    def evaluate(parameters: ArrayLike, s: ArrayLike) -> jax.Array:
        mu, weight, power, scale = parameters
        s = jnp.asarray(s)
        kernel = BesselJKernel(mu)
        weighted = lambda z: weight * scale ** -(z + power) * kernel(z + power)
        if case == "base":
            return kernel(s)
        if case == "nested":
            return weighted(s) + (s - 1) * kernel(s - 1)
        if case == "second_derivative":
            return (s - 1) * (s - 2) * weighted(s - 2)
        return 8 * weighted(s)

    return evaluate


def sympy_expression(case):
    s, mu, weight, power, scale = sp.symbols("s mu weight power scale")

    def mellin(z):
        return sp.exp(
            sp.log(2) * (z - 1)
            + sp.loggamma((mu + z) / 2)
            - sp.loggamma((mu + 2 - z) / 2)
        )

    names = dict(
        zip(
            ("s", "mu", "weight", "power", "scale"),
            (s, mu, weight, power, scale),
            strict=True,
        )
    )
    names.update(M=mellin, exp=sp.exp, log=sp.log)
    weighted = "weight*exp(-(s+power)*log(scale))*M(s+power)"
    texts = {
        "base": "M(s)",
        "nested": weighted + "+(s-1)*M(s-1)",
        "second_derivative": (
            "(s-1)*(s-2)*weight*exp(-(s-2+power)*log(scale))*M(s-2+power)"
        ),
        "repeated_sum": "+".join([weighted] * 8),
    }
    expression = sp.sympify(texts[case], locals=names)
    return (mu, weight, power, scale, s), expression


def generated_function(symbols, expression, cse):
    factory = from_mellin(
        expression, symbols[-1], parameters=symbols[:-1], strip=(-sp.oo, sp.oo), cse=cse
    )

    def evaluate(parameters: ArrayLike, s: ArrayLike) -> jax.Array:
        return factory(*parameters)(s)

    return evaluate


def elapsed_ms(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def timings(functions, arguments, rounds, loops):
    """Alternate randomized timing blocks, synchronizing every invocation."""
    samples = {name: [] for name in functions}
    rng = np.random.default_rng(42)
    for _ in range(rounds):
        for name in rng.permutation(list(functions)):
            fn = functions[name]
            start = time.perf_counter()
            for _ in range(loops):
                fn(*arguments).block_until_ready()
            samples[name].append((time.perf_counter() - start) * 1e6 / loops)
    return {
        name: {
            "median_us": float(np.median(values)),
            "min_us": min(values),
            "max_us": max(values),
        }
        for name, values in samples.items()
    }


def validate(functions, parameters, samples, rtol, atol):
    """Check values, parameter gradients, finite differences, and vmap."""
    # Keep finite differences away from high-frequency cancellation in float32.
    probe = 1.1 + 1j * jnp.linspace(0.1, 0.8, 8, dtype=parameters.dtype)
    batch = jnp.stack([parameters, parameters + jnp.array([0.1, 0.2, 0.01, 0.1])])
    values, gradients, vmaps = {}, {}, {}
    for name, fn in functions.items():
        values[name] = np.asarray(jax.jit(fn)(parameters, samples))
        loss = lambda p, fn=fn: jnp.real(jnp.sum(fn(p, probe)))
        gradients[name] = np.asarray(jax.jit(jax.grad(loss))(parameters))
        vmaps[name] = np.asarray(jax.jit(jax.vmap(fn, in_axes=(0, None)))(batch, probe))
    for name in functions:
        for results in (values, gradients, vmaps):
            np.testing.assert_allclose(
                results[name], results["direct"], rtol=rtol, atol=atol
            )
    step = 1e-5 if jax.config.jax_enable_x64 else 2e-3
    fn = functions["direct"]
    finite_difference = []
    for i in range(parameters.size):
        plus = jnp.real(jnp.sum(fn(parameters.at[i].add(step), probe)))
        minus = jnp.real(jnp.sum(fn(parameters.at[i].add(-step), probe)))
        finite_difference.append(float((plus - minus) / (2 * step)))
    np.testing.assert_allclose(
        gradients["direct"],
        finite_difference,
        rtol=1e-6 if jax.config.jax_enable_x64 else 5e-3,
        atol=1e-8 if jax.config.jax_enable_x64 else 5e-3,
    )
    return {
        name: float(
            np.max(
                np.abs(value - values["direct"])
                / np.maximum(np.abs(values["direct"]), atol)
            )
        )
        for name, value in values.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 1024, 16384])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.n, args.rounds, args.loops) < 1:
        parser.error("sizes, rounds, and loops must be positive")
    x64 = jax.config.jax_enable_x64
    dtype = jnp.float64 if x64 else jnp.float32
    parameters = jnp.array([4.5, 2.0, 0.2, 1.7], dtype=dtype)
    parameters.block_until_ready()
    report = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "jax": jax.__version__,
            "sympy": sp.__version__,
            "devices": list(map(str, jax.devices())),
            "x64": x64,
        },
        "rounds": args.rounds,
        "loops": args.loops,
        "cases": [],
    }
    # Initialize backend and numerical primitives before timing expression setup.
    BesselJKernel(parameters[0])(jnp.array([1.1 + 0.3j])).block_until_ready()
    for case in args.cases:
        _, direct_setup = elapsed_ms(partial(direct_function, case))
        parsed, parse_ms = elapsed_ms(partial(sympy_expression, case))
        symbols, expression = parsed
        functions = {"direct": direct_function(case)}
        setup = {"direct": direct_setup}
        for cse in (False, True):
            name = "sympy_cse" if cse else "sympy"
            functions[name], setup[name] = elapsed_ms(
                partial(generated_function, symbols, expression, cse)
            )
        print(
            f"{case}: direct build {direct_setup:.3f} ms, "
            f"SymPy parse {parse_ms:.3f} ms, lambdify {setup['sympy']:.3f} ms, "
            f"lambdify CSE {setup['sympy_cse']:.3f} ms",
            flush=True,
        )
        for n in args.n:
            samples = 1.1 + 1j * jnp.linspace(0.1, 80, n, dtype=dtype)
            samples.block_until_ready()
            compiled, compile_times = {}, {}
            # Clear in-memory caches before each variant. Persistent cache is
            # disabled by the CLI entry point to measure fresh compilation.
            for name, fn in functions.items():
                jax.clear_caches()
                lowered, lower_ms = elapsed_ms(
                    partial(jax.jit(fn).lower, parameters, samples)
                )
                executable, compile_ms = elapsed_ms(lowered.compile)
                compile_times[name] = {
                    "trace_lower_ms": lower_ms,
                    "compile_ms": compile_ms,
                }
                compiled[name] = executable
                for _ in range(5):
                    executable(parameters, samples).block_until_ready()
            steady = timings(compiled, (parameters, samples), args.rounds, args.loops)
            errors = validate(
                functions,
                parameters,
                samples,
                2e-10 if x64 else 3e-4,
                2e-11 if x64 else 3e-5,
            )
            # Eager dispatch is measured after warming all primitive caches.
            eager_functions = dict(functions)
            for fn in eager_functions.values():
                fn(parameters, samples).block_until_ready()
            eager = timings(eager_functions, (parameters, samples), 3, 3)
            row = {
                "case": case,
                "n": n,
                "setup_ms": setup,
                "parse_ms": parse_ms,
                "compilation": compile_times,
                "jit": steady,
                "eager": eager,
                "max_relative_error": errors,
            }
            report["cases"].append(row)
            print(
                f"  n={n}: JIT us "
                + ", ".join(
                    f"{name}={stats['median_us']:.2f}" for name, stats in steady.items()
                ),
                flush=True,
            )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    jax.config.update("jax_enable_compilation_cache", False)
    main()
