#!/usr/bin/env python3
"""Measure JAX FFTLog compile and steady-state execution times separately."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from fftloggin import BesselJKernel, forward, inverse


def _measure(name: str, fn, args: tuple, repeats: int, warmup: int):
    t0 = time.perf_counter()
    result = fn(*args)
    result.block_until_ready()
    compile_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        fn(*args).block_until_ready()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    print(
        f"{name:8s} compile={compile_ms:9.3f} ms  "
        f"steady mean={np.mean(samples):9.3f} ms  p95={np.percentile(samples, 95):9.3f} ms"
    )
    return result


def _case(n: int, batch: str, dtype):
    r = jnp.geomspace(1e-4, 1e4, n, dtype=dtype)
    dlog = jnp.log(r[-1] / r[0]) / (n - 1)
    a = r**1.3 * jnp.exp(-(r**2) / 2)
    if batch == "params":
        biases = jnp.array([0.0, 0.2, 0.4], dtype=dtype)
        log_krs = jnp.log(jnp.array([0.5, 1.0, 2.0], dtype=dtype))
        f = jax.jit(
            jax.vmap(
                lambda bias, log_kr: forward(
                    a, BesselJKernel(0.3), dlog=dlog, bias=bias, log_kr=log_kr
                )
            )
        )
        g = jax.jit(
            jax.vmap(
                lambda A, bias, log_kr: inverse(
                    A, BesselJKernel(0.3), dlog=dlog, bias=bias, log_kr=log_kr
                )
            )
        )
        return f, (biases, log_krs), g, lambda A: (A, biases, log_krs)
    if batch == "kernel":
        mus = jnp.array([0.1, 0.3, 0.5], dtype=dtype)
        f = jax.jit(jax.vmap(lambda mu: forward(a, BesselJKernel(mu), dlog=dlog)))
        g = jax.jit(jax.vmap(lambda A, mu: inverse(A, BesselJKernel(mu), dlog=dlog)))
        return f, (mus,), g, lambda A: (A, mus)
    kernel = BesselJKernel(0.3)
    return (
        jax.jit(lambda x: forward(x, kernel, dlog=dlog)),
        (a,),
        jax.jit(lambda A: inverse(A, kernel, dlog=dlog)),
        lambda A: (A,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--batch", choices=("none", "params", "kernel"), default="none")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--mode", choices=("forward", "inverse", "both"), default="both"
    )
    args = parser.parse_args()
    if args.dtype == "float64" and not jax.config.jax_enable_x64:
        parser.error("float64 requires JAX_ENABLE_X64=1")

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    for n in args.n:
        if n < 2:
            parser.error("n must be at least 2")
        forward_fn, forward_args, inverse_fn, inverse_args = _case(n, args.batch, dtype)
        print(f"n={n}, batch={args.batch}, dtype={args.dtype}")
        if args.mode in ("forward", "both"):
            transformed = _measure(
                "forward", forward_fn, forward_args, args.repeats, args.warmup
            )
        else:
            transformed = forward_fn(*forward_args)
            transformed.block_until_ready()
        if args.mode in ("inverse", "both"):
            _measure(
                "inverse",
                inverse_fn,
                inverse_args(transformed),
                args.repeats,
                args.warmup,
            )


if __name__ == "__main__":
    main()
