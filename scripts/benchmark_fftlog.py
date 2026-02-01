#!/usr/bin/env python3
"""Micro-benchmarks for FFTLog forward/inverse performance."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fftloggin import FFTLog, NumPyFFTBackend, PyFFTWBackend, SciPyFFTBackend
from fftloggin.kernels import BesselJKernel


@dataclass(frozen=True)
class BenchResult:
    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    std_ms: float


def _f(r: npt.NDArray, mu: npt.NDArray | float) -> npt.NDArray:
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


def _timeit(name: str, fn, repeats: int, warmup: int) -> BenchResult:
    for _ in range(warmup):
        fn()

    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1e3

    return BenchResult(
        name=name,
        mean_ms=float(times.mean()),
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        std_ms=float(times.std(ddof=1)) if repeats > 1 else 0.0,
    )


def _make_case(n: int, batch: str, dtype: np.dtype, backend):
    r = np.logspace(-4, 4, n, dtype=dtype)

    if batch == "params":
        mu = 0.3
        kr = np.array([0.5, 1.0, 2.0], dtype=dtype).reshape(-1, 1)
        bias = np.array([0.0, 0.2, 0.4], dtype=dtype).reshape(-1, 1)
        fftlog = FFTLog.from_array(
            r,
            kernel=BesselJKernel(mu),
            bias=bias,
            kr=kr,
            lowring=False,
            backend=backend,
        )
        a = _f(r, mu)
        label = "batch=params"
    elif batch == "kernel":
        mu = np.linspace(0.1, 0.5, 3, dtype=dtype).reshape(-1, 1)
        fftlog = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), kr=1.0, lowring=False, backend=backend
        )
        a = _f(r, mu)
        label = "batch=kernel"
    else:
        mu = 0.3
        fftlog = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), kr=1.0, lowring=False, backend=backend
        )
        a = _f(r, mu)
        label = "batch=none"

    return fftlog, a, label


def _print_results(title: str, results: list[BenchResult]) -> None:
    name_width = max(8, *(len(r.name) for r in results))
    print(title)
    print(
        f"{'name':<{name_width}}  {'mean_ms':>10}  {'p50_ms':>10}  {'p95_ms':>10}  {'std_ms':>10}"
    )
    for r in results:
        print(
            f"{r.name:<{name_width}}  {r.mean_ms:10.3f}  {r.p50_ms:10.3f}  "
            f"{r.p95_ms:10.3f}  {r.std_ms:10.3f}"
        )
    print()


def _run_benchmarks(args: argparse.Namespace) -> None:
    dtype = np.dtype(args.dtype)
    if args.backend == "numpy":
        backend = NumPyFFTBackend()
    elif args.backend == "pyfftw":
        backend = PyFFTWBackend()
    else:
        backend = SciPyFFTBackend()

    if args.a_shape is not None:
        if args.batch != "none":
            raise ValueError("--a-shape requires --batch none.")
        if len(args.n) != 1:
            raise ValueError("--a-shape requires exactly one --n value.")
        n = args.n[0]
        if args.a_shape[-1] != n:
            raise ValueError(
                f"--a-shape last dimension must match n={n}, got {args.a_shape[-1]}."
            )
        fftlog, _, _ = _make_case(n, "none", dtype, backend)
        rng = np.random.default_rng(0)
        a = rng.standard_normal(args.a_shape).astype(dtype, copy=False)
        label = f"batch=custom shape={args.a_shape}"
        results = []

        if args.mode in ("forward", "both"):
            results.append(
                _timeit("forward", lambda: fftlog.forward(a), args.repeats, args.warmup)
            )

        if args.mode in ("inverse", "both"):
            ak = fftlog.forward(a)
            results.append(
                _timeit(
                    "inverse", lambda: fftlog.inverse(ak), args.repeats, args.warmup
                )
            )

        title = f"n={n} ({label}, dtype={dtype}, backend={args.backend})"
        _print_results(title, results)
        return

    for n in args.n:
        fftlog, a, label = _make_case(n, args.batch, dtype, backend)
        results = []

        if args.mode in ("forward", "both"):
            results.append(
                _timeit("forward", lambda: fftlog.forward(a), args.repeats, args.warmup)
            )

        if args.mode in ("inverse", "both"):
            ak = fftlog.forward(a)
            results.append(
                _timeit(
                    "inverse", lambda: fftlog.inverse(ak), args.repeats, args.warmup
                )
            )

        title = f"n={n} ({label}, dtype={dtype}, backend={args.backend})"
        _print_results(title, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FFTLog forward/inverse.")
    parser.add_argument("--n", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--batch", choices=["none", "params", "kernel"], default="none")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument(
        "--backend", choices=["scipy", "numpy", "pyfftw"], default="scipy"
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["forward", "inverse", "both"], default="both"
    )
    parser.add_argument(
        "--a-shape",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        default=None,
        help="Override input array shape (comma-separated, last dim must match n).",
    )
    parser.add_argument(
        "--memray",
        nargs="?",
        const="memray.bin",
        default=None,
        help="Enable memray profiling (optional output path, default: memray.bin).",
    )

    args = parser.parse_args()
    if args.memray is None:
        _run_benchmarks(args)
        return

    try:
        import memray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "memray is not installed. Install with `uv add -g benchmark memray` "
            "or add it via your package manager."
        ) from exc

    with memray.Tracker(args.memray):
        _run_benchmarks(args)


if __name__ == "__main__":
    main()
