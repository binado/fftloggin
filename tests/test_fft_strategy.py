"""Tests for FFTLog execution strategies."""

import numpy as np
from numpy.testing import assert_allclose

from fftloggin import FFTLog, OptimizedStrategy, SimpleStrategy
from fftloggin.kernels import BesselJKernel


def test_simple_and_optimized_strategies_match():
    r = np.logspace(-3, 3, 128, dtype=np.float64)
    mu = 0.7
    bias = 0.2
    kr = 1.0

    fftlog_simple = FFTLog.from_array(
        r,
        kernel=BesselJKernel(mu),
        bias=bias,
        kr=kr,
        lowring=False,
        strategy=SimpleStrategy(),
    )
    fftlog_optimized = FFTLog.from_array(
        r,
        kernel=BesselJKernel(mu),
        bias=bias,
        kr=kr,
        lowring=False,
        strategy=OptimizedStrategy(),
    )

    rng = np.random.default_rng(123)
    a = rng.normal(size=r.shape).astype(np.float64)

    out_simple = fftlog_simple.forward(a)
    out_optimized = fftlog_optimized.forward(a)
    assert_allclose(out_optimized, out_simple, rtol=1e-10, atol=1e-12)

    inv_simple = fftlog_simple.inverse(out_simple)
    inv_optimized = fftlog_optimized.inverse(out_optimized)
    assert_allclose(inv_optimized, inv_simple, rtol=1e-10, atol=1e-12)
