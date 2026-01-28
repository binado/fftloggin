"""Tests for FFT backend selection in FFTLog."""

import numpy as np
from numpy.testing import assert_allclose

from fftloggin import FFTLog, NumPyFFTBackend
from fftloggin.kernels import BesselJKernel


def test_numpy_backend_matches_default_scipy_backend():
    r = np.logspace(-3, 3, 64, dtype=np.float64)
    mu = 0.7
    bias = 0.1
    kr = 1.0

    fftlog_scipy = FFTLog.from_array(
        r, kernel=BesselJKernel(mu), bias=bias, kr=kr, lowring=False
    )
    fftlog_numpy = FFTLog.from_array(
        r,
        kernel=BesselJKernel(mu),
        bias=bias,
        kr=kr,
        lowring=False,
        backend=NumPyFFTBackend(),
    )

    rng = np.random.default_rng(0)
    a = rng.normal(size=r.shape).astype(np.float64)

    out_scipy = fftlog_scipy.forward(a)
    out_numpy = fftlog_numpy.forward(a)
    assert_allclose(out_numpy, out_scipy, rtol=1e-10, atol=1e-12)

    inv_scipy = fftlog_scipy.inverse(out_scipy)
    inv_numpy = fftlog_numpy.inverse(out_numpy)
    assert_allclose(inv_numpy, inv_scipy, rtol=1e-10, atol=1e-12)
