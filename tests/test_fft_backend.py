"""Tests for FFT backend selection in FFTLog."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import (
    FFTLog,
    FFTWorkspace,
    NumPyFFTBackend,
    SciPyFFTBackend,
)
from fftloggin.kernels import BesselJKernel


def test_numpy_backend_matches_scipy_backend():
    r = np.logspace(-3, 3, 64, dtype=np.float64)
    mu = 0.7
    bias = 0.1
    kr = 1.0

    fftlog_scipy = FFTLog.from_array(
        r,
        kernel=BesselJKernel(mu),
        bias=bias,
        kr=kr,
        lowring=False,
        backend=SciPyFFTBackend(),
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


def test_backend_out_parameter_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.normal(size=64).astype(np.float64)
    backend = NumPyFFTBackend()

    expected_rfft = np.fft.rfft(a)
    out_rfft = np.empty_like(expected_rfft)
    got_rfft = backend.rfft(a, out=out_rfft)
    assert got_rfft is out_rfft
    assert_allclose(got_rfft, expected_rfft, rtol=1e-12, atol=1e-12)

    expected_irfft = np.fft.irfft(expected_rfft, n=a.shape[-1])
    out_irfft = np.empty_like(a)
    got_irfft = backend.irfft(expected_rfft, n=a.shape[-1], out=out_irfft)
    assert got_irfft is out_irfft
    assert_allclose(got_irfft, expected_irfft, rtol=1e-12, atol=1e-12)


def test_workspace_is_accepted_by_backends():
    rng = np.random.default_rng(2)
    a = rng.normal(size=32).astype(np.float64)
    backend = SciPyFFTBackend()
    workspace = FFTWorkspace()

    expected = np.fft.rfft(a)
    got = backend.rfft(a, workspace=workspace)
    assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_pyfftw_backend_defaults_when_available():
    pyfftw = pytest.importorskip("pyfftw")
    _ = pyfftw
    from fftloggin import PyFFTWBackend

    backend = PyFFTWBackend()
    assert isinstance(backend, PyFFTWBackend)
