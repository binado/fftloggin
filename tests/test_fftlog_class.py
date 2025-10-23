"""
Tests for the FFTLog class.

This module tests the object-oriented FFTLog class API including:
- FFTLog class instantiation (fht, ifht, from_r, from_k)
- forward() and inverse() methods
- compute_offset() method
- Integration with Grid base class
- Round-trip transforms
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import FFTLog, fht, fhtoffset, ifht
from fftloggin._backend_numexpr import NUMEXPR_AVAILABLE
from fftloggin.kernels import BesselJKernel


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_creation(use_numexpr):
    """Test FFTLog() constructor."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 128, 0.05, 0.5
    fftlog = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    assert fftlog.n == n
    assert fftlog.dln == dln
    assert_allclose(fftlog.mu, mu)
    assert fftlog.offset == 0.0
    assert fftlog.central == 1.0
    assert fftlog.bias == 0.0
    assert fftlog.use_numexpr == use_numexpr
    assert len(fftlog.r) == n
    assert len(fftlog.k) == n


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_from_r(use_numexpr):
    """Test FFTLog.from_r() class method."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-2, 2, 64)
    mu = 0.5
    fftlog = FFTLog.from_r(
        r, kernel=BesselJKernel(mu), offset=0.0, use_numexpr=use_numexpr
    )

    assert fftlog.n == 64
    assert_allclose(fftlog.mu, mu)
    assert_allclose(fftlog.r, r, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_from_k(use_numexpr):
    """Test FFTLog.from_k() class method."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    k = np.logspace(-3, 1, 128)
    mu = 1.0
    fftlog = FFTLog.from_k(
        k, kernel=BesselJKernel(mu), offset=0.0, use_numexpr=use_numexpr
    )

    assert fftlog.n == 128
    assert_allclose(fftlog.mu, mu)
    assert_allclose(fftlog.k, k, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_forward_method(use_numexpr):
    """Test FFTLog.forward() method."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 128, 0.05, 0.0
    fftlog = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Create test input
    a = np.exp(-((fftlog.r / 1.0) ** 2))

    # Perform forward transform using method
    A_method = fftlog.forward(a)

    # Perform forward transform using function
    A_function = fht(
        a,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Should be identical
    assert_allclose(A_method, A_function, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_inverse_method(use_numexpr):
    """Test FFTLog.inverse() method."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 128, 0.05, 0.0
    fftlog = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Create test input
    A = np.exp(-((fftlog.k / 1.0) ** 2))

    # Perform inverse transform using method
    a_method = fftlog.inverse(A)

    # Perform inverse transform using function
    a_function = ifht(
        A,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Should be identical
    assert_allclose(a_method, a_function, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_roundtrip(use_numexpr):
    """Test that forward + inverse recovers original data."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 128, 0.05, 0.5

    # Create forward transform instance
    fftlog_fwd = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Create inverse transform instance
    fftlog_inv = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Create test data
    a = np.exp(-((fftlog_fwd.r / 1.0) ** 2))

    # Round-trip transform
    A = fftlog_fwd.forward(a)
    a_recovered = fftlog_inv.inverse(A)

    # Should recover original (up to numerical precision)
    # Use atol for small values to avoid relative tolerance issues
    assert_allclose(a_recovered, a, rtol=1e-6, atol=1e-14)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_compute_offset(use_numexpr):
    """Test FFTLog.compute_offset() method."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    dln, mu, bias = 0.1, 0.5, 0.0
    fftlog = FFTLog(
        128,
        dln,
        kernel=BesselJKernel(mu),
        bias=bias,
        use_numexpr=use_numexpr,
    )

    # Compute offset using method
    offset_method = fftlog.compute_offset(initial=0.0)

    # Compute offset using function
    offset_function = fhtoffset(dln, mu, initial=0.0, bias=bias)

    # Should be identical
    assert_allclose(offset_method, offset_function, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_vectorized_mu(use_numexpr):
    """Test FFTLog with vectorized mu parameter."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln = 64, 0.1
    mu_vals = np.array([0.0, 0.5, 1.0])[:, np.newaxis]

    fftlog = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu_vals),
        offset=0.0,
        use_numexpr=use_numexpr,
    )

    # Check mu stored correctly
    assert fftlog.mu.shape == (3, 1)

    # Create test input
    a = np.exp(-((fftlog.r / 1.0) ** 2))

    # Perform transform
    A = fftlog.forward(a)

    # Should have shape (3, 64) - one transform per mu
    assert A.shape == (3, n)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_with_bias(use_numexpr):
    """Test FFTLog with non-zero bias."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu, bias = 128, 0.05, 0.0, 0.5
    fftlog = FFTLog(
        n,
        dln,
        kernel=BesselJKernel(mu),
        bias=bias,
        use_numexpr=use_numexpr,
    )

    assert fftlog.bias == bias

    # Create test input
    a = np.ones(n)

    # Transform with bias
    A_method = fftlog.forward(a)
    A_function = fht(
        a,
        dln,
        mu,
        kernel=BesselJKernel(mu),
        bias=bias,
        use_numexpr=use_numexpr,
    )

    assert_allclose(A_method, A_function, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_axis_parameter(use_numexpr):
    """Test FFTLog with axis parameter."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 64, 0.1, 0.0
    fftlog = FFTLog(n, dln, kernel=BesselJKernel(mu), use_numexpr=use_numexpr)

    # Create 2D test input
    a = np.ones((5, n))

    # Transform along axis=1
    A = fftlog.forward(a, axis=1)

    assert A.shape == (5, n)


def test_fftlog_kernel_validation():
    """Test that FFTLog validates kernel arguments."""
    # No kernel provided
    with pytest.raises(
        ValueError, match="Either kernel or log_kernel must be provided"
    ):
        FFTLog(64, 0.1, 0.0)

    # Both kernels provided
    with pytest.raises(
        ValueError, match="Only one of kernel or log_kernel can be provided"
    ):
        FFTLog(
            64,
            0.1,
            0.0,
            kernel=BesselJKernel(0.0),
        )


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_properties(use_numexpr):
    """Test that FFTLog has all expected properties."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 128, 0.05, 0.5
    fftlog = FFTLog(
        n,
        dln,
        mu,
        offset=0.0,
        kernel=BesselJKernel(mu),
        central=2.0,
        use_numexpr=use_numexpr,
    )

    # Test properties exist
    assert hasattr(fftlog, "r")
    assert hasattr(fftlog, "k")
    assert hasattr(fftlog, "n")
    assert hasattr(fftlog, "dln")
    assert hasattr(fftlog, "offset")
    assert hasattr(fftlog, "central")
    assert hasattr(fftlog, "mu")
    assert hasattr(fftlog, "bias")
    assert hasattr(fftlog, "use_numexpr")

    # Test property values
    assert fftlog.n == n
    assert fftlog.dln == dln
    assert fftlog.offset == 0.0
    assert fftlog.central == 2.0
    assert_allclose(fftlog.mu, mu)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_repr(use_numexpr):
    """Test FFTLog string representation."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    mu = 0.5
    fftlog = FFTLog(
        64,
        0.1,
        0.5,
        offset=0.0,
        kernel=BesselJKernel(mu),
        bias=0.5,
        use_numexpr=use_numexpr,
    )

    repr_str = repr(fftlog)

    assert "FFTLog" in repr_str
    assert "n=64" in repr_str
    assert "mu=" in repr_str
    assert "bias=0.5" in repr_str
    assert f"use_numexpr={use_numexpr}" in repr_str


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_with_custom_central(use_numexpr):
    """Test FFTLog with custom central value."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu, central = 64, 0.1, 0.0, 5.0
    fftlog = FFTLog(
        n,
        dln,
        mu,
        kernel=BesselJKernel(mu),
        central=central,
        use_numexpr=use_numexpr,
    )

    assert fftlog.central == central

    # Check that central is applied to grids
    ic = (n - 1) // 2
    assert_allclose(fftlog.r[ic], central, rtol=1e-10)
    assert_allclose(fftlog.k[ic], 1.0 / central, rtol=1e-10)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_multidimensional_transform(use_numexpr):
    """Test FFTLog with multidimensional arrays."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 32, 0.1, 0.0
    fftlog = FFTLog(n, dln, kernel=BesselJKernel(mu), use_numexpr=use_numexpr)

    # 3D input array
    a = np.random.randn(4, 5, n)

    # Transform along last axis
    A = fftlog.forward(a, axis=-1)

    assert A.shape == (4, 5, n)


def test_fftlog_properties_readonly():
    """Test that FFTLog properties are read-only."""
    fftlog = FFTLog(64, 0.1, kernel=BesselJKernel(0.5))

    # Grid properties should be read-only
    with pytest.raises(AttributeError):
        fftlog.n = 128

    with pytest.raises(AttributeError):
        fftlog.dln = 0.2

    with pytest.raises(AttributeError):
        fftlog.offset = 1.0

    # FFTLog-specific properties should also be read-only
    with pytest.raises(AttributeError):
        fftlog.mu = 1.0

    with pytest.raises(AttributeError):
        fftlog.bias = 1.0

    with pytest.raises(AttributeError):
        fftlog.use_numexpr = True
