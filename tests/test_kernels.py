"""
Tests for Mellin transform kernel classes.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import fht, ifht, FFTLog
from fftloggin._backend_numexpr import NUMEXPR_AVAILABLE
from fftloggin.kernels import BesselJKernel, Derivative


# test function, analytical Hankel transform is of the same form
def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_bessel_kernel_forward_transform(use_numexpr):
    """Test that BesselJKernel works for forward transform."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Transform with BesselJKernel
    A = fht(
        a,
        dln,
        mu,
        offset=offset,
        bias=bias,
        use_numexpr=use_numexpr,
    )

    # Expected output from original test
    expected = [
        -0.1159922613593045e-02,
        +0.1625822618458832e-02,
        -0.1949518286432330e-02,
        +0.3789220182554077e-02,
        +0.5093959119952945e-03,
        +0.2785387803618774e-01,
        +0.9944952700848897e-01,
        +0.4599202164586588e00,
        +0.3157462160881342e00,
        -0.8201236844404755e-03,
        -0.7834031308271878e-03,
        +0.3931444945110708e-03,
        -0.2697710625194777e-03,
        +0.3568398050238820e-03,
        -0.5554454827797206e-03,
        +0.8286331026468585e-03,
    ]
    expected = np.asarray(expected, dtype=np.float64)
    assert_allclose(A, expected)


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("n", [64, 63])
def test_fht_identity(n, use_numexpr):
    """Test that ifht is the inverse of fht with BesselJKernel."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(n))
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)
    offset = 0.0
    bias = 0.0

    A = fht(
        a,
        dln,
        mu,
        offset=offset,
        bias=bias,
        use_numexpr=use_numexpr,
    )
    a_ = ifht(
        A,
        dln,
        mu,
        offset=offset,
        bias=bias,
        use_numexpr=use_numexpr,
    )

    assert_allclose(a_, a, rtol=1.5e-7)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_vectorized_fht(use_numexpr):
    """Test vectorized transforms with BesselJKernel."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n = 16
    r = np.logspace(-4, 4, n)
    dln = math.log(r[1] / r[0])

    # Test with vectorized mu
    mu = np.linspace(0.1, 0.3, 3).reshape(3, 1, 1)
    a = f(r, mu)
    offset = 0.0
    bias = 0.0
    out = fht(
        a,
        dln,
        mu,
        offset,
        bias,
        use_numexpr=use_numexpr,
    )
    assert out.shape == (3, 1, n)


def test_bessel_kernel_strip():
    """Test that BesselJKernel has correct strip of convergence."""
    mu = 0.5
    kernel = BesselJKernel(mu)
    inf, sup = kernel.strip

    # Strip should be (-mu, 1.5)
    assert_allclose(inf, -mu)
    assert_allclose(sup, 1.5)


def test_bessel_kernel_forward():
    """Test BesselJKernel.forward() computation."""
    mu = 0.5
    kernel = BesselJKernel(mu)

    # Test at s=1 (should be in strip of convergence)
    s = 1.0
    result = kernel.forward(s)

    # Result should be a scalar
    assert np.isscalar(result) or result.shape == ()
    # Result should be finite
    assert np.isfinite(result)


def test_bessel_kernel_vectorized_mu():
    """Test BesselJKernel with vectorized mu parameter."""
    mu_vals = np.array([0, 0.5, 1.0])
    kernel = BesselJKernel(mu_vals)

    # Strip should be vectorized
    inf, sup = kernel.strip
    assert_allclose(inf.ravel(), -mu_vals)
    assert_allclose(sup, 1.5)

    # Forward should work with vectorized mu
    s = 1.0
    result = kernel.forward(s)
    assert result.shape == (3, 1)


def test_derivative_kernel():
    """Test Derivative kernel wrapper."""
    mu = 0.5
    base_kernel = BesselJKernel(mu)

    # Create first derivative
    d_kernel = Derivative(base_kernel, order=1)

    # Strip should be shifted by order
    inf_base, sup_base = base_kernel.strip
    inf_d, sup_d = d_kernel.strip
    assert_allclose(inf_d, inf_base + 1)
    assert_allclose(sup_d, sup_base + 1)


def test_kernel_derive_method():
    """Test Kernel.derive() method."""
    mu = 0.5
    kernel = BesselJKernel(mu)

    # Test order=0 returns self
    d0_kernel = kernel.derive(0)
    assert d0_kernel is kernel

    # Test order=1 returns Derivative
    d1_kernel = kernel.derive(1)
    assert isinstance(d1_kernel, Derivative)
    assert d1_kernel.order == 1

    # Test order=2 returns Derivative
    d2_kernel = kernel.derive(2)
    assert isinstance(d2_kernel, Derivative)
    assert d2_kernel.order == 2


def test_derivative_invalid_order():
    """Test that Derivative raises for invalid order."""
    kernel = BesselJKernel(0.5)

    with pytest.raises(
        ValueError, match="Expected derivative order to be an integer greater"
    ):
        Derivative(kernel, order=0)

    with pytest.raises(
        ValueError, match="Expected derivative order to be an integer greater"
    ):
        Derivative(kernel, order=-1)


def test_bessel_kernel_bounds_checking():
    """Test that BesselJKernel.forward() checks bounds."""
    mu = 0.5
    kernel = BesselJKernel(mu)

    # s outside strip should raise
    with pytest.raises(ValueError, match="Input array outside strip"):
        kernel.forward(-mu - 1)  # Below lower bound

    with pytest.raises(ValueError, match="Input array outside strip"):
        kernel.forward(2.0)  # Above upper bound


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fftlog_with_derivative_kernel(use_numexpr):
    """Test FFTLog with derivative kernel."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n, dln, mu = 64, 0.1, 0.5

    # Create derivative kernel
    base_kernel = BesselJKernel(mu)
    deriv_kernel = base_kernel.derive(1)

    # Create FFTLog with derivative kernel
    fftlog = FFTLog(n, dln, deriv_kernel, use_numexpr=use_numexpr)

    # Generate test input
    a = np.exp(-((fftlog.r / 1.0) ** 2))

    # Transform should work
    A = fftlog.forward(a)

    # Result should have correct shape
    assert A.shape == (n,)
    assert np.all(np.isfinite(A))
