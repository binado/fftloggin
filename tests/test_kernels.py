"""
Tests for Mellin transform kernel classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.kernels import BesselJKernel, Derivative


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


@pytest.mark.parametrize(
    "mu_shape,s_shape,expected_shape",
    [
        # (mu_shape, s_shape, expected_broadcast_shape)
        ((), (), ()),  # scalar mu, scalar s
        ((), (4,), (4,)),  # scalar mu, 1d s
        ((3,), (), (3,)),  # 1d mu, scalar s
        ((3,), (4,), (3, 4)),  # 1d mu, 1d s (broadcast)
        ((2, 3), (), (2, 3)),  # 2d mu, scalar s
        ((2, 3), (4,), (2, 3, 4)),  # 2d mu, 1d s (broadcast)
    ],
)
def test_bessel_kernel_vectorized_mu(mu_shape, s_shape, expected_shape):
    """Test BesselJKernel with vectorized mu and s parameters.

    Tests that kernel.forward(s) returns shape (*mu.shape, *s.shape).
    """
    # Create mu with specified shape
    if mu_shape == ():
        mu = 0.5
    else:
        mu = np.linspace(0.1, 0.9, np.prod(mu_shape)).reshape(mu_shape)

    # Create s with specified shape
    if s_shape == ():
        s = 1.0
    else:
        s = np.linspace(0.5, 1.5, np.prod(s_shape)).reshape(s_shape)

    kernel = BesselJKernel(mu)

    # Forward should return correct broadcast shape
    result = kernel.forward(s)
    assert result.shape == expected_shape

    # Result should be finite
    assert np.all(np.isfinite(result))


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
