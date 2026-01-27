"""
Tests for Mellin transform kernel classes.
"""

from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.kernels import (
    ArgumentOutOfDomainError,
    BesselJKernel,
    Derivative,
    Kernel,
    SphericalBesselJKernel,
)


def test_bessel_kernel_domain():
    """Test that BesselJKernel has correct domain of convergence."""
    mu = 0.5
    kernel = BesselJKernel(mu)
    inf, sup = kernel.domain

    # Domain should be (-mu, 1.5)
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    assert_allclose(inf, -mu)
    assert_allclose(sup, 1.5)


def test_spherical_bessel_kernel_domain():
    """Test that SphericalBesselJKernel has correct domain of convergence."""
    ell = 1
    kernel = SphericalBesselJKernel(ell)

    # Domain should be (-ell, 2.0)
    inf, sup = kernel.domain
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    assert_allclose(inf, -ell)
    assert_allclose(sup, 2.0)


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

    Tests that kernel(s) returns shape (*mu.shape, *s.shape).
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
    result = kernel(s)
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


def test_derivative_domain_shifts_bounds():
    """Derivative domain should shift the base kernel domain by order."""
    mu = 0.5
    order = 2
    kernel = BesselJKernel(mu)
    d_kernel = kernel.derive(order)

    inf, sup = d_kernel.domain
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    assert_allclose(inf, -mu + order)
    assert_allclose(sup, 1.5 + order)


def test_derivative_domain_preserves_none_and_inf():
    """Derivative domain should preserve None/inf bounds."""

    class DummyKernel(Kernel):
        @property
        def domain(self) -> tuple[None, float]:
            return (None, np.inf)

        def forward(self, s: np.ndarray) -> np.ndarray:
            return np.asarray(s)

    d_kernel = Derivative(DummyKernel(), order=3)
    inf, sup = d_kernel.domain
    assert inf is None
    assert np.isposinf(sup)


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


@pytest.mark.parametrize("mu", [-1, 1, 5, 10])
@pytest.mark.parametrize("s", [-11, -10.5, -5, 0 + 1j, 0 + 1j, 1 + 1j, 1.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_bessel_kernel_bounds_checking(mu: float, s: complex | float, order: int):
    """Test that BesselJKernel.__call__() correctly checks bounds."""
    kernel = BesselJKernel(mu, check_bounds=True)
    sr = s.real if isinstance(s, complex) else s
    if order > 0:
        kernel = kernel.derive(order)

    is_in_domain = (sr - order >= -mu) & (sr - order <= 1.5)

    # s outside domain should raise
    context = (
        nullcontext()
        if is_in_domain
        else pytest.raises(ArgumentOutOfDomainError, match="outside domain")
    )
    with context:
        kernel(s)


@pytest.mark.parametrize("ell", [1, 5, 10])
@pytest.mark.parametrize("s", [-11, -10.5, -5, 0 + 1j, 0 + 1j, 1 + 1j, 1.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_spherical_bessel_kernel_bounds_checking(
    ell: float, s: complex | float, order: int
):
    """Test that SphericalBesselJKernel.__call__() correctly checks bounds."""
    kernel = SphericalBesselJKernel(ell, check_bounds=True)
    if order > 0:
        kernel = kernel.derive(order)

    sr = s.real if isinstance(s, complex) else s
    inf, sup = kernel.domain
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    is_in_domain = (sr >= inf) & (sr <= sup)

    # s outside domain should raise
    context = (
        nullcontext()
        if is_in_domain
        else pytest.raises(ArgumentOutOfDomainError, match="outside domain")
    )
    with context:
        kernel(s)


def test_bessel_kernel_skips_bounds_checking():
    """Test that BesselJKernel.__call__() skips bounds checking."""
    mu = 0.5
    kernel = BesselJKernel(mu, check_bounds=False)

    # s outside domain should not raise
    kernel(-mu - 1)  # Below lower bound

    kernel(2.0)  # Above upper bound


def test_spherical_bessel_kernel_skips_bounds_checking():
    """Test that SphericalBesselJKernel.__call__() skips bounds checking."""
    ell = 1
    kernel = SphericalBesselJKernel(ell, check_bounds=False)

    # s outside domain should not raise
    mu = ell + 0.5
    kernel(-mu - 1)  # Below lower bound
    kernel(2.0)  # Above upper bound
