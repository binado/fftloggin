"""
Tests for Mellin transform kernel classes.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.kernels import (
    BesselJKernel,
    CombinedKernel,
    Derivative,
    ShiftedKernel,
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


def test_kernel_shift_method():
    """Test Kernel.shift() method."""
    kernel = BesselJKernel(0.5)

    # Test nu=0 returns self
    shifted0 = kernel.shift(0)
    assert shifted0 is kernel

    # Test nu!=0 returns ShiftedKernel
    nu = 0.25
    shifted = kernel.shift(nu)
    assert isinstance(shifted, ShiftedKernel)
    assert shifted.base is kernel
    assert shifted.nu == nu


def test_kernel_shift_method_with_array_nu():
    """Kernel.shift() should accept batched nu with trailing singleton axis."""
    kernel = BesselJKernel(0.5)
    nu = np.array([0.1, 0.3]).reshape(-1, 1)
    shifted = kernel.shift(nu)

    assert isinstance(shifted, ShiftedKernel)
    assert shifted.base is kernel
    assert_allclose(shifted.nu, nu)


def test_shifted_kernel_domain_shifts_bounds():
    """Shifted kernel domain should shift the base domain by -nu."""
    mu = 0.5
    nu = 0.25
    kernel = BesselJKernel(mu)
    shifted = kernel.shift(nu)

    inf, sup = shifted.domain
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    assert_allclose(inf, -mu - nu)
    assert_allclose(sup, 1.5 - nu)


def test_shifted_kernel_domain_with_array_nu():
    """Shifted kernel domain should broadcast with batched nu."""
    mu = 0.5
    nu = np.array([0.1, 0.3]).reshape(-1, 1)
    kernel = BesselJKernel(mu)
    shifted = kernel.shift(nu)

    inf, sup = shifted.domain
    assert_allclose(inf, -mu - nu)
    assert_allclose(sup, 1.5 - nu)


def test_shifted_kernel_evaluation_matches_shifted_argument():
    """Shifted kernel should evaluate as base(s + nu)."""
    kernel = BesselJKernel(0.5)
    nu = 0.3
    shifted = kernel.shift(nu)
    s = np.array([0.2, 0.8, 1.1]) + 0.4j

    expected = kernel(s + nu)
    got = shifted(s)
    assert_allclose(got, expected)


def test_shifted_kernel_evaluation_matches_shifted_argument_array_nu():
    """Batched nu should broadcast in shifted kernel evaluation."""
    kernel = BesselJKernel(0.5)
    nu = np.array([0.1, 0.3]).reshape(-1, 1)
    shifted = kernel.shift(nu)
    s = np.array([0.2, 0.8, 1.1]) + 0.4j

    expected = kernel(s + nu)
    got = shifted(s)
    assert_allclose(got, expected)


def test_shifted_kernel_is_in_domain():
    """Shifted kernel domain checks should delegate via s + nu."""
    kernel = BesselJKernel(0.5)
    nu = 0.4
    shifted = kernel.shift(nu)

    assert shifted.is_in_domain(0.0) == kernel.is_in_domain(0.4)
    assert shifted.is_in_domain(1.1) == kernel.is_in_domain(1.5)
    assert shifted.is_in_domain(1.2) == kernel.is_in_domain(1.6)


def test_shifted_kernel_rejects_invalid_array_nu_shape():
    """Array nu must keep sample axis free for FFTLog broadcasting."""
    kernel = BesselJKernel(0.5)
    nu = np.array([0.1, 0.2])  # shape (2,), last axis not singleton

    with pytest.raises(ValueError, match="shape\\[-1\\] == 1"):
        kernel.shift(nu)


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
def test_bessel_kernel_is_in_domain(mu: float, s: complex | float, order: int):
    """Test that is_in_domain correctly identifies valid/invalid inputs."""
    kernel = BesselJKernel(mu)
    sr = s.real if isinstance(s, complex) else s
    if order > 0:
        kernel = kernel.derive(order)

    expected = bool((sr - order >= -mu) & (sr - order <= 1.5))
    assert kernel.is_in_domain(s) == expected


@pytest.mark.parametrize("ell", [1, 5, 10])
@pytest.mark.parametrize("s", [-11, -10.5, -5, 0 + 1j, 0 + 1j, 1 + 1j, 1.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_spherical_bessel_kernel_is_in_domain(
    ell: float, s: complex | float, order: int
):
    """Test that SphericalBesselJKernel.is_in_domain works correctly."""
    kernel = SphericalBesselJKernel(ell)
    if order > 0:
        kernel = kernel.derive(order)

    sr = s.real if isinstance(s, complex) else s
    inf, sup = kernel.domain
    inf = np.asarray(inf)
    sup = np.asarray(sup)
    expected = bool((sr >= inf) & (sr <= sup))
    assert kernel.is_in_domain(s) == expected


class TestCombinedKernel:
    def test_combined_kernel_with_shifted_kernel(self):
        """Shifted kernels should compose naturally in CombinedKernel."""
        k1 = BesselJKernel(mu=0)
        k2 = BesselJKernel(mu=1)
        nu = 0.2
        kernel = CombinedKernel([k1, k2.shift(nu)])

        s = np.array([0.5, 1.0, 1.4])
        res = kernel(s)
        assert_allclose(res[0], k1(s))
        assert_allclose(res[1], k2(s + nu))

    def test_combined_kernel_raises_on_empty_list(self):
        """Test CombinedKernel raises on empty kernel list."""
        with pytest.raises(ValueError, match="At least one kernel"):
            CombinedKernel([])

    def test_combined_kernel_forward(self):
        """Test CombinedKernel stacks results correctly."""
        k1 = BesselJKernel(mu=0)
        k2 = BesselJKernel(mu=1)
        kernel = CombinedKernel([k1, k2])

        # Case 1: Scalar s
        s = 1.0
        res = kernel(s)
        # Expected shape: (2, 1) due to FFTLog compatibility fix
        assert res.shape == (2, 1)

        # Check values
        val1 = k1(s)  # scalar
        val2 = k2(s)  # scalar
        assert_allclose(res[0, 0], val1)
        assert_allclose(res[1, 0], val2)

        # Case 2: Array s
        s_arr = np.array([0.5, 1.0, 1.4])
        res_arr = kernel(s_arr)
        # Expected shape: (2, 3)
        assert res_arr.shape == (2, 3)

        val1_arr = k1(s_arr)  # shape (3,)
        val2_arr = k2(s_arr)  # shape (3,)
        assert_allclose(res_arr[0], val1_arr)
        assert_allclose(res_arr[1], val2_arr)

    def test_combined_kernel_domain(self):
        """Test CombinedKernel domain checking."""
        # k1: (-0, 1.5)
        k1 = BesselJKernel(mu=0)
        # k2: (-1, 1.5)
        k2 = BesselJKernel(mu=1)

        kernel = CombinedKernel([k1, k2])

        # Check domain property
        inf, sup = kernel.domain
        assert_allclose(inf, [0, -1])
        assert_allclose(sup, [1.5, 1.5])

        # Check is_in_domain
        # s = 0.5 (in both)
        assert kernel.is_in_domain(0.5)

        # s = -0.5 (in k2, not k1)
        assert not kernel.is_in_domain(-0.5)

        # s = 2.0 (out both)
        assert not kernel.is_in_domain(2.0)

        # Calling with out-of-domain value still computes (no bounds checking)
        result = kernel(-0.5)
        assert result is not None

    def test_combined_kernel_flattens_nested(self):
        """Test CombinedKernel flattens nested CombinedKernel instances."""
        k1 = BesselJKernel(mu=0)
        k2 = BesselJKernel(mu=1)
        k3 = BesselJKernel(mu=2)

        # Create nested CombinedKernel
        inner = CombinedKernel([k1, k2])
        outer = CombinedKernel([inner, k3])

        # Should be flattened to 3 kernels, not 2
        assert len(outer.kernels) == 3
        assert outer.kernels[0] is k1
        assert outer.kernels[1] is k2
        assert outer.kernels[2] is k3

        # Verify forward() produces same result as flat CombinedKernel
        flat = CombinedKernel([k1, k2, k3])
        s = np.array([0.5, 1.0, 1.5])

        result_nested = outer(s)
        result_flat = flat(s)

        # Shape should be identical
        assert result_nested.shape == result_flat.shape == (3, 3)
        # Values should match (ignore NaN differences)
        assert_allclose(np.nan_to_num(result_nested), np.nan_to_num(result_flat))

    def test_combined_kernel_flattens_deeply_nested(self):
        """Test CombinedKernel flattens arbitrary nesting depths."""
        k1 = BesselJKernel(mu=0)
        k2 = BesselJKernel(mu=1)
        k3 = BesselJKernel(mu=2)
        k4 = BesselJKernel(mu=3)

        # Deep nesting: [[k1], [k2, [k3]], k4]
        deep = CombinedKernel(
            [CombinedKernel([k1]), CombinedKernel([k2, CombinedKernel([k3])]), k4]
        )

        # Should be flattened to 4 kernels
        assert len(deep.kernels) == 4
        assert all(isinstance(k, BesselJKernel) for k in deep.kernels)

    def test_combined_kernel_setter_flattens(self):
        """Test CombinedKernel.kernels setter flattens nested kernels."""
        k1 = BesselJKernel(mu=0)
        k2 = BesselJKernel(mu=1)
        k3 = BesselJKernel(mu=2)

        kernel = CombinedKernel([k1])
        kernel.kernels = [CombinedKernel([k2]), k3]

        assert len(kernel.kernels) == 2
        assert kernel.kernels[0] is k2
        assert kernel.kernels[1] is k3

    def test_combined_kernel_setter_rejects_empty(self):
        """Test CombinedKernel.kernels setter rejects empty kernel list."""
        k1 = BesselJKernel(mu=0)
        kernel = CombinedKernel([k1])

        with pytest.raises(ValueError, match="At least one kernel"):
            kernel.kernels = []
