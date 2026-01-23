"""
Tests for Mellin transform kernel classes.
"""

from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.kernels import (
    BesselJKernel,
    CombinedKernel,
    Derivative,
    SphericalBesselJKernel,
)


def test_bessel_kernel_strip():
    """Test that BesselJKernel has correct strip of convergence."""
    mu = 0.5
    kernel = BesselJKernel(mu)
    inf, sup = kernel.strip

    # Strip should be (-mu, 1.5)
    assert_allclose(inf, -mu)
    assert_allclose(sup, 1.5)


def test_spherical_bessel_kernel_strip():
    """Test that SphericalBesselJKernel has correct strip of convergence."""
    ell = 1
    kernel = SphericalBesselJKernel(ell)

    # Strip should be (-mu, 1.5)
    mu = ell + 0.5
    inf, sup = kernel.strip
    assert_allclose(inf, -mu)
    assert_allclose(sup, 1.5)


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


@pytest.mark.parametrize("mu", [0, 0.5, 1, 2])
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_derivative_strip_shifted(mu, order):
    """Test that Derivative.strip is correctly shifted by derivative order."""
    kernel = BesselJKernel(mu)
    base_inf, base_sup = kernel.strip

    # Get derivative
    d_kernel = kernel.derive(order)
    d_inf, d_sup = d_kernel.strip

    # Strip should be shifted by order: (a, b) -> (a + order, b + order)
    assert_allclose(d_inf, base_inf + order)
    assert_allclose(d_sup, base_sup + order)


def test_derivative_chained_vs_direct():
    """Test that chained derivatives match direct higher-order derivatives."""
    kernel = BesselJKernel(0.5)

    # Chained: derive(1).derive(1)
    chained = kernel.derive(1).derive(1)

    # Direct: derive(2)
    direct = kernel.derive(2)

    # Both should have same strip
    assert_allclose(chained.strip[0], direct.strip[0])
    assert_allclose(chained.strip[1], direct.strip[1])

    # Both should give same results
    s = 2.5
    assert_allclose(chained.forward(s), direct.forward(s))


def test_derivative_strip_vectorized_kernel():
    """Test Derivative.strip with vectorized mu parameter."""
    mu_values = np.array([0, 1, 2])
    kernel = BesselJKernel(mu_values)

    base_inf, base_sup = kernel.strip
    # base_inf should be array([-0, -1, -2])
    # base_sup should be array([1.5, 1.5, 1.5])

    d1 = kernel.derive(1)
    d1_inf, d1_sup = d1.strip

    # Should be shifted by 1
    assert_allclose(d1_inf, base_inf + 1)  # [1, 0, -1]
    assert_allclose(d1_sup, base_sup + 1)  # [2.5, 2.5, 2.5]

    d2 = kernel.derive(2)
    d2_inf, d2_sup = d2.strip

    # Should be shifted by 2
    assert_allclose(d2_inf, base_inf + 2)  # [2, 1, 0]
    assert_allclose(d2_sup, base_sup + 2)  # [3.5, 3.5, 3.5]


def test_derivative_is_in_strip_consistency():
    """Test that Derivative.is_in_strip is consistent with Derivative.strip."""
    kernel = BesselJKernel(0.5)
    d_kernel = kernel.derive(2)

    inf, sup = d_kernel.strip

    # Values inside strip
    assert d_kernel.is_in_strip(inf + 0.1)
    assert d_kernel.is_in_strip(sup - 0.1)
    assert d_kernel.is_in_strip((inf + sup) / 2)

    # Values outside strip
    assert not d_kernel.is_in_strip(inf - 0.1)
    assert not d_kernel.is_in_strip(sup + 0.1)


@pytest.mark.parametrize("mu", [-1, 1, 5, 10])
@pytest.mark.parametrize("s", [-11, -10.5, -5, 0 + 1j, 0 + 1j, 1 + 1j, 1.5])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_bessel_kernel_bounds_checking(mu: float, s: complex | float, order: int):
    """Test that BesselJKernel.__call__() correctly checks bounds."""
    kernel = BesselJKernel(mu, check_bounds=True)
    sr = s.real if isinstance(s, complex) else s
    if order > 0:
        kernel = kernel.derive(order)

    is_in_strip = (sr - order >= -mu) & (sr - order <= 1.5)

    # s outside strip should raise
    context = (
        nullcontext()
        if is_in_strip
        else pytest.raises(ValueError, match="Input array outside strip")
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
    mu = ell + 0.5
    is_in_strip = (sr - order - 0.5 >= -mu) & (sr - order - 0.5 <= 1.5)

    # s outside strip should raise
    context = (
        nullcontext()
        if is_in_strip
        else pytest.raises(ValueError, match="Input array outside strip")
    )
    with context:
        kernel(s)


def test_bessel_kernel_skips_bounds_checking():
    """Test that BesselJKernel.__call__() skips bounds checking."""
    mu = 0.5
    kernel = BesselJKernel(mu, check_bounds=False)

    # s outside strip should not raise
    kernel(-mu - 1)  # Below lower bound

    kernel(2.0)  # Above upper bound


def test_spherical_bessel_kernel_skips_bounds_checking():
    """Test that SphericalBesselJKernel.__call__() skips bounds checking."""
    ell = 1
    kernel = SphericalBesselJKernel(ell, check_bounds=False)

    # s outside strip should not raise
    mu = ell + 0.5
    kernel(-mu - 1)  # Below lower bound
    kernel(2.0)  # Above upper bound


# CombinedKernel tests


def test_combined_kernel_empty_list():
    """Test that CombinedKernel raises for empty kernel list."""
    with pytest.raises(ValueError, match="requires at least one kernel"):
        CombinedKernel([])


def test_combined_kernel_single_kernel():
    """Test CombinedKernel with a single kernel."""
    kernel = CombinedKernel([BesselJKernel(0)])
    result = kernel.forward(1.0)

    # Should have shape (1,) for scalar s
    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_combined_kernel_multiple_kernels():
    """Test CombinedKernel with multiple kernels."""
    kernels = [BesselJKernel(0), BesselJKernel(1), BesselJKernel(2)]
    combined = CombinedKernel(kernels)

    s = 1.0
    result = combined.forward(s)

    # Should stack along axis 0
    assert result.shape == (3,)

    # Each element should match individual kernel
    for i, k in enumerate(kernels):
        assert_allclose(result[i], k.forward(s))


def test_combined_kernel_with_array_s():
    """Test CombinedKernel with array input."""
    kernels = [BesselJKernel(0), BesselJKernel(1)]
    combined = CombinedKernel(kernels)

    s = np.array([0.5, 1.0, 1.5])
    result = combined.forward(s)

    # Should have shape (len(kernels), len(s))
    assert result.shape == (2, 3)

    # Verify each row matches individual kernel
    assert_allclose(result[0], kernels[0].forward(s))
    assert_allclose(result[1], kernels[1].forward(s))


def test_combined_kernel_strip_intersection():
    """Test that CombinedKernel strip is intersection of component strips."""
    # BesselJKernel(0) has strip (-0, 1.5)
    # BesselJKernel(1) has strip (-1, 1.5)
    # Intersection should be (0, 1.5) - most restrictive
    combined = CombinedKernel([BesselJKernel(0), BesselJKernel(1)])

    inf, sup = combined.strip
    assert_allclose(inf, 0.0)  # max(-0, -1) = 0
    assert_allclose(sup, 1.5)


def test_combined_kernel_strip_with_derivative():
    """Test CombinedKernel strip with derivative kernels."""
    # BesselJKernel(0) has strip (-0, 1.5)
    # BesselJKernel(0).derive(1) shifts strip by 1: (-0+1, 1.5+1) = (1, 2.5)
    # Intersection should be (1, 1.5)
    base = BesselJKernel(0)
    combined = CombinedKernel([base, base.derive(1)])

    inf, sup = combined.strip

    # The derivative shifts the strip: effective strip is (1, 2.5)
    # Intersection with base (0, 1.5) is (1, 1.5)
    assert_allclose(inf, 1.0)
    assert_allclose(sup, 1.5)


def test_combined_kernel_derive():
    """Test CombinedKernel.derive() method."""
    kernels = [BesselJKernel(0), BesselJKernel(1)]
    combined = CombinedKernel(kernels)

    # Test order=0 returns self
    d0 = combined.derive(0)
    assert d0 is combined

    # Test order=1
    d1 = combined.derive(1)
    assert isinstance(d1, CombinedKernel)
    assert len(d1.kernels) == 2

    # Each component should be a Derivative
    for k in d1.kernels:
        assert isinstance(k, Derivative)
        assert k.order == 1

    # Test order=2
    d2 = combined.derive(2)
    for k in d2.kernels:
        assert isinstance(k, Derivative)
        assert k.order == 2


def test_combined_kernel_bounds_checking():
    """Test CombinedKernel bounds checking."""
    combined = CombinedKernel([BesselJKernel(0), BesselJKernel(1)], check_bounds=True)

    # Strip intersection is (0, 1.5)
    # s = 1.0 is inside
    result = combined(1.0)
    assert result.shape == (2,)

    # s = -0.5 is outside (below intersection lower bound)
    with pytest.raises(ValueError, match="Input array outside strip"):
        combined(-0.5)

    # s = 2.0 is outside (above intersection upper bound)
    with pytest.raises(ValueError, match="Input array outside strip"):
        combined(2.0)


def test_combined_kernel_skip_bounds_checking():
    """Test CombinedKernel with bounds checking disabled."""
    combined = CombinedKernel([BesselJKernel(0), BesselJKernel(1)], check_bounds=False)

    # Should not raise even outside strip
    result = combined(-0.5)
    assert result.shape == (2,)

    result = combined(2.0)
    assert result.shape == (2,)


def test_combined_kernel_heterogeneous_types():
    """Test CombinedKernel with different kernel types."""
    kernels = [
        BesselJKernel(0),
        SphericalBesselJKernel(1),
        BesselJKernel(0).derive(1),
    ]
    combined = CombinedKernel(kernels)

    s = 1.0
    result = combined.forward(s)

    assert result.shape == (3,)

    # Verify each matches individual kernel
    for i, k in enumerate(kernels):
        assert_allclose(result[i], k.forward(s))


def test_combined_kernel_preserves_check_bounds():
    """Test that CombinedKernel preserves check_bounds in derive()."""
    combined = CombinedKernel([BesselJKernel(0)], check_bounds=False)
    derived = combined.derive(1)

    assert derived.check_bounds is False


@pytest.mark.parametrize(
    "mu_values,s_shape,expected_shape",
    [
        # Single kernel, various s shapes
        ([0], (), (1,)),
        ([0], (4,), (1, 4)),
        # Multiple kernels, scalar s
        ([0, 1], (), (2,)),
        ([0, 1, 2], (), (3,)),
        # Multiple kernels, array s
        ([0, 1], (4,), (2, 4)),
        ([0, 1, 2], (5,), (3, 5)),
    ],
)
def test_combined_kernel_output_shapes(mu_values, s_shape, expected_shape):
    """Test CombinedKernel output shapes with various inputs."""
    kernels = [BesselJKernel(mu) for mu in mu_values]
    combined = CombinedKernel(kernels)

    if s_shape == ():
        s = 1.0
    else:
        s = np.linspace(0.5, 1.5, np.prod(s_shape)).reshape(s_shape)

    result = combined.forward(s)
    assert result.shape == expected_shape
