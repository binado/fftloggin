"""
Tests for custom Mellin transform kernels.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import fht, ifht
from fftloggin._backend_numexpr import NUMEXPR_AVAILABLE
from fftloggin.kernels import (
    bessel_derivative_mellin_log_kernel,
    bessel_mellin_kernel,
    bessel_mellin_log_kernel,
)


# test function, analytical Hankel transform is of the same form
def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_bessel_kernel_forward_transform(use_numexpr):
    """Test that explicit Bessel kernel works for forward transform."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Transform with explicit log_kernel
    A = fht(
        a,
        dln,
        mu,
        offset=offset,
        bias=bias,
        log_kernel=bessel_mellin_log_kernel,
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
def test_bessel_kernel_vs_log_kernel(use_numexpr):
    """Test that kernel and log_kernel give equivalent results."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.5
    a = f(r, mu)

    # Transform with log_kernel
    A_log = fht(
        a, dln, mu, log_kernel=bessel_mellin_log_kernel, use_numexpr=use_numexpr
    )

    # Transform with kernel
    A_regular = fht(a, dln, mu, kernel=bessel_mellin_kernel, use_numexpr=use_numexpr)

    assert_allclose(A_log, A_regular, rtol=1e-12)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_kernel_xor_log_kernel(use_numexpr):
    """Verify that exactly one of kernel/log_kernel must be provided."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.5
    a = f(r, mu)

    # Both None should raise ValueError
    with pytest.raises(
        ValueError, match="Either 'kernel' or 'log_kernel' must be provided"
    ):
        fht(a, dln, mu, use_numexpr=use_numexpr)

    # Both provided should raise ValueError
    with pytest.raises(ValueError, match="Cannot specify both"):
        fht(
            a,
            dln,
            mu,
            kernel=bessel_mellin_kernel,
            log_kernel=bessel_mellin_log_kernel,
            use_numexpr=use_numexpr,
        )


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("n", [64, 63])
def test_fht_identity_with_kernel(n, use_numexpr):
    """Test that ifht is the inverse of fht with explicit kernel."""
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
        log_kernel=bessel_mellin_log_kernel,
        use_numexpr=use_numexpr,
    )
    a_ = ifht(
        A,
        dln,
        mu,
        offset=offset,
        bias=bias,
        log_kernel=bessel_mellin_log_kernel,
        use_numexpr=use_numexpr,
    )

    assert_allclose(a_, a, rtol=1.5e-7)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_vectorized_fht_with_kernel(use_numexpr):
    """Test vectorized transforms with explicit kernel."""
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
        log_kernel=bessel_mellin_log_kernel,
        use_numexpr=use_numexpr,
    )
    assert out.shape == (3, 1, n)


def test_custom_power_law_kernel():
    """Test with a simple custom power-law kernel."""

    def power_law_log_kernel(mu, y, q):
        """Simple power law kernel: k^(-mu) which has Mellin transform related to Gamma functions."""
        # This is a simplified test kernel
        # M[r^alpha](s) = Gamma(alpha + s) for appropriate alpha
        # For testing purposes, use a simple form
        return -mu * np.log(1 + y**2)  # Simplified for testing

    r = np.logspace(-2, 2, 32)
    dln = math.log(r[1] / r[0])
    mu = 1.0
    a = np.exp(-(r**2))

    # This should not raise an error
    A = fht(a, dln, mu, log_kernel=power_law_log_kernel)

    # Check that we got a result of the right shape
    assert A.shape == a.shape
    assert np.all(np.isfinite(A))


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_bessel_derivative_kernel_basic(use_numexpr):
    """Basic test that Bessel derivative kernel works."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-2, 2, 32)
    dln = math.log(r[1] / r[0])
    mu = 1.0
    a = np.exp(-(r**2))

    # Should not raise an error
    A = fht(
        a,
        dln,
        mu,
        log_kernel=bessel_derivative_mellin_log_kernel,
        use_numexpr=use_numexpr,
    )

    # Check shape and finiteness
    assert A.shape == a.shape
    assert np.all(np.isfinite(A))


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_axis_parameter_with_kernel(use_numexpr):
    """Test that the axis parameter works with custom kernels."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.3
    a = np.asarray(f(r, mu))

    # Test different axis values
    for shape, axis in [((1, 16, 1), 1), ((16, 1, 1), 0), ((1, 1, 16), 2)]:
        a_reshaped = np.reshape(a, shape)
        A = fht(
            a_reshaped,
            dln,
            mu,
            axis=axis,
            log_kernel=bessel_mellin_log_kernel,
            use_numexpr=use_numexpr,
        )
        assert A.shape == shape
