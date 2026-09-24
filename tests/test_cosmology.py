"""Tests for the double spherical Bessel transforms in ``fftloggin.cosmology``."""

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import loggamma, spherical_jn

from fftloggin import get_paired_grids
from fftloggin.cosmology import double_spherical_bessel_table, unequal_time_kernel

N = 512
DLOG = 0.02
HALF_WIDTH = 100


def reference_mellin(ell, s, t):
    """Mellin transform of j_ell(x) j_ell(t x) from Assassi et al. eq. (2.20)."""
    if t > 1:
        return t ** (-s) * reference_mellin(ell, s, 1 / t)
    prefactor = (
        2 ** (s - 1)
        * mpmath.pi**2
        * mpmath.gamma(ell + s / 2)
        / (mpmath.gamma((3 - s) / 2) * mpmath.gamma(ell + 1.5))
        * t**ell
    )
    series = mpmath.hyp2f1((s - 1) / 2, ell + s / 2, ell + 1.5, t * t)
    return complex(prefactor * series / (4 * mpmath.pi))


def frequency(m, bias):
    return 1 + bias + 2j * np.pi * m / (N * DLOG)


@pytest.fixture
def gaussian():
    k = np.exp(DLOG * (np.arange(N) - (N - 1) / 2))

    def a(center):
        return jnp.exp(-((jnp.log(k) - center) ** 2) / (2 * 0.3**2))

    return k, a


@pytest.mark.parametrize(
    ("ell", "bias", "oversample", "rtol"),
    [(2, 0.0, 4, 5e-3), (2, -2.0, 1, 5e-4), (20, -2.0, 4, 1e-4)],
)
@pytest.mark.parametrize("m", [0, 5, 40])
def test_table_matches_hypergeometric(x64, ell, bias, oversample, rtol, m):
    table = double_spherical_bessel_table(
        ell, N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH, oversample=oversample
    )
    s = frequency(m, bias)
    offsets = np.arange(-60, 61, 10)
    expected = np.array([reference_mellin(ell, s, np.exp(j * DLOG)) for j in offsets])
    # Errors are measured against the peak; the far tails are tiny.
    assert_allclose(
        table[m, offsets + HALF_WIDTH],
        expected,
        atol=rtol * np.max(np.abs(expected)),
    )


@pytest.mark.parametrize("ell", [2, 10])
def test_table_diagonal_matches_gauss_closed_form(x64, ell):
    bias = -2.0
    table = double_spherical_bessel_table(
        ell, N, dlog=DLOG, bias=bias, half_width=0, oversample=4
    )
    s = frequency(np.arange(N // 2), bias)
    log_value = (
        (s - 1) * np.log(2)
        + 2 * np.log(np.pi)
        + loggamma(ell + s / 2)
        + loggamma(2 - s)
        - 2 * loggamma((3 - s) / 2)
        - loggamma(ell + 2 - s / 2)
        - np.log(4 * np.pi)
    )
    expected = np.exp(log_value)
    assert_allclose(table[:-1, 0], expected, atol=1e-5 * np.max(np.abs(expected)))


@pytest.mark.parametrize(("ell", "bias"), [(1, 0.0), (2, -1.0)])
@pytest.mark.parametrize("offset", [-15, 0, 20])
@pytest.mark.parametrize("row", [-40, 0, 40])
def test_kernel_matches_quadrature(x64, gaussian, ell, bias, offset, row):
    k, a = gaussian
    table = double_spherical_bessel_table(
        ell, N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH, oversample=4
    )
    result = unequal_time_kernel(a(0.0), table, dlog=DLOG, bias=bias)
    chi, _ = get_paired_grids(k=k)
    chi, t = float(chi[N // 2 + row]), np.exp(offset * DLOG)

    def integrand(q):
        weight = np.exp(-(np.log(q) ** 2) / (2 * 0.3**2))
        return weight * spherical_jn(ell, q * chi) * spherical_jn(ell, q * t * chi)

    expected = chi * quad(integrand, 0, np.inf, limit=400)[0]
    assert_allclose(result[N // 2 + row, offset + HALF_WIDTH], expected, atol=1e-6)


def test_kernel_supports_jit_vmap_and_grad(x64, gaussian):
    _, a = gaussian
    ells = jnp.array([2.0, 3.0])
    make = jax.vmap(
        lambda ell: double_spherical_bessel_table(
            ell, N, dlog=DLOG, bias=-2.0, half_width=5
        )
    )
    tables = make(ells)

    @jax.jit
    def total(center, table):
        kernel = unequal_time_kernel(a(center), table, dlog=DLOG, bias=-2.0)
        return jnp.sum(kernel[:, 5])

    for ell, table in zip(ells, tables, strict=True):
        single = double_spherical_bessel_table(
            float(ell), N, dlog=DLOG, bias=-2.0, half_width=5
        )
        assert_allclose(table, single, rtol=1e-12)
        grad = jax.grad(total)(0.1, table)
        step = 1e-4
        finite = (total(0.1 + step, table) - total(0.1 - step, table)) / (2 * step)
        assert_allclose(grad, finite, rtol=1e-6)


@pytest.mark.parametrize(("half_width", "oversample"), [(-1, 1), (2, 0)])
def test_table_rejects_invalid_sizes(half_width, oversample):
    with pytest.raises(ValueError):
        double_spherical_bessel_table(
            2, N, dlog=DLOG, half_width=half_width, oversample=oversample
        )
