"""Tests for the double spherical Bessel transforms in ``fftloggin.cosmology``."""

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import loggamma, spherical_jn

from fftloggin import (
    Derivative,
    SphericalBesselJKernel,
    forward,
    get_paired_grids,
    inverse,
)
from fftloggin.cosmology import double_spherical_bessel_plan, kernel_product_plan

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
    ("ell", "bias", "rtol"),
    [(2, 0.0, 5e-2), (2, -2.0, 5e-4), (20, -2.0, 1e-3)],
)
@pytest.mark.parametrize("m", [0, 5, 40])
def test_table_matches_hypergeometric(x64, ell, bias, rtol, m):
    table = double_spherical_bessel_plan(
        ell, N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH
    ).coeffs
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
    table = double_spherical_bessel_plan(
        ell, N, dlog=DLOG, bias=bias, half_width=0
    ).coeffs
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
    assert_allclose(table[:-1, 0], expected, atol=5e-4 * np.max(np.abs(expected)))


@pytest.mark.parametrize(("ell", "bias"), [(1, 0.0), (2, -1.0)])
@pytest.mark.parametrize("offset", [-15, 0, 20])
@pytest.mark.parametrize("row", [-40, 0, 40])
def test_kernel_matches_quadrature(x64, gaussian, ell, bias, offset, row):
    k, a = gaussian
    plan = double_spherical_bessel_plan(
        ell, N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH
    )
    result = forward(a(0.0), plan)
    chi, _ = get_paired_grids(k=k)
    chi, t = float(chi[N // 2 + row]), np.exp(offset * DLOG)

    def integrand(q):
        weight = np.exp(-(np.log(q) ** 2) / (2 * 0.3**2))
        return weight * spherical_jn(ell, q * chi) * spherical_jn(ell, q * t * chi)

    expected = chi * quad(integrand, 0, np.inf, limit=400)[0]
    assert_allclose(result[N // 2 + row, offset + HALF_WIDTH], expected, atol=1e-6)


def bessel_derivative(ell, order):
    kernel = SphericalBesselJKernel(float(ell))
    return kernel if order == 0 else Derivative(kernel, order)


@pytest.mark.parametrize(
    ("ell", "orders", "bias"),
    [
        (2, (0, 0), 0.5),
        (20, (0, 0), -1.0),
        (10, (0, 2), 0.5),
        (10, (2, 0), -0.5),
        (10, (2, 2), 0.5),
        (10, (1, 2), 0.0),
    ],
)
def test_contraction_matches_single_kernel_transforms(x64, gaussian, ell, orders, bias):
    k, a = gaussian
    chi, _ = get_paired_grids(k=k)
    chi = np.asarray(chi)
    windows = [
        np.exp(-((np.log(chi) - center) ** 2) / (2 * 0.1**2)) for center in (0.0, 0.2)
    ]
    kernels = [bessel_derivative(ell, order) for order in orders]
    f = [
        np.asarray(forward(w, kernel, dlog=DLOG)) / k
        for w, kernel in zip(windows, kernels, strict=True)
    ]
    expected = DLOG * np.sum(k * np.asarray(a(np.log(ell))) * f[0] * f[1])

    plan = kernel_product_plan(
        kernels[0], kernels[1], N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH
    )
    result = np.asarray(forward(a(np.log(ell)), plan))
    rows = np.arange(HALF_WIDTH, N - HALF_WIDTH)
    partners = rows[:, None] + np.arange(-HALF_WIDTH, HALF_WIDTH + 1)
    band = (windows[1] * chi)[partners]
    contracted = DLOG**2 * np.einsum("i,it,it->", windows[0][rows], result[rows], band)
    assert_allclose(contracted, expected, rtol=1e-9)


@pytest.mark.parametrize("orders", [(0, 0), (0, 2)])
def test_kernel_transposes_with_coordinate_ratio(x64, gaussian, orders):
    k, a = gaussian
    ell, bias, width = 3, 0.0, 10
    first, second = (bessel_derivative(ell, order) for order in orders)
    chi = np.asarray(get_paired_grids(k=k)[0])

    def kernel(one, two):
        plan = kernel_product_plan(one, two, N, dlog=DLOG, bias=bias, half_width=width)
        return np.asarray(forward(a(0.0), plan))

    forward_pair, swapped_pair = kernel(first, second), kernel(second, first)
    rows = np.arange(N // 2 - 40, N // 2 + 41)
    for j in (-7, 0, 5):
        # K_21(chi_i, chi_(i+j)) = (chi_i / chi_(i+j)) * K_12(chi_(i+j), chi_i)
        assert_allclose(
            swapped_pair[rows, width + j],
            chi[rows] / chi[rows + j] * forward_pair[rows + j, width - j],
            # Pointwise values carry the table's truncation error; a missing
            # coordinate ratio would be off by up to 15% here.
            atol=1e-3 * np.max(np.abs(forward_pair[rows])),
        )


def spherical_jn_second_derivative(ell, x):
    value = spherical_jn(ell, x)
    slope = spherical_jn(ell, x, derivative=True)
    return -2 / x * slope - (1 - ell * (ell + 1) / x**2) * value


@pytest.mark.parametrize("offset", [-15, 0, 20])
@pytest.mark.parametrize("row", [-40, 0, 40])
def test_mixed_kernel_matches_quadrature(x64, gaussian, offset, row):
    k, a = gaussian
    ell, bias = 3, 0.0
    first = SphericalBesselJKernel(float(ell))
    plan = kernel_product_plan(
        first, Derivative(first, 2), N, dlog=DLOG, bias=bias, half_width=HALF_WIDTH
    )
    result = forward(a(0.0), plan)
    chi, _ = get_paired_grids(k=k)
    chi, t = float(chi[N // 2 + row]), np.exp(offset * DLOG)

    def integrand(q):
        weight = np.exp(-(np.log(q) ** 2) / (2 * 0.3**2))
        return (
            weight
            * spherical_jn(ell, q * chi)
            * spherical_jn_second_derivative(ell, q * t * chi)
        )

    expected = chi * quad(integrand, 0, np.inf, limit=400)[0]
    assert_allclose(result[N // 2 + row, offset + HALF_WIDTH], expected, atol=1e-6)


def test_double_plan_matches_kernel_product(x64):
    kernel = SphericalBesselJKernel(4.0)
    assert_allclose(
        double_spherical_bessel_plan(
            4.0, N, dlog=DLOG, bias=-1.0, log_kr=0.3, half_width=5
        ).coeffs,
        kernel_product_plan(
            kernel, kernel, N, dlog=DLOG, bias=-1.0, log_kr=0.3, half_width=5
        ).coeffs,
    )


def test_plan_supports_jit_vmap_and_grad(x64, gaussian):
    _, a = gaussian
    ells = jnp.array([2.0, 3.0])
    plans = jax.vmap(
        lambda ell: double_spherical_bessel_plan(
            ell, N, dlog=DLOG, bias=-2.0, half_width=5
        )
    )(ells)

    @jax.jit
    def total(center, plan):
        return jnp.sum(forward(a(center), plan)[:, 5])

    batched = jax.vmap(total, in_axes=(None, 0))(0.1, plans)
    for i, ell in enumerate(ells):
        single = double_spherical_bessel_plan(
            float(ell), N, dlog=DLOG, bias=-2.0, half_width=5
        )
        assert_allclose(plans.coeffs[i], single.coeffs, rtol=1e-12)
        assert_allclose(batched[i], total(0.1, single), rtol=1e-12)
        grad = jax.grad(total)(0.1, single)
        step = 1e-4
        finite = (total(0.1 + step, single) - total(0.1 - step, single)) / (2 * step)
        assert_allclose(grad, finite, rtol=1e-6)


@pytest.mark.parametrize("log_kr", [-0.4, 0.7])
def test_log_kr_shifts_output_grid(x64, gaussian, log_kr):
    k, a = gaussian
    ell, bias, offset, row = 2, -1.0, 10, 30
    plan = double_spherical_bessel_plan(
        ell, N, dlog=DLOG, bias=bias, log_kr=log_kr, half_width=HALF_WIDTH
    )
    result = forward(a(0.0), plan)
    chi, _ = get_paired_grids(k=k, log_kr=log_kr)
    chi, t = float(chi[N // 2 + row]), np.exp(offset * DLOG)

    def integrand(q):
        weight = np.exp(-(np.log(q) ** 2) / (2 * 0.3**2))
        return weight * spherical_jn(ell, q * chi) * spherical_jn(ell, q * t * chi)

    expected = chi * quad(integrand, 0, np.inf, limit=400)[0]
    assert_allclose(result[N // 2 + row, offset + HALF_WIDTH], expected, atol=1e-6)


def test_forward_rejects_plan_for_another_size(gaussian):
    _, a = gaussian
    plan = double_spherical_bessel_plan(2, N // 2, dlog=DLOG, half_width=3)
    with pytest.raises(ValueError, match="n="):
        forward(a(0.0), plan)


def test_inverse_rejects_kernel_pair_plan(gaussian):
    _, a = gaussian
    plan = double_spherical_bessel_plan(2, N, dlog=DLOG, half_width=3)
    with pytest.raises(ValueError, match="single-kernel"):
        inverse(a(0.0), plan)


def test_plan_rejects_negative_half_width():
    with pytest.raises(ValueError):
        double_spherical_bessel_plan(2, N, dlog=DLOG, half_width=-1)
