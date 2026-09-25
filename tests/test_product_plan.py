"""Tests for combining single-kernel plans with ``product_plan``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import spherical_jn

from fftloggin import (
    Derivative,
    SphericalBesselJKernel,
    forward,
    get_paired_grids,
    inverse,
    plan,
    product_plan,
)

N = 512
DLOG = 0.02
HALF_WIDTH = 100


def bessel(ell, order=0):
    kernel = SphericalBesselJKernel(float(ell))
    return kernel if order == 0 else kernel.transform(Derivative(order))


def split_plans(first, second, n, bias, log_kr=0.0, dlog=DLOG):
    """Plans whose product has ``bias``, each centred in its kernel's strip."""
    (lower1, upper1), (lower2, upper2) = first.domain, second.domain
    real_s = 1 + bias
    q = (max(lower1, real_s - upper2) + min(upper1, real_s - lower2)) / 2
    return (
        plan(first, n, dlog=dlog, bias=q - 1, log_kr=log_kr),
        plan(second, n, dlog=dlog, bias=real_s - 1 - q, log_kr=log_kr),
    )


@pytest.fixture
def gaussian():
    k = np.exp(DLOG * (np.arange(N) - (N - 1) / 2))

    def a(center):
        return jnp.exp(-((jnp.log(k) - center) ** 2) / (2 * 0.3**2))

    return k, a


@pytest.mark.parametrize(
    ("ells", "orders", "bias"),
    [((2, 2), (0, 0), 0.5), ((0, 3), (0, 0), 0.5), ((10, 10), (0, 2), 0.5)],
)
@pytest.mark.parametrize("log_kr", [0.0, 0.7])
def test_band_matches_two_dimensional_transform(x64, ells, orders, bias, log_kr):
    n, dlog = 64, 0.1
    k = np.exp(dlog * (np.arange(n) - (n - 1) / 2))
    chi = np.asarray(get_paired_grids(k=k, log_kr=log_kr)[0])
    a = np.exp(-(np.log(k) ** 2) / 2)
    first, second = split_plans(
        bessel(ells[0], orders[0]),
        bessel(ells[1], orders[1]),
        n,
        bias,
        log_kr,
        dlog,
    )
    columns = jax.vmap(forward, in_axes=(0, None), out_axes=1)
    identity = jnp.eye(n)
    full = (
        columns(identity, first) @ np.diag(a / (k * dlog)) @ columns(identity, second).T
    )
    full = full / chi[None, :]

    band = np.asarray(forward(a, product_plan(first, second, half_width=n - 1)))
    rows, offsets = np.meshgrid(np.arange(n), np.arange(-(n - 1), n), indexing="ij")
    inside = (rows + offsets >= 0) & (rows + offsets < n)
    expected = full[rows[inside], (rows + offsets)[inside]]
    assert_allclose(
        band[inside], expected, rtol=0, atol=1e-12 * np.max(np.abs(expected))
    )


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
    chi = np.asarray(get_paired_grids(k=k)[0])
    windows = [
        np.exp(-((np.log(chi) - center) ** 2) / (2 * 0.1**2)) for center in (0.0, 0.2)
    ]
    kernels = [bessel(ell, order) for order in orders]
    f = [
        np.asarray(forward(w, kernel, dlog=DLOG)) / k
        for w, kernel in zip(windows, kernels, strict=True)
    ]
    expected = DLOG * np.sum(k * np.asarray(a(np.log(ell))) * f[0] * f[1])

    pp = product_plan(
        *split_plans(kernels[0], kernels[1], N, bias), half_width=HALF_WIDTH
    )
    result = np.asarray(forward(a(np.log(ell)), pp))
    rows = np.arange(HALF_WIDTH, N - HALF_WIDTH)
    partners = rows[:, None] + np.arange(-HALF_WIDTH, HALF_WIDTH + 1)
    band = (windows[1] * chi)[partners]
    contracted = DLOG**2 * np.einsum("i,it,it->", windows[0][rows], result[rows], band)
    assert_allclose(contracted, expected, rtol=1e-9)


@pytest.mark.parametrize("orders", [(0, 0), (0, 2)])
def test_kernel_transposes_with_coordinate_ratio(x64, gaussian, orders):
    k, a = gaussian
    ell, bias, width = 3, 0.0, 10
    first, second = (bessel(ell, order) for order in orders)
    chi = np.asarray(get_paired_grids(k=k)[0])

    def kernel(one, two):
        pp = product_plan(*split_plans(one, two, N, bias), half_width=width)
        return np.asarray(forward(a(0.0), pp))

    forward_pair, swapped_pair = kernel(first, second), kernel(second, first)
    rows = np.arange(N // 2 - 40, N // 2 + 41)
    for j in (-7, 0, 5):
        # K_21(chi_i, chi_(i+j)) = (chi_i / chi_(i+j)) * K_12(chi_(i+j), chi_i)
        assert_allclose(
            swapped_pair[rows, width + j],
            chi[rows] / chi[rows + j] * forward_pair[rows + j, width - j],
            # A missing coordinate ratio would be off by up to 15% here.
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
    first = bessel(ell)
    pp = product_plan(
        *split_plans(first, first.transform(Derivative(2)), N, bias),
        half_width=HALF_WIDTH,
    )
    result = forward(a(0.0), pp)
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


def test_product_plan_supports_jit_vmap_and_grad(x64, gaussian):
    _, a = gaussian
    width = 5

    def make(ell):
        kernel = SphericalBesselJKernel(ell)
        one = plan(kernel, N, dlog=DLOG, bias=-0.5)
        return product_plan(one, one, half_width=width)

    ells = jnp.array([2.0, 3.0])
    plans = jax.jit(jax.vmap(make))(ells)

    @jax.jit
    def total(center, pp):
        return forward(a(center), pp)[N // 2, width + 2]

    batched = jax.vmap(total, in_axes=(None, 0))(0.1, plans)
    for i, ell in enumerate(ells):
        single = make(float(ell))
        scale = np.max(np.abs(single.coeffs))
        assert_allclose(plans.coeffs[i], single.coeffs, rtol=0, atol=1e-12 * scale)
        assert_allclose(batched[i], total(0.1, single), rtol=1e-12)
        grad = jax.grad(total)(0.1, single)
        step = 1e-4
        finite = (total(0.1 + step, single) - total(0.1 - step, single)) / (2 * step)
        assert_allclose(grad, finite, rtol=1e-6)


def test_product_plan_has_combined_bias():
    one = plan(bessel(2), N, dlog=DLOG, bias=-0.25)
    two = plan(bessel(2), N, dlog=DLOG, bias=0.5)
    pp = product_plan(one, two, half_width=3)
    assert pp.coeffs.shape == (N // 2 + 1, 7)
    assert_allclose(pp.bias, 1.25)


def test_product_plan_rejects_negative_half_width():
    one = plan(bessel(2), N, dlog=DLOG)
    with pytest.raises(ValueError, match="half_width"):
        product_plan(one, one, half_width=-1)


def test_product_plan_rejects_other_sample_count():
    with pytest.raises(ValueError, match="sample counts"):
        product_plan(
            plan(bessel(2), N, dlog=DLOG),
            plan(bessel(2), N // 2, dlog=DLOG),
            half_width=3,
        )


def test_product_plan_rejects_band_plans():
    one = plan(bessel(2), N, dlog=DLOG)
    band = product_plan(one, one, half_width=3)
    with pytest.raises(ValueError, match="single-kernel"):
        product_plan(band, one, half_width=3)


def test_inverse_rejects_product_plan(gaussian):
    _, a = gaussian
    one = plan(bessel(2), N, dlog=DLOG)
    with pytest.raises(ValueError, match="single-kernel"):
        inverse(a(0.0), product_plan(one, one, half_width=3))
