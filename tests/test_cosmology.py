"""Tests for the double spherical Bessel transforms in ``fftloggin.cosmology``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import loggamma, spherical_jn

from fftloggin import (
    SphericalBesselJKernel,
    forward,
    get_paired_grids,
    inverse,
    plan,
    product_plan,
)
from fftloggin.cosmology import double_spherical_bessel_plan

N = 512
DLOG = 0.02
HALF_WIDTH = 100


def frequency(m, bias):
    return 1 + bias + 2j * np.pi * m / (N * DLOG)


@pytest.fixture
def gaussian():
    k = np.exp(DLOG * (np.arange(N) - (N - 1) / 2))

    def a(center):
        return jnp.exp(-((jnp.log(k) - center) ** 2) / (2 * 0.3**2))

    return k, a


@pytest.mark.parametrize("ell", [2, 10])
def test_table_diagonal_matches_gauss_closed_form(x64, ell):
    bias = -2.0
    table = double_spherical_bessel_plan(
        ell, N, dlog=DLOG, bias=bias, max_offset=0
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
        ell, N, dlog=DLOG, bias=bias, max_offset=HALF_WIDTH
    )
    result = forward(a(0.0), plan)
    chi, _ = get_paired_grids(k=k)
    chi, t = float(chi[N // 2 + row]), np.exp(offset * DLOG)

    def integrand(q):
        weight = np.exp(-(np.log(q) ** 2) / (2 * 0.3**2))
        return weight * spherical_jn(ell, q * chi) * spherical_jn(ell, q * t * chi)

    expected = chi * quad(integrand, 0, np.inf, limit=400)[0]
    assert_allclose(result[N // 2 + row, offset + HALF_WIDTH], expected, atol=1e-6)


def test_double_plan_is_product_of_centred_plans(x64):
    ell, bias, log_kr = 4.0, -1.0, 0.3
    kernel = SphericalBesselJKernel(ell)
    # The strip of j_ell is (-ell, 2); both factors sit mid-way in theirs.
    q = (max(-ell, bias - 1) + min(2.0, 1 + bias + ell)) / 2
    one = plan(kernel, N, dlog=DLOG, bias=q - 1, log_kr=log_kr)
    two = plan(kernel, N, dlog=DLOG, bias=bias - q, log_kr=log_kr)
    assert_allclose(
        double_spherical_bessel_plan(
            ell, N, dlog=DLOG, bias=bias, log_kr=log_kr, max_offset=5
        ).coeffs,
        product_plan(one, two, max_offset=5).coeffs,
        rtol=1e-12,
    )


def test_plan_supports_jit_vmap_and_grad(x64, gaussian):
    _, a = gaussian
    ells = jnp.array([2.0, 3.0])
    plans = jax.vmap(
        lambda ell: double_spherical_bessel_plan(
            ell, N, dlog=DLOG, bias=-2.0, max_offset=5
        )
    )(ells)

    @jax.jit
    def total(center, plan):
        return jnp.sum(forward(a(center), plan)[:, 5])

    batched = jax.vmap(total, in_axes=(None, 0))(0.1, plans)
    for i, ell in enumerate(ells):
        single = double_spherical_bessel_plan(
            float(ell), N, dlog=DLOG, bias=-2.0, max_offset=5
        )
        # Batched FFTs round differently; compare against the peak.
        scale = np.max(np.abs(single.coeffs))
        assert_allclose(plans.coeffs[i], single.coeffs, rtol=0, atol=1e-12 * scale)
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
        ell, N, dlog=DLOG, bias=bias, log_kr=log_kr, max_offset=HALF_WIDTH
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
    plan = double_spherical_bessel_plan(2, N // 2, dlog=DLOG, max_offset=3)
    with pytest.raises(ValueError, match="n="):
        forward(a(0.0), plan)


def test_inverse_rejects_kernel_pair_plan(gaussian):
    _, a = gaussian
    plan = double_spherical_bessel_plan(2, N, dlog=DLOG, max_offset=3)
    with pytest.raises(ValueError, match="single-kernel"):
        inverse(a(0.0), plan)


def test_plan_rejects_negative_max_offset():
    with pytest.raises(ValueError):
        double_spherical_bessel_plan(2, N, dlog=DLOG, max_offset=-1)
