"""Tests for weights that taper samples at the edges of their support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import simpson
from scipy.special import j0, j1

from fftloggin import (
    BesselJKernel,
    Padding,
    forward,
    get_paired_grids,
    infer_dlog,
    lowring_log_kr,
    plan,
    taper,
)

SHAPES = ("cosine", "planck")


@pytest.fixture
def dlog():
    return 4 * np.log(10.0) / 127


@pytest.fixture
def r(dlog):
    return jnp.exp(np.log(1e-4) + dlog * jnp.arange(128))


@pytest.fixture
def width(dlog):
    # Eight samples per ramp, so samples fall at its start, middle and end.
    return 8 * dlog


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": "top"},
        {"shape": "gaussian"},
        {"lo": 1e-3, "side": "hi"},
        {"hi": 1.0, "side": "lo"},
    ],
)
def test_rejects_invalid_arguments(r, kwargs):
    with pytest.raises(ValueError):
        taper(r, 0.1, **kwargs)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_ramp_rises_from_zero_at_edge_to_one_at_width(r, width, shape, side):
    weights = taper(r, width, side=side, shape=shape)
    ramp = weights if side == "lo" else weights[::-1]
    assert ramp[0] == 0.0
    assert_allclose(ramp[4], 0.5, atol=1e-5)
    assert_allclose(ramp[8:], 1.0, atol=1e-5)
    assert np.all(np.diff(ramp[:9]) >= 0)


@pytest.mark.parametrize("shape", SHAPES)
def test_explicit_edges_zero_samples_beyond_support(r, width, shape):
    weights = taper(r, width, lo=r[20], hi=r[100], shape=shape)
    assert_array_equal(weights[:21], 0.0)
    assert_array_equal(weights[100:], 0.0)
    assert_allclose(weights[28:93], 1.0, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_both_sides_multiply_the_two_ramps(r, shape):
    # Wide ramps overlap in the middle of the grid.
    width = 6.0
    lower = taper(r, width, side="lo", shape=shape)
    upper = taper(r, width, side="hi", shape=shape)
    assert_allclose(taper(r, width, shape=shape), lower * upper, rtol=1e-6)


def test_tapering_commutes_with_padding(r, width):
    padding = Padding(10)
    tapered_then_padded = padding(r * taper(r, width))
    padded_then_tapered = padding(r) * taper(padding.grid(r), width, lo=r[0], hi=r[-1])
    assert_allclose(padded_then_tapered, tapered_then_padded, rtol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_works_under_jit_with_traced_width_and_edges(r, width, shape):
    @jax.jit
    def weights(x, width, lo, hi):
        return taper(x, width, lo=lo, hi=hi, shape=shape)

    assert_allclose(
        weights(r, width, r[20], r[100]),
        taper(r, width, lo=r[20], hi=r[100], shape=shape),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_vmap_over_width_matches_loop(r, dlog, shape):
    widths = jnp.array([4.0, 8.0, 16.0]) * dlog
    mapped = jax.vmap(lambda w: taper(r, w, shape=shape))(widths)
    looped = jnp.stack([taper(r, w, shape=shape) for w in widths])
    assert_allclose(mapped, looped, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_gradient_with_respect_to_width_matches_finite_difference(x64, r, width, shape):
    # Samples sit exactly at the ends of each ramp, where the Planck ramp's
    # 1/t and 1/(1 - t) diverge.
    def total(w):
        return jnp.sum(r * taper(r, w, shape=shape))

    grad = jax.grad(total)(width)
    step = 1e-6
    finite_difference = (total(width + step) - total(width - step)) / (2 * step)
    assert np.isfinite(grad)
    assert_allclose(grad, finite_difference, rtol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
def test_taper_suppresses_ringing_from_a_step(x64, shape):
    # a(r) = r on r <= 1 steps to zero at the last sample. Tapered, its
    # transform k * integral(a(r) * w(r) * J0(k*r), r) is computed by
    # quadrature; untapered, it is J1(k).
    n = 512
    r = jnp.logspace(-4.0, 0.0, n)
    dlog = infer_dlog(r)
    kernel = BesselJKernel(0.0)
    log_kr = lowring_log_kr(kernel, dlog=dlog)
    _, k = get_paired_grids(r=r, log_kr=log_kr)
    padding = Padding(n // 2)
    p = plan(kernel, n + 2 * padding.width, dlog=dlog, log_kr=log_kr)

    def transform(a):
        return np.asarray(padding.crop(forward(padding(a), p)))

    width = 0.3
    tapered = transform(r * taper(r, width, side="hi", shape=shape))
    untapered = transform(r)

    middle = slice(n // 4, 3 * n // 4, 8)
    k_middle = np.asarray(k[middle])
    r_dense = np.linspace(0.0, 1.0, 2**18 + 1)[1:]
    integrand = r_dense * np.asarray(taper(r_dense, width, side="hi", shape=shape))
    exact = np.array(
        [q * simpson(integrand * j0(q * r_dense), x=r_dense) for q in k_middle]
    )

    error_tapered = np.max(np.abs(tapered[middle] - exact))
    error_untapered = np.max(np.abs(untapered[middle] - j1(k_middle)))
    assert error_tapered < error_untapered / 100
