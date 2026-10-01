"""Tests for weights that taper samples at the edges of their support."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fftloggin import Padding, taper

SHAPES = ("cosine", "planck")


@pytest.fixture
def dlog():
    return 4 * np.log(10.0) / 127


@pytest.fixture
def r(dlog):
    return jnp.exp(np.log(1e-4) + dlog * jnp.arange(128))


@pytest.fixture
def width(dlog):
    # A whole number of samples per ramp, so samples fall exactly at its ends.
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

    # XLA may fuse the ramp under jit and round it differently by an ulp.
    assert_allclose(
        weights(r, width, r[20], r[100]),
        taper(r, width, lo=r[20], hi=r[100], shape=shape),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_vmap_over_width_matches_loop(r, dlog, shape):
    widths = jnp.array([4.0, 8.0, 16.0]) * dlog
    mapped = jax.vmap(lambda w: taper(r, w, shape=shape))(widths)
    looped = jnp.stack([taper(r, w, shape=shape) for w in widths])
    assert_array_equal(mapped, looped)


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
