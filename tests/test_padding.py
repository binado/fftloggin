"""Tests for symmetric padding of logarithmic grids and samples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import j1

from fftloggin import (
    BesselJKernel,
    Padding,
    SphericalBesselJKernel,
    forward,
    get_array_center,
    get_paired_grids,
    infer_dlog,
    lowring_log_kr,
    plan,
    product_plan,
)


@pytest.fixture
def dlog():
    return 4 * np.log(10.0) / 127


@pytest.fixture
def r(dlog):
    return jnp.exp(np.log(1e-4) + dlog * jnp.arange(128))


@pytest.fixture
def samples():
    return jnp.arange(2 * 16, dtype=jnp.float32).reshape(2, 16) + 1.0


@pytest.mark.parametrize(
    ("width", "error"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("2", TypeError)],
)
def test_rejects_invalid_width(width, error):
    with pytest.raises(error):
        Padding(width)


@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize("axis", [0, -1])
def test_pads_zeros_along_axis(samples, width, axis):
    padded = Padding(width)(samples, axis=axis)
    expected = np.pad(
        np.asarray(samples),
        [(width, width) if i == axis % samples.ndim else (0, 0) for i in range(2)],
    )
    assert_array_equal(padded, expected)


@pytest.mark.parametrize("width", [0, 3])
@pytest.mark.parametrize("axis", [0, -1])
def test_crop_inverts_padding(samples, width, axis):
    padding = Padding(width)
    assert_array_equal(padding.crop(padding(samples, axis=axis), axis=axis), samples)


@pytest.mark.parametrize("width", [0, 5])
def test_grid_extends_with_same_spacing_and_centre(r, dlog, width):
    extended = Padding(width).grid(r)
    assert extended.shape == (r.shape[0] + 2 * width,)
    assert_array_equal(extended[width : width + r.shape[0]], r)
    assert_allclose(jnp.diff(jnp.log(extended)), dlog, rtol=1e-4)
    assert_allclose(get_array_center(extended), get_array_center(r), rtol=1e-5)


def test_cropped_output_grid_is_paired_with_original_grid(r, dlog):
    padding = Padding(17)
    log_kr = lowring_log_kr(BesselJKernel(0.0), dlog=dlog, log_kr=1.0)
    _, k = get_paired_grids(r=r, log_kr=log_kr)
    _, k_padded = get_paired_grids(r=padding.grid(r), log_kr=log_kr)
    assert_allclose(padding.crop(k_padded), k, rtol=1e-6)


def test_padding_suppresses_wrap_around_at_output_edges(x64):
    # a(r) = r on r <= 1 has a step at the last sample, and
    # k * integral(a(r) * J0(k*r), r) = J1(k).
    n = 512
    r = jnp.logspace(-4.0, 0.0, n)
    dlog = infer_dlog(r)
    kernel = BesselJKernel(0.0)
    log_kr = lowring_log_kr(kernel, dlog=dlog)
    _, k = get_paired_grids(r=r, log_kr=log_kr)
    padding = Padding(n // 2)
    padded = padding(r)

    unpadded = forward(r, kernel, dlog=dlog, log_kr=log_kr)
    p = plan(kernel, padded.shape[0], dlog=dlog, log_kr=log_kr)
    cropped = padding.crop(forward(padded, p))

    edge = slice(-n // 10, None)
    exact = j1(np.asarray(k[edge]))
    error_unpadded = np.max(np.abs(unpadded[edge] - exact))
    error_padded = np.max(np.abs(cropped[edge] - exact))
    assert cropped.shape == (n,)
    assert error_padded < error_unpadded / 10


def test_padding_is_a_jit_argument(r):
    @jax.jit
    def transform(a, padding):
        padded = padding(a)
        p = plan(BesselJKernel(0.0), padded.shape[0], dlog=0.1)
        return padding.crop(forward(padded, p))

    result = transform(r, Padding(4))
    assert result.shape == r.shape
    assert np.all(np.isfinite(result))


def test_grid_works_under_jit(r):
    padding = Padding(4)
    assert_allclose(jax.jit(padding.grid)(r), padding.grid(r), rtol=1e-6)


def test_vmap_matches_last_axis(samples):
    padding = Padding(3)
    assert_array_equal(jax.vmap(padding)(samples), padding(samples, axis=-1))


def test_gradient_flows_through_padding(r, dlog):
    padding = Padding(8)
    p = plan(BesselJKernel(0.0), r.shape[0] + 16, dlog=dlog)
    grad = jax.grad(lambda a: jnp.sum(padding.crop(forward(padding(a), p))))(r)
    assert grad.shape == r.shape
    assert np.all(np.isfinite(grad))


def test_crop_keeps_product_plan_columns(r, dlog):
    padding = Padding(6)
    padded = padding(r)
    n = padded.shape[0]
    j = SphericalBesselJKernel(2.0)
    band = product_plan(
        j, j, n=n, dlog=dlog, first_bias=-0.25, second_bias=-0.25, max_offset=3
    )
    assert padding.crop(forward(padded, band)).shape == (r.shape[0], 7)
