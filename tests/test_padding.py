"""Tests for symmetric padding of logarithmic grids and samples."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.fft import next_fast_len
from scipy.special import j1

from fftloggin import (
    BesselJKernel,
    Padding,
    SphericalBesselJKernel,
    fast_size,
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


def _smooth_upto(limit, radices):
    values = {1}
    for radix in radices:
        grown = set()
        for value in values:
            product = value
            while product <= limit:
                grown.add(product)
                product *= radix
        values = grown
    return values


def _next_smooth(n, radices, parity=None):
    odd_radices = tuple(radix for radix in radices if radix != 2)
    search = odd_radices if parity == "odd" else radices
    limit = max(n, 2)
    while True:
        candidates = [
            value
            for value in _smooth_upto(limit, search)
            if value >= n
            and (parity is None or value % 2 == (0 if parity == "even" else 1))
        ]
        if candidates:
            return min(candidates)
        limit *= 2


@pytest.mark.parametrize(
    "n", [1, 2, 3, 7, 8, 11, 37, 2048, 2050, 2498, 2664, 9992, 10007]
)
@pytest.mark.parametrize("radices", [(2, 3, 5, 7), (2, 3, 5), (2, 3, 5, 7, 11)])
@pytest.mark.parametrize("parity", [None, "even", "odd"])
def test_fast_size_is_minimal_smooth_length(n, radices, parity):
    result = fast_size(n, radices=radices, parity=parity)
    assert result >= n
    if parity == "even":
        assert result % 2 == 0
    elif parity == "odd":
        assert result % 2 == 1
    assert result == _next_smooth(n, radices, parity)


@pytest.mark.parametrize("radices", [(2, 3, 5, 7), (2, 3, 5), (3, 5, 7)])
@pytest.mark.parametrize("parity", [None, "even", "odd"])
def test_fast_size_matches_enumeration_up_to_few_thousand(radices, parity):
    if parity == "even" and 2 not in radices:
        with pytest.raises(ValueError, match="even parity"):
            fast_size(8, radices=radices, parity=parity)
        return
    max_n = 2000
    search = (
        tuple(radix for radix in radices if radix != 2)
        if parity == "odd"
        else radices
    )
    smooth = sorted(
        value
        for value in _smooth_upto(max_n * 4, search)
        if parity is None or value % 2 == (0 if parity == "even" else 1)
    )
    assert smooth[-1] >= max_n
    for n in range(1, max_n + 1):
        expected = next(value for value in smooth if value >= n)
        assert fast_size(n, radices=radices, parity=parity) == expected


@pytest.mark.parametrize("n", [1, 2, 37, 2048, 2050, 2498, 2664, 9992, 10007, 93059])
def test_fast_size_matches_scipy_for_pocketfft_radices(n):
    assert fast_size(n, radices=(2, 3, 5)) == next_fast_len(n, real=True)
    assert fast_size(n, radices=(2, 3, 5, 7, 11)) == next_fast_len(n, real=False)


def test_fast_size_ignores_duplicate_radices():
    assert fast_size(10, radices=(2, 2, 3, 3)) == fast_size(10, radices=(2, 3))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n": True}, TypeError),
        ({"n": 1.5}, TypeError),
        ({"n": "8"}, TypeError),
        ({"n": 0}, ValueError),
        ({"n": -1}, ValueError),
        ({"n": 8, "radices": [2, 3, 5, 7]}, TypeError),
        ({"n": 8, "radices": ()}, ValueError),
        ({"n": 8, "radices": (1, 2)}, ValueError),
        ({"n": 8, "radices": (4,)}, ValueError),
        ({"n": 8, "radices": (2, True)}, TypeError),
        ({"n": 8, "parity": "even", "radices": (3, 5, 7)}, ValueError),
        ({"n": 8, "parity": "odd", "radices": (2,)}, ValueError),
        ({"n": 8, "parity": "both"}, ValueError),
    ],
)
def test_fast_size_rejects_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        fast_size(**kwargs)


def test_fast_padding_grows_fifteen_percent_pad_of_2048():
    padding = Padding.fast(2048, math.ceil(0.15 * 2048))
    assert padding.width == 320
    assert 2048 + 2 * padding.width == 2688


def test_fast_padding_is_zero_when_length_is_already_smooth():
    assert Padding.fast(2048).width == 0
    assert Padding.fast(2187).width == 0


@pytest.mark.parametrize("n", [11, 128, 2048, 2049])
@pytest.mark.parametrize("min_width", [0, 10, 308])
def test_fast_padding_length_matches_fast_size_parity(n, min_width):
    padding = Padding.fast(n, min_width)
    parity = "even" if n % 2 == 0 else "odd"
    assert padding.width >= min_width
    assert n + 2 * padding.width == fast_size(
        n + 2 * min_width, parity=parity
    )


def test_fast_padding_preserves_grid_centre(r, dlog):
    n = int(r.shape[0])
    padding = Padding.fast(n, min_width=10)
    extended = padding.grid(r)
    assert extended.shape == (n + 2 * padding.width,)
    assert_array_equal(extended[padding.width : padding.width + n], r)
    assert_allclose(jnp.diff(jnp.log(extended)), dlog, rtol=1e-4)
    assert_allclose(get_array_center(extended), get_array_center(r), rtol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n": True}, TypeError),
        ({"n": 8, "min_width": True}, TypeError),
        ({"n": 8, "min_width": 1.5}, TypeError),
        ({"n": 0}, ValueError),
        ({"n": 8, "min_width": -1}, ValueError),
        ({"n": 8, "radices": (3, 5, 7)}, ValueError),
        ({"n": 9, "radices": (2,)}, ValueError),
    ],
)
def test_fast_padding_rejects_invalid_arguments(kwargs, error):
    with pytest.raises(error):
        Padding.fast(**kwargs)
