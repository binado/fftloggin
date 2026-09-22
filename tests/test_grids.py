"""Tests for logarithmic FFTLog coordinate helpers."""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from fftloggin.grids import get_array_center, get_paired_grids, infer_log_kr


def test_get_array_center_typical_values():
    x = jnp.array([1.0, 16.0])
    assert_allclose(get_array_center(x), 4.0)


def test_get_array_center_avoids_float32_underflow():
    x = jnp.array([1e-30, 1e-30], dtype=jnp.float32)
    center = get_array_center(x)
    assert center.dtype == jnp.float32
    assert_allclose(center, np.float32(1e-30), rtol=1e-6)


def test_get_array_center_avoids_float32_overflow():
    x = jnp.array([1e30, 1e30], dtype=jnp.float32)
    center = get_array_center(x)
    assert np.isfinite(center)
    assert_allclose(center, np.float32(1e30), rtol=1e-6)


def test_get_paired_grids_from_r_typical_values():
    x = jnp.array([1.0, 10.0])
    r, k = get_paired_grids(r=x, log_kr=0.0)
    assert_allclose(r, x)
    assert_allclose(k, [0.1, 1.0])


def test_get_paired_grids_from_k_typical_values():
    x = jnp.array([0.1, 1.0])
    r, k = get_paired_grids(k=x, log_kr=0.0)
    assert_allclose(r, [1.0, 10.0])
    assert_allclose(k, x)


def test_get_paired_grids_requires_exactly_one_grid():
    x = jnp.array([1.0, 10.0])
    with np.testing.assert_raises_regex(ValueError, "exactly one of r or k"):
        get_paired_grids()
    with np.testing.assert_raises_regex(ValueError, "exactly one of r or k"):
        get_paired_grids(r=x, k=x)


def test_get_paired_grids_avoids_float32_exp_overflow():
    x = jnp.array([1e20, 1e20], dtype=jnp.float32)
    _, other = get_paired_grids(r=x, log_kr=jnp.float32(100.0))
    expected = np.exp(np.float64(100.0) - np.log(np.float64(1e20)))
    assert other.dtype == jnp.float32
    assert np.all(np.isfinite(other))
    assert_allclose(other, expected, rtol=1e-5)


def test_get_paired_grids_involution():
    x = jnp.array([0.1, 1.0, 10.0])
    log_kr = jnp.array(0.5)
    _, k = get_paired_grids(r=x, log_kr=log_kr)
    r_back, _ = get_paired_grids(k=k, log_kr=log_kr)
    assert_allclose(r_back, x)


def test_get_paired_grids_works_with_jit_and_vmap():
    r = jnp.array([1.0, 10.0])
    log_krs = jnp.array([0.0, 1.0])
    paired = jax.jit(lambda values: get_paired_grids(r=values))(r)
    batched = jax.vmap(lambda log_kr: get_paired_grids(r=r, log_kr=log_kr))(
        log_krs
    )
    assert_allclose(paired[0], r)
    assert_allclose(paired[1], [0.1, 1.0])
    assert_allclose(batched[0], jnp.stack((r, r)))
    assert_allclose(batched[1][0], [0.1, 1.0])
    assert_allclose(batched[1][1], jnp.exp(1.0) * jnp.array([0.1, 1.0]))


def test_infer_log_kr_ycenter_matches_geometric_center():
    x = jnp.array([1.0, 16.0])
    log_kr = infer_log_kr(x, ycenter=2.0)
    assert_allclose(log_kr, np.log(8.0))


def test_infer_log_kr_ymax_avoids_float32_product_overflow():
    x = jnp.array([1e20, 1e21], dtype=jnp.float32)
    log_kr = infer_log_kr(x, ymax=jnp.float32(1e20))
    expected = 2 * np.log(np.float64(1e20))
    assert np.isfinite(log_kr)
    assert_allclose(log_kr, expected, rtol=1e-5)


def test_infer_log_kr_ymin_avoids_float32_product_overflow():
    x = jnp.array([1e19, 1e20], dtype=jnp.float32)
    log_kr = infer_log_kr(x, ymin=jnp.float32(1e20))
    expected = 2 * np.log(np.float64(1e20))
    assert np.isfinite(log_kr)
    assert_allclose(log_kr, expected, rtol=1e-5)


def test_infer_log_kr_ycenter_avoids_float32_product_overflow():
    x = jnp.array([1e20, 1e20], dtype=jnp.float32)
    log_kr = infer_log_kr(x, ycenter=jnp.float32(1e20))
    expected = 2 * np.log(np.float64(1e20))
    assert np.isfinite(log_kr)
    assert_allclose(log_kr, expected, rtol=1e-5)
