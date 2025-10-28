"""
Test suite for Grid class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fftloggin.grids import (
    Grid,
    get_array_center,
    get_other_array,
    infer_dlog,
    infer_logc,
)


@pytest.mark.parametrize(
    "rbounds",
    [
        (1e-4, 1e0),
        (1e-2, 1e0),
        (1e-2, 1e2),
    ],
)
@pytest.mark.parametrize("n", [3, 63, 64])
@pytest.mark.parametrize("logc", [-1, 0, 1])
def test_grid_constructor(rbounds: tuple[float, float], n: int, logc: float):
    """Test basic Grid construction."""
    rmin, rmax = rbounds
    r = np.geomspace(rmin, rmax, n)
    k = np.exp(logc) / r[::-1]
    grid = Grid(r, k)

    assert grid.n == n
    assert_array_equal(grid.r, r)
    assert_array_equal(grid.k, k)
    assert_allclose(grid.logc, logc)

    rc = np.sqrt(rmin * rmax)
    assert_allclose(grid.rcenter, rc)

    kc = np.exp(logc) / rc
    assert_allclose(grid.kcenter, kc)


@pytest.mark.parametrize("n", [3, 63, 64])
@pytest.mark.parametrize("batch_shape", [(1,), (1, 1)])
def test_grid_constructor_with_ndarrays(n: int, batch_shape: tuple[int, ...]):
    r = np.logspace(-2, 2, n).reshape(*batch_shape, -1)
    logc = np.zeros(batch_shape)
    k = np.exp(logc) / r[..., ::-1]
    grid = Grid(r, k)
    assert grid.n == n
    assert_allclose(logc, grid.logc)
    assert_array_equal(grid.r, r)
    assert_array_equal(grid.k, k)

    rc = np.sqrt(r[..., 0] * r[..., -1])
    assert_allclose(grid.rcenter, rc)

    kc = np.exp(logc) / rc
    assert_allclose(grid.kcenter, kc)


def test_grid_single_element_raises():
    """Test that Grid raises error for single-element arrays.

    Single-element arrays don't have well-defined log-spacing,
    so Grid validation should catch this.
    """
    r = np.array([1.0])
    k = np.array([1.0])
    # Single element arrays have NaN dlog, which fails validation
    with pytest.raises(ValueError, match="must have at least 2 elements"):
        Grid(r, k)


def test_grid_two_elements():
    """Test Grid with two-element arrays."""
    # Create r with known log-spacing
    r = np.array([1.0, 10.0])
    logc = 0.0
    # k must have same log-spacing: dlog = log(10/1) = log(10)
    # k = exp(logc) / r[::-1] = 1 / [10, 1] = [0.1, 1.0]
    # Check: log(1.0/0.1) = log(10) ✓
    k = np.exp(logc) / r[::-1]

    grid = Grid(r, k)
    assert grid.n == 2
    assert_allclose(grid.logc, logc)

    # Verify centers
    assert_allclose(grid.rcenter, np.sqrt(1.0 * 10.0))  # sqrt(10)
    assert_allclose(grid.kcenter, np.sqrt(0.1 * 1.0))  # sqrt(0.1)


def test_grid_array_setter():
    r = np.logspace(-2, 2, 128)
    k = 1 / r[::-1]
    grid = Grid(r, k)
    assert_allclose(grid.logc, 0)

    new_r = np.logspace(-1, 1, 64)
    new_k = 1 / new_r[::-1]
    grid.r = new_r
    assert grid.n == new_r.shape[-1]
    assert_array_equal(grid.r, new_r)
    assert_allclose(grid.k, new_k)
    assert_allclose(grid.logc, 0)
    assert_allclose(grid.rcenter, 1)
    assert_allclose(grid.kcenter, 1)

    new_k = np.logspace(-4, 4, 256)
    new_r = 1 / new_k[::-1]
    grid.k = new_k
    assert grid.n == new_k.shape[-1]
    assert_array_equal(grid.k, new_k)
    assert_allclose(grid.r, new_r)
    assert_allclose(grid.logc, 0)
    assert_allclose(grid.rcenter, 1)
    assert_allclose(grid.kcenter, 1)


def test_grid_array_setter_non_logspaced_r():
    """Test that Grid setter validates log-spacing for r."""
    r = np.logspace(-2, 2, 128)
    k = 1 / r[::-1]
    grid = Grid(r, k)

    # Try to set r to non-log-spaced array
    non_logspaced_r = np.linspace(0.01, 100, 128)
    with pytest.raises(ValueError):
        grid.r = non_logspaced_r


def test_grid_array_setter_non_logspaced_k():
    """Test that Grid setter validates log-spacing for k."""
    r = np.logspace(-2, 2, 128)
    k = 1 / r[::-1]
    grid = Grid(r, k)

    # Try to set k to non-log-spaced array
    non_logspaced_k = np.linspace(0.01, 100, 128)
    with pytest.raises(ValueError):
        grid.k = non_logspaced_k


def test_get_array_center():
    """Test get_array_center function."""
    r = np.logspace(-2, 2, 128)
    center = get_array_center(r)
    expected_center = np.sqrt(r[0] * r[-1])
    assert_allclose(center, expected_center)


def test_get_array_center_batched():
    """Test get_array_center with batched arrays."""
    r = np.logspace(-2, 2, 128).reshape(1, -1)
    center = get_array_center(r)
    expected_center = np.sqrt(r[..., 0] * r[..., -1])
    assert_allclose(center, expected_center)

    # Test with 2D batch
    r = np.logspace(-2, 2, 128).reshape(1, 1, -1)
    center = get_array_center(r)
    expected_center = np.sqrt(r[..., 0] * r[..., -1])
    assert_allclose(center, expected_center)


def test_get_array_center_geometric_mean():
    """Test that get_array_center returns geometric mean of endpoints."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    center = get_array_center(x)
    expected = np.sqrt(1.0 * 16.0)  # = 4.0
    assert_allclose(center, expected)


def test_infer_logc_with_ymax():
    """Test infer_logc with ymax parameter."""
    r = np.logspace(-2, 2, 128)
    ymax = 100.0
    logc = infer_logc(r, ymax=ymax)

    expected_logc = np.log(ymax * r.min())
    assert np.isclose(logc, expected_logc)


def test_infer_logc_with_ymin():
    """Test infer_logc with ymin parameter."""
    r = np.logspace(-2, 2, 128)
    ymin = 0.01
    logc = infer_logc(r, ymin=ymin)

    expected_logc = np.log(ymin * r.max())
    assert np.isclose(logc, expected_logc)


def test_infer_logc_priority():
    """Test that infer_logc respects priority order: logc > ycenter > ymax > ymin."""
    r = np.logspace(-2, 2, 128)

    # When multiple are provided, logc should win
    logc = infer_logc(r, logc=0.1, ycenter=1.0, ymax=100.0, ymin=0.01)
    assert logc == 0.1

    # When logc is not provided, ycenter should win
    logc = infer_logc(r, ycenter=1.0, ymax=100.0, ymin=0.01)
    r_center = np.sqrt(r.min() * r.max())
    assert np.isclose(logc, np.log(1.0 * r_center))


def test_infer_logc_no_args():
    """Test that infer_logc raises error when no arguments provided."""
    r = np.logspace(-2, 2, 128)

    with pytest.raises(ValueError, match="must be provided"):
        infer_logc(r)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_infer_logc_batched_logc(batch_shape: tuple[int, ...]):
    """Test infer_logc with batched logc parameter."""
    r = np.logspace(-2, 2, 128)
    logc_values = np.random.randn(*batch_shape)
    logc = infer_logc(r, logc=logc_values)
    assert_array_equal(logc, logc_values)
    assert logc.shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_infer_logc_batched_ycenter(batch_shape: tuple[int, ...]):
    """Test infer_logc with batched ycenter parameter."""
    r = np.logspace(-2, 2, 128)
    ycenter_values = np.random.rand(*batch_shape) + 0.5
    logc = infer_logc(r, ycenter=ycenter_values)

    r_center = np.sqrt(r.min() * r.max())
    expected_logc = np.log(ycenter_values * r_center)
    assert_allclose(logc, expected_logc)
    assert logc.shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_infer_logc_batched_ymax(batch_shape: tuple[int, ...]):
    """Test infer_logc with batched ymax parameter."""
    r = np.logspace(-2, 2, 128)
    ymax_values = np.random.rand(*batch_shape) * 100 + 1
    logc = infer_logc(r, ymax=ymax_values)

    expected_logc = np.log(ymax_values * r.min())
    assert_allclose(logc, expected_logc)
    assert logc.shape == batch_shape


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_infer_logc_batched_ymin(batch_shape: tuple[int, ...]):
    """Test infer_logc with batched ymin parameter."""
    r = np.logspace(-2, 2, 128)
    ymin_values = np.random.rand(*batch_shape) * 0.1 + 0.001
    logc = infer_logc(r, ymin=ymin_values)

    expected_logc = np.log(ymin_values * r.max())
    assert_allclose(logc, expected_logc)
    assert logc.shape == batch_shape


# ============================================================================
# Tests for batched infer_dlog
# ============================================================================


def test_infer_dlog_batched():
    """Test infer_dlog with batched arrays."""
    # Test with 1D batch: shape (2, 64)
    n = 64
    r = np.tile(np.logspace(-2, 2, n), (2, 1))
    dlog = infer_dlog(r)

    expected_dlog = np.log(r[..., 1] / r[..., 0])
    assert_allclose(dlog, expected_dlog)
    assert dlog.shape == (2,)

    # Test with 2D batch: shape (2, 3, 32)
    n = 32
    r = np.tile(np.logspace(-2, 2, n), (2, 3, 1))
    dlog = infer_dlog(r)

    expected_dlog = np.log(r[..., 1] / r[..., 0])
    assert_allclose(dlog, expected_dlog)
    assert dlog.shape == (2, 3)


def test_infer_dlog_single_element():
    """Test that infer_dlog handles single-element arrays."""
    r = np.array([1.0])
    dlog = infer_dlog(r)
    # For single element, dlog calculation gives nan (0/0)
    assert np.isnan(dlog)


def test_infer_dlog_two_elements():
    """Test infer_dlog with two-element array."""
    r = np.array([1.0, 10.0])
    dlog = infer_dlog(r)
    expected_dlog = np.log(10.0 / 1.0)
    assert_allclose(dlog, expected_dlog)


def test_get_other_array_involution():
    """Test that get_other_array is an involution (applying it twice gives identity)."""
    r = np.logspace(-2, 2, 128)
    logc = 0.0

    # Apply transformation twice
    k = get_other_array(r, logc)
    r_back = get_other_array(k, logc)

    # Should get back original array
    assert_allclose(r_back, r, rtol=1e-10)


def test_get_other_array_involution_batched():
    """Test symmetry property with batched inputs."""
    # Test with 1D batch
    r = np.logspace(-2, 2, 128).reshape(2, -1)
    logc = np.random.randn(2, 1)  # Shape (2, 1) for broadcasting
    k = get_other_array(r, logc)
    r_reconstructed = get_other_array(k, logc)
    assert_allclose(r_reconstructed, r)

    # Test with 2D batch
    r = np.logspace(-2, 2, 126).reshape(2, 3, -1)
    logc = np.random.randn(2, 3, 1)  # Shape (2, 3, 1) for broadcasting
    k = get_other_array(r, logc)
    r_reconstructed = get_other_array(k, logc)
    assert_allclose(r_reconstructed, r)
