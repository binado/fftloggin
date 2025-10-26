"""
Test suite for Grid class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.fftlog import FFTLog
from fftloggin.grids import Grid, get_other_array, infer_dlog, infer_logc
from fftloggin.kernels import BesselJKernel


def test_grid_construction():
    """Test basic Grid construction."""
    r = np.logspace(-2, 2, 128)
    k = np.logspace(-1, 1, 128)
    grid = Grid(r, k)

    assert len(grid.r) == 128
    assert len(grid.k) == 128
    assert grid.n == 128


def test_grid_properties():
    """Test Grid properties."""
    r = np.logspace(-2, 2, 128)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0))
    grid = fftlog.create_grid(r=r)

    # Test that basic properties are accessible
    assert grid.n == 128
    assert np.isclose(grid.dlog, np.log(r[1] / r[0]))
    assert np.isclose(grid.rcenter, np.sqrt(r.min() * r.max()))
    assert np.isclose(grid.kcenter, np.sqrt(grid.k.min() * grid.k.max()))
    assert isinstance(grid.logc, float)


def test_grid_copy():
    """Test Grid copy method."""
    r = np.logspace(-2, 2, 128)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0))
    grid = fftlog.create_grid(r=r)

    grid_copy = grid.copy()
    assert np.array_equal(grid_copy.r, grid.r)
    assert np.array_equal(grid_copy.k, grid.k)
    assert grid_copy is not grid


def test_grid_mismatched_lengths():
    """Test that Grid raises error for mismatched array lengths."""
    r = np.logspace(-2, 2, 128)
    k = np.logspace(-1, 1, 64)  # Wrong length

    with pytest.raises(ValueError, match="must have the same length"):
        Grid(r, k)


def test_grid_repr():
    """Test Grid string representation."""
    r = np.logspace(-2, 2, 128)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0))
    grid = fftlog.create_grid(r=r)

    repr_str = repr(grid)
    assert "Grid" in repr_str
    assert "n=128" in repr_str


def test_get_other_array():
    """Test get_other_array function."""
    r = np.logspace(-2, 2, 128)
    logc = 0.0
    k = get_other_array(r, logc)

    # k should be symmetric with respect to reversal
    k_reconstructed = get_other_array(k, logc)
    assert_allclose(k_reconstructed, r)


def test_infer_dlog():
    """Test infer_dlog function."""
    r = np.logspace(-2, 2, 128)
    dlog = infer_dlog(r)

    expected_dlog = np.log(r[1] / r[0])
    assert np.isclose(dlog, expected_dlog)


def test_infer_logc_with_logc():
    """Test infer_logc with logc parameter."""
    r = np.logspace(-2, 2, 128)
    logc = infer_logc(r, logc=0.5)
    assert logc == 0.5


def test_infer_logc_with_ycenter():
    """Test infer_logc with ycenter parameter."""
    r = np.logspace(-2, 2, 128)
    ycenter = 1.0
    logc = infer_logc(r, ycenter=ycenter)

    r_center = np.sqrt(r.min() * r.max())
    expected_logc = np.log(ycenter * r_center)
    assert np.isclose(logc, expected_logc)


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
