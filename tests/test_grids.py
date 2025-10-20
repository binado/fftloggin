"""
Tests for Grid class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import Grid, fht, ifht
from fftloggin.kernels import bessel_mellin_log_kernel


def test_grid_fht_basic():
    """Test Grid.fht() creates grid with correct properties."""
    grid = Grid.fht(128, dln=0.1)

    assert grid.n == 128
    assert_allclose(grid.dln, 0.1, rtol=1e-14)
    assert_allclose(grid.offset, 0.0, rtol=1e-14)
    assert_allclose(grid.central, 1.0, rtol=1e-14)
    assert grid.mode == "fht"
    assert grid.r.shape == (128,)
    assert grid.k.shape == (128,)


def test_grid_ifht_basic():
    """Test Grid.ifht() creates grid with correct properties."""
    grid = Grid.ifht(64, dln=0.05)

    assert grid.n == 64
    assert_allclose(grid.dln, 0.05, rtol=1e-14)
    assert grid.mode == "ifht"
    assert grid.k.shape == (64,)
    assert grid.r.shape == (64,)


def test_grid_logarithmic_spacing():
    """Test that grids are logarithmically spaced."""
    grid = Grid.fht(256, dln=0.1)

    # Check r grid spacing
    r_spacing = np.diff(np.log(grid.r))
    assert_allclose(r_spacing, 0.1, rtol=1e-13)

    # Check k grid spacing
    k_spacing = np.diff(np.log(grid.k))
    assert_allclose(k_spacing, 0.1, rtol=1e-13)


@pytest.mark.parametrize("offset", [0.0, 0.5, -0.5, 1.0])
def test_grid_fht_offset(offset):
    """Test Grid.fht with different offsets."""
    grid = Grid.fht(128, dln=0.1, offset=offset)

    ic = (grid.n - 1) // 2
    # r_c should be central (default 1.0)
    assert_allclose(grid.r[ic], 1.0, rtol=1e-14)
    # k_c = exp(offset) / central
    k_c_expected = np.exp(offset)
    assert_allclose(grid.k[ic], k_c_expected, rtol=1e-14)


@pytest.mark.parametrize("central", [0.5, 1.0, 2.0, 10.0])
def test_grid_fht_central(central):
    """Test Grid.fht with different central values."""
    grid = Grid.fht(128, dln=0.1, offset=0.0, central=central)

    ic = (grid.n - 1) // 2
    i = np.arange(grid.n)

    # r grid should be multiplied by central
    r_expected = central * np.exp((i - ic) * 0.1)
    assert_allclose(grid.r, r_expected, rtol=1e-14)

    # k grid should be divided by central
    k_expected = (1.0 / central) * np.exp((i - ic) * 0.1)
    assert_allclose(grid.k, k_expected, rtol=1e-14)

    # Check central values
    assert_allclose(grid.r[ic], central, rtol=1e-14)
    assert_allclose(grid.k[ic], 1.0 / central, rtol=1e-14)


def test_grid_from_r_basic():
    """Test Grid.from_r() with basic parameters."""
    # Create grid using fht
    grid_orig = Grid.fht(64, dln=0.1, offset=0.0, central=1.0)

    # Create from r array
    grid_from_r = Grid.from_r(grid_orig.r, offset=0.0)

    # Should match
    assert_allclose(grid_from_r.k, grid_orig.k, rtol=1e-14)
    assert_allclose(grid_from_r.dln, grid_orig.dln, rtol=1e-10)
    assert_allclose(grid_from_r.central, grid_orig.central, rtol=1e-14)


def test_grid_from_k_basic():
    """Test Grid.from_k() with basic parameters."""
    # Create grid using ifht
    grid_orig = Grid.ifht(64, dln=0.1, offset=0.0, central=1.0)

    # Create from k array
    grid_from_k = Grid.from_k(grid_orig.k, offset=0.0)

    # Should match
    assert_allclose(grid_from_k.r, grid_orig.r, rtol=1e-14)
    assert_allclose(grid_from_k.dln, grid_orig.dln, rtol=1e-10)
    assert_allclose(grid_from_k.central, grid_orig.central, rtol=1e-14)


def test_grid_from_r_with_central():
    """Test Grid.from_r() infers central correctly."""
    central = 2.5
    grid_orig = Grid.fht(128, dln=0.05, offset=0.0, central=central)

    # Create from r
    grid_from_r = Grid.from_r(grid_orig.r, offset=0.0)

    # Should match
    assert_allclose(grid_from_r.k, grid_orig.k, rtol=1e-14)
    assert_allclose(grid_from_r.central, central, rtol=1e-14)


def test_grid_from_r_with_logspace():
    """Test Grid.from_r() works with np.logspace arrays."""
    r = np.logspace(-3, 3, 128)

    # Get grid
    grid = Grid.from_r(r, offset=0.0)

    # Check shapes match
    assert grid.r.shape == r.shape
    assert grid.k.shape == r.shape

    # Check k is also log-spaced with same dln
    dln_r = np.mean(np.diff(np.log(r)))
    dln_k = np.mean(np.diff(np.log(grid.k)))
    assert_allclose(dln_k, dln_r, rtol=1e-10)


def test_grid_from_r_errors():
    """Test Grid.from_r() error handling."""
    # Not 1D
    with pytest.raises(ValueError, match="must be a 1D array"):
        Grid.from_r(np.ones((5, 5)), offset=0.0)

    # Too few points
    with pytest.raises(ValueError, match="must have at least 2 points"):
        Grid.from_r(np.array([1.0]), offset=0.0)

    # Not uniformly spaced in log
    r_bad = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    with pytest.raises(ValueError, match="not uniformly spaced in log"):
        Grid.from_r(r_bad, offset=0.0)


def test_grid_from_k_errors():
    """Test Grid.from_k() error handling."""
    # Not 1D
    with pytest.raises(ValueError, match="must be a 1D array"):
        Grid.from_k(np.ones((5, 5)), offset=0.0)

    # Too few points
    with pytest.raises(ValueError, match="must have at least 2 points"):
        Grid.from_k(np.array([1.0]), offset=0.0)

    # Not uniformly spaced in log
    k_bad = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    with pytest.raises(ValueError, match="not uniformly spaced in log"):
        Grid.from_k(k_bad, offset=0.0)


def test_grid_mode_validation():
    """Test that Grid validates mode parameter."""
    with pytest.raises(ValueError, match="mode must be 'fht' or 'ifht'"):
        Grid(128, 0.1, 0.0, 1.0, mode="invalid")


def test_grid_repr():
    """Test Grid string representation."""
    grid = Grid.fht(128, dln=0.123456, offset=0.5, central=2.0)

    repr_str = repr(grid)
    assert "Grid(" in repr_str
    assert "n=128" in repr_str
    assert "dln=0.123456" in repr_str
    assert "offset=0.500000" in repr_str
    assert "central=2.000000" in repr_str
    assert "mode='fht'" in repr_str


def test_grid_consistency_with_transform():
    """Test that Grid works with fht transform."""
    grid = Grid.fht(128, dln=0.1, offset=0.0)

    # Create test function on r grid
    a = np.exp(-(grid.r**2))

    # Perform transform
    A = fht(
        a, grid.dln, mu=0.5, offset=grid.offset, log_kernel=bessel_mellin_log_kernel
    )

    # Check that A has the right shape to be evaluated on k grid
    assert A.shape == grid.k.shape


def test_grid_roundtrip():
    """Test full roundtrip: Grid -> fht -> ifht -> verify."""
    grid = Grid.fht(64, dln=0.1, offset=0.0)

    # Define function on r grid
    a_original = np.exp(-(grid.r**2))

    # Forward transform
    A = fht(
        a_original,
        grid.dln,
        mu=0.5,
        offset=grid.offset,
        log_kernel=bessel_mellin_log_kernel,
    )

    # Inverse transform
    a_reconstructed = ifht(
        A, grid.dln, mu=0.5, offset=grid.offset, log_kernel=bessel_mellin_log_kernel
    )

    # Should reconstruct original (use absolute tolerance for values near zero)
    assert_allclose(a_reconstructed, a_original, rtol=1e-6, atol=1e-15)


@pytest.mark.parametrize("n", [32, 64, 127, 128, 256])
def test_grid_various_sizes(n):
    """Test Grid with various array sizes (odd and even)."""
    grid = Grid.fht(n, dln=0.1)

    assert len(grid.r) == n
    assert len(grid.k) == n

    # Check logarithmic spacing
    assert_allclose(np.diff(np.log(grid.r)), 0.1, rtol=1e-13)
    assert_allclose(np.diff(np.log(grid.k)), 0.1, rtol=1e-13)


def test_grid_positive_values():
    """Test that all grid values are positive."""
    grid = Grid.fht(128, dln=0.1, offset=0.5)

    assert np.all(grid.r > 0)
    assert np.all(grid.k > 0)


def test_grid_properties_readonly():
    """Test that grid properties are read-only."""
    grid = Grid.fht(64, dln=0.1)

    # These should all raise AttributeError
    with pytest.raises(AttributeError):
        grid.n = 100
    with pytest.raises(AttributeError):
        grid.dln = 0.2
    with pytest.raises(AttributeError):
        grid.offset = 1.0
    with pytest.raises(AttributeError):
        grid.central = 2.0
    with pytest.raises(AttributeError):
        grid.mode = "ifht"


def test_grid_fht_ifht_symmetry():
    """Test that fht and ifht grids have consistent structure."""
    grid_fht = Grid.fht(128, dln=0.1, offset=0.0)
    grid_ifht = Grid.ifht(128, dln=0.1, offset=0.0)

    # When offset=0, all central values should be 1
    ic = (128 - 1) // 2
    assert_allclose(grid_fht.r[ic], 1.0, rtol=1e-14)
    assert_allclose(grid_fht.k[ic], 1.0, rtol=1e-14)
    assert_allclose(grid_ifht.k[ic], 1.0, rtol=1e-14)
    assert_allclose(grid_ifht.r[ic], 1.0, rtol=1e-14)
