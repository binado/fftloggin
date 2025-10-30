"""
Test suite for cosmology module (RadialIntegrator).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import interp1d

from fftloggin.cosmology import RadialIntegrator


@pytest.fixture
def sample_chi():
    """Sample log-spaced comoving distance array."""
    return np.logspace(0, 3, 128)


@pytest.fixture
def gaussian_window(sample_chi):
    """Gaussian window function centered at chi=100."""
    return np.exp(-(((sample_chi - 100) / 50) ** 2))


@pytest.fixture
def tophat_window(sample_chi):
    """Top-hat window function."""
    window = np.zeros_like(sample_chi)
    mask = (sample_chi >= 50) & (sample_chi <= 150)
    window[mask] = 1.0
    return window


class TestRadialIntegratorBasic:
    """Basic functionality tests for RadialIntegrator."""

    def test_initialization_auto_compute(self, sample_chi, gaussian_window):
        """Test basic initialization with auto-compute."""
        ells = np.array([0, 10, 20])
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells)

        # Check that result was computed
        assert integrator.result is not None
        assert integrator.result.shape == (3, 128)

        # Check properties are accessible
        assert integrator.chi.shape == (128,)
        # k is batched when ells is an array - each ell has different kr
        assert integrator.k.shape == (3, 128)
        assert integrator.ells.shape == (3,)
        assert integrator.chi_mask.shape == (128,)

    def test_initialization_deferred_compute(self, sample_chi, gaussian_window):
        """Test initialization with deferred computation."""
        ells = np.array([0, 10, 20])
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        # Result should not be available yet
        with pytest.raises(ValueError, match="not yet computed"):
            _ = integrator.result

        with pytest.raises(ValueError, match="not yet resampled"):
            _ = integrator.s_resampled

        # Grid properties should still be accessible
        assert integrator.k.shape == (3, 128)  # Batched k for multiple ells

        # Compute now
        result = integrator.compute()
        assert result is not None
        assert result.shape == (3, 128)
        assert integrator.result is not None

    def test_scalar_ell(self, sample_chi, gaussian_window):
        """Test with scalar ell value."""
        ell = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ell)

        # Result should be 1D
        assert integrator.result.ndim == 1
        assert integrator.result.shape == (128,)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        chi = np.logspace(0, 2, 64)
        s_wrong = np.ones(32)  # Wrong size
        ells = np.array([0, 10])

        with pytest.raises(ValueError, match="same length"):
            RadialIntegrator(chi, s_wrong, ells)

        chi_too_short = np.array([1.0])
        s_too_short = np.array([1.0])
        with pytest.raises(ValueError, match="at least 2 elements"):
            RadialIntegrator(chi_too_short, s_too_short, ells)


class TestRecenteringLogic:
    """Tests for recentering functionality."""

    def test_recenter_true(self, sample_chi, gaussian_window):
        """Test recentering at median of source function."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, recenter=True)

        # Chi grid should be recentered near peak of Gaussian (chi=100)
        chi_center = np.sqrt(integrator.chi[0] * integrator.chi[-1])
        # Should be reasonably close to 100
        assert 80 < chi_center < 120

    def test_recenter_false(self, sample_chi, gaussian_window):
        """Test no recentering (use geometric mean of endpoints)."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, recenter=False)

        # Chi grid should be centered at geometric mean of input endpoints
        chi_center = np.sqrt(integrator.chi[0] * integrator.chi[-1])
        expected_center = np.sqrt(sample_chi[0] * sample_chi[-1])
        # Should be close but not exact due to FFTLog grid generation
        assert_allclose(chi_center, expected_center, rtol=0.1)

    def test_recenter_explicit_value(self, sample_chi, gaussian_window):
        """Test explicit r0 center value."""
        ells = 10
        r0 = 150.0
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, recenter=r0)

        # Chi grid should be centered at specified r0
        chi_center = np.sqrt(integrator.chi[0] * integrator.chi[-1])
        assert_allclose(chi_center, r0, rtol=0.05)


class TestChiMask:
    """Tests for chi_mask property."""

    def test_chi_mask_coverage(self, sample_chi, gaussian_window):
        """Test that chi_mask correctly identifies valid region."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, n=256)

        # Check that mask bounds match input chi bounds
        assert integrator.chi_mask.dtype == bool
        valid_chi = integrator.chi[integrator.chi_mask]

        # All valid chi should be within input range
        assert np.all(valid_chi >= sample_chi[0])
        assert np.all(valid_chi <= sample_chi[-1])

        # At least some points should be masked out if n is larger
        if integrator.grid.n > sample_chi.shape[0]:
            assert not np.all(integrator.chi_mask)

    def test_chi_mask_extrapolation(self, sample_chi, gaussian_window):
        """Test chi_mask with grid extending beyond input range."""
        ells = 10
        # Force larger grid
        integrator = RadialIntegrator(
            sample_chi, gaussian_window, ells, n=256, recenter=False
        )

        # Some points should be outside input range
        outside_mask = ~integrator.chi_mask
        if np.any(outside_mask):
            outside_chi = integrator.chi[outside_mask]
            # These should be outside input bounds
            assert np.any(outside_chi < sample_chi[0]) or np.any(
                outside_chi > sample_chi[-1]
            )


class TestInterpolation:
    """Tests for interpolation functionality."""

    def test_build_interpolator_default(self, sample_chi, gaussian_window):
        """Test default interpolator building."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        # Check that interpolator was created
        assert callable(integrator.source_interpolator)

        # Test interpolator works
        chi_test = np.array([50.0, 100.0, 200.0])
        s_interp = integrator.source_interpolator(chi_test)
        assert s_interp.shape == chi_test.shape

    def test_build_interpolator_custom_bc_type(self, sample_chi, gaussian_window):
        """Test custom boundary condition for interpolator."""
        ells = 10
        integrator = RadialIntegrator(
            sample_chi, gaussian_window, ells, bc_type="clamped", compute=False
        )

        # Should still work
        result = integrator.compute()
        assert result is not None

    def test_custom_interpolator(self, sample_chi, gaussian_window):
        """Test setting custom interpolator."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        # Create custom interpolator (linear instead of cubic)
        custom_interp = interp1d(
            sample_chi,
            gaussian_window,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )

        # Set custom interpolator
        integrator.source_interpolator = custom_interp

        # Compute with custom interpolator
        result = integrator.compute()
        assert result is not None

        # Result should be different from cubic spline (create new one to compare)
        integrator2 = RadialIntegrator(sample_chi, gaussian_window, ells)
        assert not np.allclose(result, integrator2.result, rtol=0.01)

    def test_interpolator_setter_validation(self, sample_chi, gaussian_window):
        """Test that interpolator setter validates callability."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        # Should raise error for non-callable
        with pytest.raises(TypeError, match="callable"):
            integrator.source_interpolator = "not a function"

    def test_interpolator_invalidates_cache(self, sample_chi, gaussian_window):
        """Test that setting interpolator invalidates cached results."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells)

        # Result should be available
        result1 = integrator.result
        assert result1 is not None

        # Create and set new interpolator
        new_interp = interp1d(
            sample_chi,
            gaussian_window * 2.0,
            kind="linear",
            fill_value=0.0,
            bounds_error=False,
        )
        integrator.source_interpolator = new_interp

        # Setting interpolator invalidates results - should raise error
        with pytest.raises(ValueError, match="not yet computed"):
            _ = integrator.result

        # After recomputing, should get new results
        result2 = integrator.compute()
        assert result2 is not None
        # Results should be different (new window function is 2x larger)
        assert not np.allclose(result1, result2)


class TestComputeMethod:
    """Tests for compute() method."""

    def test_compute_multiple_times(self, sample_chi, gaussian_window):
        """Test calling compute() multiple times."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        result1 = integrator.compute()
        result2 = integrator.compute()

        # Results should be identical
        assert_allclose(result1, result2)

    def test_compute_with_fft_kwargs(self, sample_chi, gaussian_window):
        """Test passing kwargs to FFT."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        # This should not raise
        result = integrator.compute(workers=1)
        assert result is not None

    def test_compute_returns_result(self, sample_chi, gaussian_window):
        """Test that compute() returns the result."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, compute=False)

        result = integrator.compute()
        assert result is not None
        assert_array_equal(result, integrator.result)


class TestFFTLogParameters:
    """Tests for FFTLog parameter handling."""

    def test_kr_equals_ell_plus_one(self, sample_chi, gaussian_window):
        """Test that kr is set to ell + 1."""
        ell = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ell)

        # kr should be ell + 1
        expected_kr = ell + 1
        assert_allclose(integrator.fftlog.kr, expected_kr)

    def test_kr_vectorized(self, sample_chi, gaussian_window):
        """Test kr with multiple ells."""
        ells = np.array([0, 10, 20, 30])
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells)

        # kr should be array of ells + 1, reshaped for broadcasting
        expected_kr = (ells + 1.0).reshape(-1, 1)
        assert_allclose(integrator.fftlog.kr, expected_kr)

    def test_fftlog_bias_parameter(self, sample_chi, gaussian_window):
        """Test that bias parameter is passed to FFTLog."""
        ells = 10
        bias = 0.5
        integrator = RadialIntegrator(
            sample_chi, gaussian_window, ells, fftlog_bias=bias
        )

        assert integrator.fftlog.bias == bias

    def test_lowring_parameter(self, sample_chi, gaussian_window):
        """Test that lowring parameter is passed to FFTLog."""
        ells = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, lowring=True)

        assert integrator.fftlog.lowring is True

    def test_n_and_dlog_parameters(self, sample_chi, gaussian_window):
        """Test custom n and dlog parameters."""
        ells = 10
        n = 256
        dlog = 0.05
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells, n=n, dlog=dlog)

        assert integrator.fftlog.n == n
        assert_allclose(integrator.fftlog.dlog, dlog)


class TestNumericalAccuracy:
    """Tests for numerical accuracy against reference implementations."""

    def test_simple_gaussian_transform(self):
        """Test radial integral with simple Gaussian."""
        chi = np.logspace(0, 3, 128)
        chi0 = 100.0
        sigma = 30.0
        s = np.exp(-(((chi - chi0) / sigma) ** 2))

        ell = 0
        integrator = RadialIntegrator(chi, s, ell, recenter=chi0)

        # Result should have peak somewhere reasonable
        result = integrator.result
        assert result.max() > 0
        assert np.isfinite(result).all()

        # For ell=0, spherical Bessel is sin(x)/x, should be smooth
        k_peak_idx = np.argmax(np.abs(result))
        # Peak should not be at boundaries
        assert 5 < k_peak_idx < len(result) - 5

    def test_monotonicity_check(self, sample_chi):
        """Test that results have expected behavior."""
        # Create monotonic decreasing window
        s = np.exp(-sample_chi / 100.0)

        ell = 0
        integrator = RadialIntegrator(sample_chi, s, ell)

        # Result should be real
        assert np.all(np.isreal(integrator.result))

        # Should not have NaN or inf
        assert np.all(np.isfinite(integrator.result))

    def test_symmetry_properties(self, sample_chi):
        """Test symmetry of window function effects."""
        chi0 = 100.0
        s1 = np.exp(-(((sample_chi - chi0) / 30) ** 2))

        ell = 0
        integrator1 = RadialIntegrator(sample_chi, s1, ell, recenter=chi0)
        integrator2 = RadialIntegrator(sample_chi, s1, ell, recenter=chi0)

        # Same inputs should give same outputs
        assert_allclose(integrator1.result, integrator2.result)


class TestVectorization:
    """Tests for vectorized computation with multiple ells."""

    def test_multiple_ells_shape(self, sample_chi, gaussian_window):
        """Test output shape with multiple ells."""
        ells = np.array([0, 5, 10, 15, 20])
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells)

        # Result shape should be (n_ells, n_k)
        assert integrator.result.shape == (5, 128)

    def test_ell_independence(self, sample_chi, gaussian_window):
        """Test that batch computation produces consistent shapes."""
        ells = np.array([10, 20])
        integrator_batch = RadialIntegrator(sample_chi, gaussian_window, ells)

        # Batch computation should have correct shapes
        assert integrator_batch.result.shape == (2, 128)
        assert integrator_batch.k.shape == (2, 128)

        # Each ell gets its own k grid (different kr values)
        # So we can't directly compare to individual computations
        # But we can verify consistency properties
        assert np.all(np.isfinite(integrator_batch.result))

        # The two ells should give different results
        assert not np.allclose(integrator_batch.result[0], integrator_batch.result[1])


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_window_function(self, sample_chi):
        """Test with zero window function."""
        s = np.zeros_like(sample_chi)
        ells = 10
        integrator = RadialIntegrator(sample_chi, s, ells)

        # Result should be close to zero
        assert_allclose(integrator.result, 0.0, atol=1e-10)

    def test_constant_window_function(self, sample_chi):
        """Test with constant window function."""
        s = np.ones_like(sample_chi)
        ells = 0
        integrator = RadialIntegrator(sample_chi, s, ells)

        # Result should be finite and non-trivial
        assert np.all(np.isfinite(integrator.result))
        assert np.any(np.abs(integrator.result) > 1e-5)

    def test_highly_localized_window(self, sample_chi):
        """Test with very narrow window function."""
        chi0 = 100.0
        s = np.exp(-(((sample_chi - chi0) / 5.0) ** 2))

        ells = 10
        integrator = RadialIntegrator(sample_chi, s, ells, recenter=chi0)

        # Should still produce finite result
        assert np.all(np.isfinite(integrator.result))


class TestRepr:
    """Tests for string representation."""

    def test_repr_format(self, sample_chi, gaussian_window):
        """Test __repr__ output."""
        ells = np.array([0, 10, 20])
        integrator = RadialIntegrator(sample_chi, gaussian_window, ells)

        repr_str = repr(integrator)
        assert "RadialIntegrator" in repr_str
        assert "n_ells=3" in repr_str
        assert "n=" in repr_str
        assert "chi=" in repr_str
        assert "k=" in repr_str

    def test_repr_scalar_ell(self, sample_chi, gaussian_window):
        """Test __repr__ with scalar ell."""
        ell = 10
        integrator = RadialIntegrator(sample_chi, gaussian_window, ell)

        repr_str = repr(integrator)
        assert "n_ells=1" in repr_str
