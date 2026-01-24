"""
Tests to verify that FFTLog functions properly use kernel bounds checking.

This test suite specifically verifies that the changes to use kernel.__call__
instead of kernel.forward enable proper bounds checking in the FFTLog algorithm.
"""

import numpy as np
import pytest

from fftloggin.fftlog import FFTLog, compute_kernel_coefficients, optimal_logcenter
from fftloggin.kernels import BesselJKernel


def test_optimal_logcenter_respects_bounds():
    """Test that optimal_logcenter respects kernel bounds checking."""
    mu = 0.5
    dlog = 0.1

    # With bounds checking enabled (default)
    kernel = BesselJKernel(mu, check_bounds=True)

    # This should work - within bounds
    result = optimal_logcenter(kernel, dlog, bias=0.0)
    assert np.isfinite(result)

    # With extreme bias that pushes outside strip, should raise
    kernel_strict = BesselJKernel(mu, check_bounds=True)
    # For BesselJKernel(0.5), strip is (-0.5, 1.5)
    # s = 1j * pi / dlog + 1 ≈ 1 + 31.4j
    # s + bias needs real part in (-0.5, 1.5)
    # So bias = -10 would give s + bias ≈ -9 + 31.4j, outside strip
    with pytest.raises(ValueError, match="Input array outside strip"):
        optimal_logcenter(kernel_strict, dlog, bias=-10.0)

    # With bounds checking disabled, should work
    kernel_permissive = BesselJKernel(mu, check_bounds=False)
    result = optimal_logcenter(kernel_permissive, dlog, bias=-10.0)
    # Should return a result even though we're outside the strip
    assert result is not None


def test_compute_kernel_coefficients_respects_bounds():
    """Test that compute_kernel_coefficients respects kernel bounds checking."""
    mu = 0.5
    n = 128
    dlog = 0.1
    kr = 1.0

    # With bounds checking enabled (default)
    kernel = BesselJKernel(mu, check_bounds=True)

    # This should work - within bounds for typical values
    coeffs = compute_kernel_coefficients(kernel, n, kr, dlog, bias=0.0)
    assert coeffs.shape == (n // 2 + 1,)
    assert np.all(np.isfinite(coeffs))

    # With extreme bias that pushes outside strip, should raise
    kernel_strict = BesselJKernel(mu, check_bounds=True)
    # For BesselJKernel(0.5), strip is (-0.5, 1.5)
    # The values of s range from 1 to something complex
    # s + bias with large negative bias will be outside strip
    with pytest.raises(ValueError, match="Input array outside strip"):
        compute_kernel_coefficients(kernel_strict, n, kr, dlog, bias=-10.0)

    # With bounds checking disabled, should work
    kernel_permissive = BesselJKernel(mu, check_bounds=False)
    coeffs = compute_kernel_coefficients(kernel_permissive, n, kr, dlog, bias=-10.0)
    # Should return coefficients even though we're outside the strip
    assert coeffs.shape == (n // 2 + 1,)


def test_fftlog_instance_respects_bounds():
    """Test that FFTLog instance respects kernel bounds checking."""
    mu = 0.5
    n = 128
    dlog = 0.1

    # Create test data
    r = np.logspace(-2, 2, n)
    a = np.exp(-((r / 1.0) ** 2))

    # With bounds checking enabled (default)
    kernel = BesselJKernel(mu, check_bounds=True)

    # This should work - within bounds for typical values
    fftlog = FFTLog(kernel=kernel, n=n, dlog=dlog, bias=0.0, kr=1.0, lowring=False)
    result = fftlog.forward(a)
    assert result.shape == (n,)
    assert np.all(np.isfinite(result))

    # With extreme bias that pushes outside strip, should raise during initialization
    # because kernel_coefficients are computed as a cached property
    kernel_strict = BesselJKernel(mu, check_bounds=True)
    fftlog_bad = FFTLog(
        kernel=kernel_strict, n=n, dlog=dlog, bias=-10.0, kr=1.0, lowring=False
    )
    with pytest.raises(ValueError, match="Input array outside strip"):
        # Force computation of kernel_coefficients
        _ = fftlog_bad.kernel_coefficients

    # With bounds checking disabled, should work
    kernel_permissive = BesselJKernel(mu, check_bounds=False)
    fftlog_permissive = FFTLog(
        kernel=kernel_permissive, n=n, dlog=dlog, bias=-10.0, kr=1.0, lowring=False
    )
    result = fftlog_permissive.forward(a)
    # Should return a result even though we're outside the strip
    assert result.shape == (n,)


def test_lowring_optimal_logcenter_respects_bounds():
    """Test that FFTLog with lowring=True respects bounds in optimal_logcenter."""
    mu = 0.5
    n = 128
    dlog = 0.1

    # Create test data
    r = np.logspace(-2, 2, n)
    a = np.exp(-((r / 1.0) ** 2))

    # With bounds checking enabled and lowring=True
    kernel = BesselJKernel(mu, check_bounds=True)

    # This should work
    fftlog = FFTLog(kernel=kernel, n=n, dlog=dlog, bias=0.0, kr=1.0, lowring=True)
    result = fftlog.forward(a)
    assert result.shape == (n,)

    # With extreme bias and lowring=True, should raise when computing optimal_logcenter
    kernel_strict = BesselJKernel(mu, check_bounds=True)
    with pytest.raises(ValueError, match="Input array outside strip"):
        fftlog_bad = FFTLog(
            kernel=kernel_strict, n=n, dlog=dlog, bias=-10.0, kr=1.0, lowring=True
        )
        # Force computation of kr which calls optimal_logcenter
        _ = fftlog_bad.kr
