"""
Test suite for domain checking workflow.

Tests the domain validation system that catches invalid FFTLog
configurations at construction time via warnings.
"""

import warnings

import numpy as np
import pytest

from fftloggin import (
    ArgumentOutOfDomainError,
    DomainCheckWarning,
    FFTLog,
)
from fftloggin.kernels import BesselJKernel


class TestFFTLogDomainValidation:
    """Tests for FFTLog domain validation (always warns)."""

    def test_valid_bias_no_warning(self):
        """Valid bias should not warn."""
        kernel = BesselJKernel(mu=0)  # domain: (0, 1.5)
        # bias=0 means s_real=1, which is in (0, 1.5)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FFTLog(kernel, n=64, dlog=0.1, bias=0.0)

    def test_invalid_bias_warns(self):
        """Invalid bias should issue DomainCheckWarning."""
        kernel = BesselJKernel(mu=0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=1.0)

    def test_default_warns_for_invalid_bias(self):
        """Default behavior should warn for invalid bias."""
        kernel = BesselJKernel(mu=0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=1.0)


class TestFFTLogCheckDomainMethod:
    """Tests for FFTLog.check_domain() method."""

    def test_check_domain_returns_true_for_valid(self):
        """check_domain() should return True for valid configurations."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=0.0)
        assert fftlog.check_domain() is True

    def test_check_domain_returns_false_for_invalid(self):
        """check_domain() should return False for invalid configurations."""
        kernel = BesselJKernel(mu=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=1.0)
        assert fftlog.check_domain() is False


class TestFFTLogPropertySetterValidation:
    """Tests for validation on property setters."""

    def test_kernel_setter_validates(self):
        """Setting kernel property should trigger validation."""
        kernel_valid = BesselJKernel(mu=0)

        # Create with valid kernel, bias=0 is valid for mu=0
        fftlog = FFTLog(kernel_valid, n=64, dlog=0.1, bias=0.0)

        # Setting kernel should work if still valid
        fftlog.kernel = BesselJKernel(mu=1)  # bias=0 still valid
        assert fftlog.check_domain() is True

    def test_bias_setter_warns_for_invalid(self):
        """Setting invalid bias should warn."""
        kernel = BesselJKernel(mu=0)

        fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=0.0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            fftlog.bias = 1.0


class TestFFTLogFromArray:
    """Tests for FFTLog.from_array() domain validation."""

    def test_from_array_default_warns(self):
        """from_array() should warn by default for invalid bias."""
        r = np.logspace(-2, 2, 64)
        kernel = BesselJKernel(mu=0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog.from_array(r, kernel, bias=1.0)


class TestDerivativeKernelInFFTLog:
    """Tests for derivative kernels with FFTLog domain checking."""

    def test_derivative_kernel_validation(self):
        """Derivative kernels should be properly validated in FFTLog."""
        kernel = BesselJKernel(mu=0).derive(1)

        # For 1st derivative, effective bias = bias - 1
        # bias=1.0 -> effective=0.0 -> s_real=1.0 is in (0, 1.5), valid

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=1.0)
            assert fftlog.check_domain() is True

    def test_derivative_kernel_invalid_bias(self):
        """Derivative kernel with invalid bias should warn."""
        kernel = BesselJKernel(mu=0).derive(1)

        # bias=-0.5 -> effective=-1.5 -> s_real=-0.5 is NOT in (0, 1.5)
        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=-0.5)


class TestArgumentOutOfDomainError:
    """Tests for ArgumentOutOfDomainError exception attributes."""

    def test_exception_is_subclass_of_valueerror(self):
        """ArgumentOutOfDomainError should be a ValueError subclass."""
        assert issubclass(ArgumentOutOfDomainError, ValueError)
