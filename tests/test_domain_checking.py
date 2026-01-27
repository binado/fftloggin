"""
Test suite for domain checking workflow.

Tests the domain validation system introduced to catch invalid FFTLog
configurations at construction time rather than at transform time.
"""

import warnings

import numpy as np
import pytest

from fftloggin import (
    ArgumentOutOfDomainError,
    DomainCheckMode,
    DomainCheckWarning,
    FFTLog,
)
from fftloggin.kernels import BesselJKernel


class TestKernelContextManagers:
    """Tests for Kernel context managers."""

    def test_checking_enabled_context_manager(self):
        """checking_enabled() should temporarily enable bounds checking."""
        kernel = BesselJKernel(mu=0, check_bounds=False)
        assert kernel.check_bounds is False

        with kernel.checking_enabled() as k:
            assert k is kernel
            assert kernel.check_bounds is True
            # Should raise when bounds checking is enabled
            with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
                kernel(-1.0)  # Outside domain (0, 1.5)

        # Should be restored
        assert kernel.check_bounds is False

    def test_checking_disabled_context_manager(self):
        """checking_disabled() should temporarily disable bounds checking."""
        kernel = BesselJKernel(mu=0, check_bounds=True)
        assert kernel.check_bounds is True

        with kernel.checking_disabled() as k:
            assert k is kernel
            assert kernel.check_bounds is False
            # Should NOT raise when bounds checking is disabled
            result = kernel(-1.0)  # Outside domain, but no error
            assert result is not None

        # Should be restored
        assert kernel.check_bounds is True

    def test_context_manager_restores_on_exception(self):
        """Context manager should restore state even on exception."""
        kernel = BesselJKernel(mu=0, check_bounds=False)

        with pytest.raises(RuntimeError):
            with kernel.checking_enabled():
                assert kernel.check_bounds is True
                # Intentionally raise an exception to test that the context manager
                # properly restores the original state in the __exit__ method
                raise RuntimeError("Test exception")

        # Should still be restored
        assert kernel.check_bounds is False


class TestFFTLogDomainCheckMode:
    """Tests for FFTLog domain checking modes."""

    def test_valid_bias_no_warning(self):
        """Valid bias should not raise or warn in any mode."""
        kernel = BesselJKernel(mu=0)  # domain: (0, 1.5)
        # bias=0 means s_real=1, which is in (0, 1.5)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise in any mode
            FFTLog(kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.RAISE)
            FFTLog(kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.WARN)
            FFTLog(
                kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.SILENT
            )

    def test_invalid_bias_raises_in_raise_mode(self):
        """Invalid bias should raise ArgumentOutOfDomainError in RAISE mode."""
        kernel = BesselJKernel(mu=0)  # domain: (0, 1.5)
        # bias=1 means s_real=2, which is outside (0, 1.5)

        with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.RAISE)

    def test_invalid_bias_warns_in_warn_mode(self):
        """Invalid bias should issue DomainCheckWarning in WARN mode."""
        kernel = BesselJKernel(mu=0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.WARN)

    def test_invalid_bias_silent_in_silent_mode(self):
        """Invalid bias should not warn or raise in SILENT mode."""
        kernel = BesselJKernel(mu=0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise or warn
            fftlog = FFTLog(
                kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
            )
            assert fftlog is not None

    @pytest.mark.parametrize(
        ("mode", "expect_warn", "expect_raise"),
        [
            (DomainCheckMode.SILENT, False, False),
            (DomainCheckMode.WARN, True, False),
            (DomainCheckMode.RAISE, False, True),
        ],
    )
    def test_invalid_bias_behavior_all_modes(
        self,
        mode: DomainCheckMode,
        expect_warn: bool,
        expect_raise: bool,
    ):
        """Invalid bias should respect SILENT, WARN, and ERROR/RAISE modes."""
        kernel = BesselJKernel(mu=0)

        if expect_raise:
            with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
                FFTLog(kernel, n=64, dlog=0.1, bias=1.0, check_domain=mode)
            return

        if expect_warn:
            with pytest.warns(DomainCheckWarning, match="outside domain"):
                FFTLog(kernel, n=64, dlog=0.1, bias=1.0, check_domain=mode)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                FFTLog(kernel, n=64, dlog=0.1, bias=1.0, check_domain=mode)

    def test_string_mode_accepted(self):
        """String mode values should work."""
        kernel = BesselJKernel(mu=0)

        fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=0.0, check_domain="silent")
        assert fftlog.domain_check_mode == DomainCheckMode.SILENT

        fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=0.0, check_domain="warn")
        assert fftlog.domain_check_mode == DomainCheckMode.WARN

        fftlog = FFTLog(kernel, n=64, dlog=0.1, bias=0.0, check_domain="raise")
        assert fftlog.domain_check_mode == DomainCheckMode.RAISE

    def test_default_mode_is_warn(self):
        """Default check_domain mode should be WARN."""
        kernel = BesselJKernel(mu=0)

        # Invalid bias with default mode should warn
        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog(kernel, n=64, dlog=0.1, bias=1.0)


class TestFFTLogCheckDomainMethod:
    """Tests for FFTLog.check_domain() method."""

    def test_check_domain_returns_true_for_valid(self):
        """check_domain() should return True for valid configurations."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.SILENT
        )
        assert fftlog.check_domain() is True

    def test_check_domain_returns_false_for_invalid(self):
        """check_domain() should return False for invalid configurations."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
        )
        assert fftlog.check_domain() is False

    def test_check_domain_raises_when_requested(self):
        """check_domain(raise_exception=True) should raise on invalid."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
        )

        with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
            fftlog.check_domain(raise_exception=True)

    def test_check_domain_no_raise_when_valid(self):
        """check_domain(raise_exception=True) should not raise when valid."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.SILENT
        )

        result = fftlog.check_domain(raise_exception=True)
        assert result is True


class TestFFTLogPropertySetterValidation:
    """Tests for validation on property setters."""

    def test_kernel_setter_validates(self):
        """Setting kernel property should trigger validation."""
        kernel_valid = BesselJKernel(mu=0)

        # Create with valid kernel, bias=0 is valid for mu=0
        fftlog = FFTLog(
            kernel_valid, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.RAISE
        )

        # Setting kernel should work if still valid
        fftlog.kernel = BesselJKernel(mu=1)  # bias=0 still valid
        assert fftlog.check_domain() is True

    def test_bias_setter_validates(self):
        """Setting bias property should trigger validation."""
        kernel = BesselJKernel(mu=0)

        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.RAISE
        )

        # Setting invalid bias should raise
        with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
            fftlog.bias = 1.0

    def test_bias_setter_warns_in_warn_mode(self):
        """Setting invalid bias should warn in WARN mode."""
        kernel = BesselJKernel(mu=0)

        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=0.0, check_domain=DomainCheckMode.WARN
        )

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            fftlog.bias = 1.0


class TestFFTLogFromArray:
    """Tests for FFTLog.from_array() with check_domain."""

    def test_from_array_accepts_check_domain(self):
        """from_array() should accept check_domain parameter."""
        r = np.logspace(-2, 2, 64)
        kernel = BesselJKernel(mu=0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fftlog = FFTLog.from_array(
                r, kernel, bias=0.0, check_domain=DomainCheckMode.SILENT
            )
            assert fftlog.domain_check_mode == DomainCheckMode.SILENT

    def test_from_array_default_warns(self):
        """from_array() should warn by default for invalid bias."""
        r = np.logspace(-2, 2, 64)
        kernel = BesselJKernel(mu=0)

        with pytest.warns(DomainCheckWarning, match="outside domain"):
            FFTLog.from_array(r, kernel, bias=1.0)

    def test_from_array_raise_mode(self):
        """from_array() should raise in RAISE mode."""
        r = np.logspace(-2, 2, 64)
        kernel = BesselJKernel(mu=0)

        with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
            FFTLog.from_array(r, kernel, bias=1.0, check_domain=DomainCheckMode.RAISE)


class TestDerivativeKernelInFFTLog:
    """Tests for derivative kernels with FFTLog domain checking."""

    def test_derivative_kernel_validation(self):
        """Derivative kernels should be properly validated in FFTLog."""
        kernel = BesselJKernel(mu=0).derive(1)

        # For 1st derivative, effective bias = bias - 1
        # bias=1.5 -> effective=0.5 -> s_real=1.5 (boundary, should fail strict)
        # bias=1.0 -> effective=0.0 -> s_real=1.0 is in (0, 1.5), valid

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fftlog = FFTLog(
                kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.RAISE
            )
            assert fftlog.check_domain() is True

    def test_derivative_kernel_invalid_bias(self):
        """Derivative kernel with invalid bias should fail validation."""
        kernel = BesselJKernel(mu=0).derive(1)

        # bias=-0.5 -> effective=-1.5 -> s_real=-0.5 is NOT in (0, 1.5)
        with pytest.raises(ArgumentOutOfDomainError, match="outside domain"):
            FFTLog(
                kernel, n=64, dlog=0.1, bias=-0.5, check_domain=DomainCheckMode.RAISE
            )


class TestArgumentOutOfDomainError:
    """Tests for ArgumentOutOfDomainError exception attributes."""

    def test_exception_stores_s_values(self):
        """Exception should store the s values that violated domain."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
        )

        try:
            fftlog.check_domain(raise_exception=True)
        except ArgumentOutOfDomainError as e:
            assert hasattr(e, "s")
            assert np.allclose(e.s, 2.0)  # s = 1 + bias = 2
        else:
            pytest.fail("Expected ArgumentOutOfDomainError")

    def test_exception_stores_kernel(self):
        """Exception should store the kernel instance."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
        )

        try:
            fftlog.check_domain(raise_exception=True)
        except ArgumentOutOfDomainError as e:
            assert hasattr(e, "kernel")
            assert e.kernel is fftlog.kernel
        else:
            pytest.fail("Expected ArgumentOutOfDomainError")

    def test_exception_stores_domain(self):
        """Exception should store the domain tuple."""
        kernel = BesselJKernel(mu=0)
        fftlog = FFTLog(
            kernel, n=64, dlog=0.1, bias=1.0, check_domain=DomainCheckMode.SILENT
        )

        try:
            fftlog.check_domain(raise_exception=True)
        except ArgumentOutOfDomainError as e:
            assert hasattr(e, "domain")
            domain_min, domain_max = e.domain
            assert np.allclose(domain_min, 0.0)  # -mu for mu=0
            assert np.allclose(domain_max, 1.5)
        else:
            pytest.fail("Expected ArgumentOutOfDomainError")

    def test_exception_message_format(self):
        """Exception message should include range and domain."""
        kernel = BesselJKernel(mu=0)

        try:
            kernel(2.0)  # Outside domain (0, 1.5)
        except ArgumentOutOfDomainError as e:
            message = str(e)
            assert "Actual range:" in message
            assert "Valid domain:" in message
            assert "2" in message  # The s_value
            assert "1.5" in message  # The upper bound
        else:
            pytest.fail("Expected ArgumentOutOfDomainError")

    def test_exception_is_subclass_of_valueerror(self):
        """ArgumentOutOfDomainError should be a ValueError subclass."""
        kernel = BesselJKernel(mu=0)

        # Should be catchable as ValueError for backward compatibility
        with pytest.raises(ValueError):
            kernel(2.0)
