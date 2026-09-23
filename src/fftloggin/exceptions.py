"""
Exceptions and warnings for fftloggin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .kernels import Kernel

__all__ = (
    "ArgumentOutOfDomainError",
    "DomainCheckWarning",
)


class DomainCheckWarning(UserWarning):
    """Warning emitted when a concrete transform bias is outside a kernel's strip."""


class ArgumentOutOfDomainError(ValueError):
    """Exception raised when input values fall outside kernel's domain of convergence.

    This exception describes values outside the Mellin strip of convergence.
    It is used by the eager ``validate_parameters`` helper to construct a
    diagnostic warning; the JAX transform path performs no host-side checks.

    Parameters
    ----------
    s : array_like
        The actual s-values that violated the domain constraint.
    kernel : Kernel
        The kernel instance that defines the domain.
    context : str, optional
        Additional context about where/why the error occurred.

    Attributes
    ----------
    s : ndarray
        The actual s-values that violated the domain constraint.
    kernel : Kernel
        The kernel instance that defines the domain.
    domain : tuple
        The kernel's domain of convergence (lower, upper).

    Use ``fftloggin.validate_parameters`` for eager domain diagnostics.
    """

    def __init__(self, s: npt.ArrayLike, kernel: Kernel, context: str = ""):
        self.s = np.asarray(s)
        self.kernel = kernel
        self.domain = kernel.domain

        # Compute actual range from s (use real part for complex values)
        s_real = np.real(self.s)
        s_min, s_max = float(np.min(s_real)), float(np.max(s_real))
        domain_min, domain_max = self.domain

        # Handle array-valued domain bounds (for vectorized kernels)
        if np.ndim(domain_min) > 0:
            domain_min = float(np.min(domain_min))
        if np.ndim(domain_max) > 0:
            domain_max = float(np.max(domain_max))

        # Build informative message
        message = (
            f"Input values outside domain of convergence.\n"
            f"  Actual range: [{s_min:.4g}, {s_max:.4g}]\n"
            f"  Valid domain: ({domain_min:.4g}, {domain_max:.4g})"
        )

        if context:
            message += f"\n  Context: {context}"

        super().__init__(message)
