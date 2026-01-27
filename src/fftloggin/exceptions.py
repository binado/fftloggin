"""
Exceptions and warnings for fftloggin.
"""

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
    """
    Warning raised when domain validation fails in non-fatal modes.
    """


class ArgumentOutOfDomainError(ValueError):
    """Exception raised when input values fall outside kernel's domain of convergence.

    This exception indicates that the bias parameter or input array causes the
    effective s-values to fall outside the strip of convergence where the Mellin
    transform is mathematically valid.

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

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> from fftloggin.exceptions import ArgumentOutOfDomainError
    >>> kernel = BesselJKernel(mu=0)  # domain: (0, 1.5)
    >>> try:
    ...     kernel(2.0)  # Outside domain
    ... except ArgumentOutOfDomainError as e:
    ...     print(f"Values {e.s} outside domain {e.domain}")  # doctest: +SKIP
    Values [2.0] outside domain (0.0, 1.5)
    """

    def __init__(self, s: npt.ArrayLike, kernel: "Kernel", context: str = ""):
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
