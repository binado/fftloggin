"""
Mellin transform kernels for FFTLog algorithm.

This module provides kernel functions that compute the Mellin transform
of various integral kernels used in generalized FFTLog transforms.
"""

import numpy as np
import numpy.typing as npt
from scipy import special

from .utils import outer_broadcast

__all__ = (
    "Kernel",
    "BesselJKernel",
    "Derivative",
)

LOG_2 = np.log(2)


class Kernel:
    """
    Base class for Mellin transform kernels.

    A kernel represents the Mellin transform of an integral kernel function.
    Kernels have a strip of convergence in the complex plane where the
    transform is well-defined.

    Parameters
    ----------
    bias : float, optional
        Power-law bias exponent for improved numerical stability (default: 0.0).

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> kernel = BesselJKernel(mu=0.5, bias=0.0)
    >>> # Get second derivative
    >>> d2_kernel = kernel.derive(2)
    """

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    @property
    def strip(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Strip of convergence (inf, sup) where the transform is defined.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Lower and upper bounds of the strip in the complex plane.
        """
        raise NotImplementedError

    def _forward(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute the Mellin transform at s (internal implementation).

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        ndarray
            Mellin transform evaluated at s.
        """
        raise NotImplementedError

    def forward(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute the Mellin transform at s with bounds checking.

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        ndarray
            Mellin transform evaluated at s.

        Raises
        ------
        ValueError
            If s is outside the strip of convergence.
        """
        s = np.asarray(s)
        inf, sup = self.strip
        # Strip of convergence applies to the real part of s
        s_real = np.real(s)
        # Reshape inf/sup to have trailing dimensions for proper broadcasting
        inf, _ = outer_broadcast(inf, s)
        sup, _ = outer_broadcast(sup, s)
        in_bounds = (s_real >= inf) & (s_real <= sup)
        if not np.all(in_bounds):
            raise ValueError("Input array outside strip of definition of the transform")

        return self._forward(s)

    def derive(self, order: int = 1) -> "Kernel":
        """
        Return the nth derivative of this kernel.

        Uses the relationship: M[d^n/dr^n f](s) = (-1)^n * (s-n)...(s-1) * M[f](s-n)

        Parameters
        ----------
        order : int, optional
            Order of derivative (must be >= 0). Default is 1.

        Returns
        -------
        Kernel
            A new Kernel representing the nth derivative.
            If order is 0, returns self unchanged.

        Examples
        --------
        >>> kernel = BesselJKernel(mu=0.5)
        >>> d_kernel = kernel.derive(1)  # First derivative
        >>> d2_kernel = kernel.derive(2)  # Second derivative
        """
        if order == 0:
            return self
        return Derivative(self, order)


class Derivative(Kernel):
    def __init__(self, transform: Kernel, order: int) -> None:
        super().__init__(bias=transform.bias)
        self.transform = transform
        if order < 1:
            raise ValueError(
                "Expected derivative order to be an integer greater than or equal to 1"
            )

        self.order = order

    @property
    def strip(self) -> tuple[float, float]:
        inf, sup = self.transform.strip
        return (inf + self.order, sup + self.order)

    def _forward(self, s: npt.ArrayLike) -> np.ndarray:
        s = np.asarray(s)
        sign = 1 - 2 * (self.order % 2)
        return (
            sign
            * special.poch(s - self.order, self.order)
            * self.transform.forward(s - self.order)
        )


class BesselJKernel(Kernel):
    """
    Mellin transform kernel for Bessel function J_μ.

    This kernel represents the standard Hankel transform with Bessel functions.
    The Mellin transform is:

        M[J_μ](s) = 2^(s-1) * Γ((μ+s)/2) / Γ((μ+2-s)/2)

    Parameters
    ----------
    mu : array_like
        Order of the Bessel function. Can be scalar or array.
    bias : float, optional
        Power-law bias exponent for improved numerical stability (default: 0.0).

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> import numpy as np
    >>> # Single order
    >>> kernel = BesselJKernel(mu=0.5)
    >>> # Multiple orders (for vectorized transforms)
    >>> kernels = BesselJKernel(mu=np.array([0, 0.5, 1.0]))
    >>> # With bias
    >>> kernel_biased = BesselJKernel(mu=0.5, bias=0.1)

    Notes
    -----
    The strip of convergence is (-μ, 1.5) in the complex s-plane.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
    """

    def __init__(self, mu: npt.ArrayLike, bias: float = 0.0) -> None:
        super().__init__(bias=bias)
        self.mu = np.asarray(mu)

    @property
    def strip(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Strip of convergence: (-μ, 1.5)."""
        return (-self.mu, 1.5)

    def _forward(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute M[J_μ](s) = 2^(s-1) * Γ((μ+s)/2) / Γ((μ+2-s)/2).

        Implementation uses log-gamma for numerical stability.
        """
        s = np.asarray(s)
        # Reshape mu and s to enable proper broadcasting: (mu_shape, ..., s_shape)
        if self.mu.ndim > 0 and s.ndim > 0:
            # Add trailing dimensions to mu and leading dimensions to s
            mu, s = outer_broadcast(self.mu, s)
        else:
            mu = self.mu
        logforward = (
            LOG_2 * (s - 1)
            + special.loggamma(0.5 * (mu + s))
            - special.loggamma(0.5 * (mu + 2 - s))
        )
        return np.exp(logforward)
