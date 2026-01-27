"""
Mellin transform kernels for FFTLog algorithm.

This module provides kernel functions that compute the Mellin transform
of various integral kernels used in generalized FFTLog transforms.
"""

from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
from scipy import special

from .exceptions import ArgumentOutOfDomainError
from .utils import safe_broadcast

__all__ = (
    "ArgumentOutOfDomainError",
    "Kernel",
    "BesselJKernel",
    "Derivative",
)


LOG_2 = np.log(2)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2)


class Kernel:
    """
    Base class for Mellin transform kernels.

    A kernel represents the Mellin transform of an integral kernel function.
    Kernels have a domain (strip of convergence) in the complex plane where
    the transform is well-defined.

    Parameters
    ----------
    check_bounds : bool, optional
        If True, validate that input values are within the domain.
        Default is True.

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> kernel = BesselJKernel(mu=0.5)
    >>> # Get second derivative
    >>> d2_kernel = kernel.derive(2)

    Notes
    -----
    The domain (strip of convergence) is a range in the complex plane where
    the Mellin transform is well-defined and analytic.

    See Also
    --------
    BesselJKernel : Standard Hankel transform kernel using Bessel functions
    Derivative : Compute derivatives of kernels
    """

    def __init__(self, check_bounds: bool = True) -> None:
        self.check_bounds = check_bounds

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Domain of convergence (inf, sup) where the transform is defined.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Lower and upper bounds of the domain (strip of convergence)
            in the complex plane.
        """
        return (-np.inf, np.inf)

    def forward(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute the Mellin transform at s.

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        ndarray
            Mellin transform evaluated at s.

        Notes
        -----
        Bounds checking is not performed in this method; it is done in __call__.
        Subclasses should override this method to implement the Mellin transform.
        """
        raise NotImplementedError

    def is_in_domain(self, s: npt.ArrayLike) -> bool:
        """
        Check if s is within the domain of convergence.

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        bool
            True if all values are within the domain.
        """
        inf, sup = self.domain
        # Domain applies to the real part of s
        s_real = np.real(s)
        # Reshape inf/sup to have trailing dimensions for proper broadcasting
        inf, _ = safe_broadcast(inf, s)
        sup, _ = safe_broadcast(sup, s)
        in_bounds = (s_real >= inf) & (s_real <= sup)
        return bool(np.all(in_bounds))

    def __call__(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute the Mellin transform at s with optional bounds checking.

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
        ArgumentOutOfDomainError
            If s is outside the domain of convergence and check_bounds is True.
        """
        if self.check_bounds:
            if not self.is_in_domain(s):
                raise ArgumentOutOfDomainError(
                    s=s, kernel=self, context="when calling kernel directly"
                )

        s = np.asarray(s)
        return self.forward(s)

    def derive(self, order: int = 1) -> "Kernel":
        r"""
        Return the nth derivative of this kernel.

        Uses the Mellin transform property:

        .. math::

            M\left[\frac{d^n}{dr^n} f\right](s) = (-1)^n \frac{\Gamma(s)}{\Gamma(s-n)} M[f](s-n)

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

        Notes
        -----
        The derivative is computed using the Mellin transform property, which
        relates derivatives in real space to shifts in the complex frequency
        domain.
        """
        if order == 0:
            return self
        return Derivative(self, order)

    @contextmanager
    def checking_enabled(self):
        """
        Context manager that temporarily enables bounds checking.

        Within this context, the kernel's ``check_bounds`` attribute is set
        to ``True``, and restored to its original value upon exit.

        Yields
        ------
        Kernel
            This kernel instance with bounds checking enabled.

        Examples
        --------
        >>> kernel = BesselJKernel(mu=0.5, check_bounds=False)
        >>> with kernel.checking_enabled():
        ...     pass  # Will validate bounds when kernel is called
        >>> # check_bounds is restored to False here
        """
        original = self.check_bounds
        self.check_bounds = True
        try:
            yield self
        finally:
            self.check_bounds = original

    @contextmanager
    def checking_disabled(self):
        """
        Context manager that temporarily disables bounds checking.

        Within this context, the kernel's ``check_bounds`` attribute is set
        to ``False``, and restored to its original value upon exit.

        Yields
        ------
        Kernel
            This kernel instance with bounds checking disabled.

        Examples
        --------
        >>> kernel = BesselJKernel(mu=0.5, check_bounds=True)
        >>> with kernel.checking_disabled():
        ...     pass  # Will skip bounds validation when kernel is called
        >>> # check_bounds is restored to True here
        """
        original = self.check_bounds
        self.check_bounds = False
        try:
            yield self
        finally:
            self.check_bounds = original


class Derivative(Kernel):
    r"""
    Kernel representing the nth derivative of another kernel.

    This class implements the Mellin transform property for derivatives:

    .. math::

        M\left[\frac{d^n}{dr^n} f\right](s) = (-1)^n \frac{\Gamma(s)}{\Gamma(s-n)} M[f](s-n)

    Parameters
    ----------
    transform : Kernel
        The base kernel to differentiate.
    order : int
        Order of the derivative (must be >= 1).

    Raises
    ------
    ValueError
        If order < 1.

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> kernel = BesselJKernel(mu=0.5)
    >>> d_kernel = Derivative(kernel, 1)  # First derivative
    >>> result = d_kernel.forward(2.0)

    Notes
    -----
    The derivative kernel inherits the domain of convergence from the base
    kernel, adjusted for the derivative order.

    See Also
    --------
    Kernel.derive : Recommended way to compute derivatives
    """

    def __init__(self, transform: Kernel, order: int) -> None:
        super().__init__(check_bounds=transform.check_bounds)
        self.transform = transform
        if order < 1:
            raise ValueError(
                "Expected derivative order to be an integer greater than or equal to 1"
            )

        self.order = order

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        inf, sup = self.transform.domain
        return np.asarray(inf) + self.order, np.asarray(sup) + self.order

    def is_in_domain(self, s: npt.ArrayLike) -> bool:
        s = np.asarray(s)
        return self.transform.is_in_domain(s - self.order)

    def forward(self, s: npt.ArrayLike) -> np.ndarray:
        s = np.asarray(s)
        sign = 1 - 2 * (self.order % 2)
        return (
            sign
            * np.prod(s.reshape(*s.shape, 1) - np.arange(1, self.order + 1), axis=-1)
            * self.transform.forward(s - self.order)
        )


class BesselJKernel(Kernel):
    r"""
    Mellin transform kernel for Bessel function :math:`J_\\mu`.

    This kernel represents the standard Hankel transform with Bessel functions.
    The Mellin transform is given by:

    .. math::

        M[J_\\mu](s) = 2^{s-1} \\frac{\\Gamma\\left(\\frac{\\mu+s}{2}\\right)}{\\Gamma\\left(\\frac{\\mu+2-s}{2}\\right)}

    Parameters
    ----------
    mu : array_like
        Order of the Bessel function. Can be scalar or array.
    check_bounds : bool, optional
        If True, validate that input values are within the domain.
        Default is True.

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> import numpy as np
    >>> # Single order
    >>> kernel = BesselJKernel(mu=0.5)
    >>> # Multiple orders (for vectorized transforms)
    >>> kernels = BesselJKernel(mu=np.array([0, 0.5, 1.0]))
    >>> # Compute Mellin transform at s = 1.0
    >>> result = kernel.forward(1.0)

    Notes
    -----
    The domain of convergence (strip) is :math:`(-\\mu, 1.5)` in the complex
    :math:`s`-plane. The kernel uses log-gamma functions for numerical stability
    in the Mellin transform computation.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    See Also
    --------
    SphericalBesselJKernel : Related spherical Bessel function kernel
    Derivative : Compute derivatives of kernels
    """

    def __init__(self, mu: npt.ArrayLike, check_bounds: bool = True) -> None:
        super().__init__(check_bounds=check_bounds)
        self.mu = np.asarray(mu)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Domain of convergence: (-mu, 1.5)."""
        return (-self.mu, 1.5 * np.ones_like(self.mu))

    def forward(self, s: npt.ArrayLike) -> np.ndarray:
        """
        Compute the Mellin transform.

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        ndarray
            Mellin transform evaluated at s.

        Notes
        -----
        The implementation uses log-gamma functions for numerical stability
        to avoid overflow/underflow in direct gamma computations.
        """
        # Reshape mu and s to enable proper broadcasting
        mu, s = safe_broadcast(self.mu, s)
        logforward = (
            LOG_2 * (s - 1)
            + special.loggamma(0.5 * (mu + s))
            - special.loggamma(0.5 * (mu + 2 - s))
        )
        return np.exp(logforward)


class SphericalBesselJKernel(BesselJKernel):
    r"""
    Mellin transform of the spherical Bessel function of the first kind, :math:`j_\mu`.
    It is related to :math:`J_\mu` by
    .. math::

        j_\ell(x) = \sqrt{\frac{\pi}{2x}} * J_{\ell+1/2}(x)

    Their Mellin transforms are therefore related by
    .. math::

        M[j_\ell](s) = \sqrt{\frac{\pi}{2}} * M[J_{\ell+1/2}]\left( s + \frac{1}{2} \right)

    """

    def __init__(self, ell: npt.ArrayLike, check_bounds: bool = True) -> None:
        mu = np.asarray(ell) + 0.5
        super().__init__(mu, check_bounds=check_bounds)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        inf, sup = super().domain
        return (np.asarray(inf) + 0.5, np.asarray(sup) + 0.5)

    def forward(self, s: npt.ArrayLike) -> np.ndarray:
        s = np.asarray(s)
        return super().forward(s - 0.5) * SQRT_PI_OVER_2
