"""
Mellin transform kernels for FFTLog algorithm.

This module provides kernel functions that compute the Mellin transform
of various integral kernels used in generalized FFTLog transforms.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
from array_api_compat import array_namespace, is_array_api_obj
from scipy import special

from .exceptions import ArgumentOutOfDomainError

__all__ = (
    "ArgumentOutOfDomainError",
    "Kernel",
    "BesselJKernel",
    "Derivative",
    "CombinedKernel",
)


LOG_2 = np.log(2)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2)


def _get_namespace(*arrays: Any) -> Any:
    for arr in arrays:
        if is_array_api_obj(arr):
            return array_namespace(arr)
    return np


def _safe_broadcast_namespace(left: Any, right: Any, xp: Any) -> tuple[Any, Any]:
    left = xp.asarray(left)
    right = xp.asarray(right)

    if left.ndim > 0 and right.ndim > 0:
        n_trailing_ones = 0
        for dim in left.shape[::-1]:
            if dim == 1:
                n_trailing_ones += 1
            else:
                break
        n_dims_to_add = max(0, right.ndim - n_trailing_ones)
        if n_dims_to_add > 0:
            left = left.reshape(left.shape + (1,) * n_dims_to_add)

    return left, right


def _kernel_forward(kernel: "Kernel", s: Any) -> Any:
    # Ensure forward() always receives an array object with a discoverable namespace.
    if not is_array_api_obj(s):
        s = np.asarray(s)
    return kernel.forward(s)


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
        tuple[float | NDArray, float | NDArray]
            Lower and upper bounds of the strip of convergence in the
            complex plane. The base class returns plain floats; subclasses
            may return arrays when the bounds depend on vectorized parameters
            (e.g. a batch of ``mu`` values in :class:`BesselJKernel`).
        """
        return (-np.inf, np.inf)

    def forward(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
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

    def is_in_domain(self, s: npt.ArrayLike, xp: Any | None = None) -> bool:
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
        xp = xp or _get_namespace(s)
        inf, sup = self.domain
        s_real = xp.real(xp.asarray(s))
        inf, _ = _safe_broadcast_namespace(inf, s_real, xp=xp)
        sup, _ = _safe_broadcast_namespace(sup, s_real, xp=xp)
        in_bounds = (s_real >= inf) & (s_real <= sup)
        return bool(np.all(np.asarray(in_bounds)))

    def __call__(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
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
        xp = xp or _get_namespace(s)
        if self.check_bounds:
            if not self.is_in_domain(s, xp=xp):
                raise ArgumentOutOfDomainError(
                    s=s, kernel=self, context="when calling kernel directly"
                )

        s = xp.asarray(s)
        return _kernel_forward(self, s)

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

    def is_in_domain(self, s: npt.ArrayLike, xp: Any | None = None) -> bool:
        xp = xp or _get_namespace(s)
        s = xp.asarray(s)
        return self.transform.is_in_domain(s - self.order, xp=xp)

    def forward(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
        xp = xp or _get_namespace(s)
        s = xp.asarray(s)
        sign = 1 - 2 * (self.order % 2)
        shifted = s.reshape(s.shape + (1,)) - xp.arange(1, self.order + 1)
        return (
            sign
            * xp.prod(shifted, axis=-1)
            * _kernel_forward(self.transform, s - self.order)
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
    Derivative : Compute derivatives of kernels
    """

    def __init__(self, mu: npt.ArrayLike, check_bounds: bool = True) -> None:
        super().__init__(check_bounds=check_bounds)
        self.mu = np.asarray(mu)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Domain of convergence: (-mu, 1.5)."""
        return (-self.mu, 1.5 * np.ones_like(self.mu))

    def forward(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
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
        xp = xp or _get_namespace(s)
        # Reshape mu and s to enable proper broadcasting
        mu, s = _safe_broadcast_namespace(self.mu, s, xp=xp)
        try:
            mu = np.asarray(mu)
            s = np.asarray(s)
        except Exception as err:
            raise TypeError(
                "Kernel namespace path is not supported by BesselJKernel's "
                "SciPy-based implementation. Provide an Array-API-compliant "
                "kernel implementation for this namespace."
            ) from err
        logforward = (
            LOG_2 * (s - 1)
            + special.loggamma(0.5 * (mu + s))
            - special.loggamma(0.5 * (mu + 2 - s))
        )
        return xp.asarray(np.exp(logforward))


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

    def forward(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
        xp = xp or _get_namespace(s)
        s = xp.asarray(s)
        return super().forward(s - 0.5) * xp.asarray(SQRT_PI_OVER_2)


class CombinedKernel(Kernel):
    """
    Kernel that combines multiple kernels by stacking outputs along axis 0.

    This enables mixing different kernel types, orders, and derivatives in a
    single transform operation. The combined kernel behaves like a batched kernel,
    where the batch dimension corresponds to the list of kernels.

    Parameters
    ----------
    kernels : Sequence[Kernel]
        Sequence of kernel instances to combine.
    check_bounds : bool, optional
        If True, validate that input values are within the domain of all kernels.
        Default is True.

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel, CombinedKernel
    >>> k1 = BesselJKernel(mu=0)
    >>> k2 = BesselJKernel(mu=1)
    >>> kernel = CombinedKernel([k1, k2])
    >>> # Transform uses both kernels, returning shape (2, n)
    """

    def __init__(self, kernels: Sequence[Kernel], check_bounds: bool = True) -> None:
        super().__init__(check_bounds=check_bounds)
        self.kernels = kernels

    @staticmethod
    def _flatten_kernels(kernels: Sequence[Kernel]) -> list[Kernel]:
        # Flatten nested CombinedKernels for a unified interface
        flattened = []
        for k in kernels:
            if isinstance(k, CombinedKernel):
                flattened.extend(k.kernels)
            else:
                flattened.append(k)

        if len(flattened) == 0:
            raise ValueError("At least one kernel must be provided to CombinedKernel")
        return flattened

    @property
    def kernels(self) -> list[Kernel]:
        return self._kernels

    @kernels.setter
    def kernels(self, kernels: Sequence[Kernel]) -> None:
        self._kernels = self._flatten_kernels(kernels)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Domain of convergence for the combined kernel.

        Returns stacked domains of all sub-kernels.
        """
        domains = [k.domain for k in self.kernels]
        infs = np.broadcast_arrays(*[d[0] for d in domains])
        sups = np.broadcast_arrays(*[d[1] for d in domains])
        return np.stack(infs, axis=0), np.stack(sups, axis=0)

    def forward(self, s: npt.ArrayLike, xp: Any | None = None) -> npt.NDArray | Any:
        """
        Compute the Mellin transform for all kernels and stack results.

        Parameters
        ----------
        s : array_like
            Complex frequency variable.

        Returns
        -------
        ndarray
            Stacked Mellin transforms. Shape will be (N, ...) where N is
            the number of kernels. If s is scalar, returns shape (N, 1)
            for compatibility with FFTLog batch processing.
        """
        xp = xp or _get_namespace(s)
        s = xp.asarray(s)
        results = [_kernel_forward(k, s) for k in self.kernels]
        shape = np.broadcast_shapes(*(r.shape for r in results))
        results = [xp.broadcast_to(r, shape) for r in results]
        res = xp.stack(results, axis=0)

        # Scalar input needs shape (N, 1) for FFTLog broadcasting
        if xp.asarray(s).ndim == 0 and res.ndim == 1:
            res = res[..., np.newaxis]

        return res
