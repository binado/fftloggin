"""
Mellin transform kernels for FFTLog algorithm.

This module provides kernel functions that compute the Mellin transform
of various integral kernels used in generalized FFTLog transforms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Literal, Self, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy import special

from .utils import safe_broadcast

__all__ = (
    "Kernel",
    "BesselJKernel",
    "ShiftedKernel",
    "Derivative",
    "CombinedKernel",
)


LOG_2 = np.log(2)
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2)


class Kernel:
    """
    Base class for Mellin transform kernels.

    A kernel represents the Mellin transform of an integral kernel function.
    Kernels have a domain (strip of convergence) in the complex plane where
    the transform is well-defined.

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

    def __init__(self) -> None:
        pass

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

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
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

    @overload
    def derive(self, order: Literal[0]) -> Self: ...

    @overload
    def derive(self, order: int = 1) -> Derivative[Self]: ...

    def derive(self, order: int = 1) -> Self | Derivative[Self]:
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
        Self | Derivative[Self]
            If ``order`` is 0, returns ``self`` unchanged.
            Otherwise returns a :class:`Derivative` wrapper around ``self``.

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

    @overload
    def shift(self, nu: Literal[0] = 0) -> Self: ...

    @overload
    def shift(self, nu: float) -> ShiftedKernel[Self]: ...

    def shift(self, nu: float = 0.0) -> Self | ShiftedKernel[Self]:
        """
        Return a kernel with Mellin argument shift ``s -> s + nu``.

        Parameters
        ----------
        nu : float, optional
            Additive shift in Mellin-space argument. Default is 0.0.

        Returns
        -------
        Self | ShiftedKernel[Self]
            If ``nu`` is 0, returns ``self`` unchanged.
            Otherwise returns a :class:`ShiftedKernel` wrapper.
        """
        if nu == 0:
            return self
        return ShiftedKernel(self, nu)


K = TypeVar("K", bound=Kernel)


class ShiftedKernel(Kernel, Generic[K]):
    """
    Kernel wrapper that applies an additive Mellin-space argument shift.

    Semantics:
        shifted(s) = base(s + nu)
    """

    def __init__(self, base: K, nu: float) -> None:
        super().__init__()
        self.base = base
        self.nu = float(nu)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        inf, sup = self.base.domain
        return np.asarray(inf) - self.nu, np.asarray(sup) - self.nu

    def is_in_domain(self, s: npt.ArrayLike) -> bool:
        s = np.asarray(s)
        return self.base.is_in_domain(s + self.nu)

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
        return self.base(np.asarray(s) + self.nu)


class Derivative(Kernel, Generic[K]):
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
    >>> result = d_kernel(2.0)

    Notes
    -----
    The derivative kernel inherits the domain of convergence from the base
    kernel, adjusted for the derivative order.

    See Also
    --------
    Kernel.derive : Recommended way to compute derivatives
    """

    def __init__(self, transform: K, order: int) -> None:
        super().__init__()
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

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
        s = np.asarray(s)
        sign = 1 - 2 * (self.order % 2)
        return (
            sign
            * np.prod(s.reshape(*s.shape, 1) - np.arange(1, self.order + 1), axis=-1)
            * self.transform(s - self.order)
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

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel
    >>> import numpy as np
    >>> # Single order
    >>> kernel = BesselJKernel(mu=0.5)
    >>> # Multiple orders (for vectorized transforms)
    >>> kernels = BesselJKernel(mu=np.array([0, 0.5, 1.0]))
    >>> # Compute Mellin transform at s = 1.0
    >>> result = kernel(1.0)

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

    def __init__(self, mu: npt.ArrayLike) -> None:
        super().__init__()
        self.mu = np.asarray(mu)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Domain of convergence: (-mu, 1.5)."""
        return (-self.mu, 1.5 * np.ones_like(self.mu))

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
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

    def __init__(self, ell: npt.ArrayLike) -> None:
        mu = np.asarray(ell) + 0.5
        super().__init__(mu)

    @property
    def domain(self) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        inf, sup = super().domain
        return (np.asarray(inf) + 0.5, np.asarray(sup) + 0.5)

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
        s = np.asarray(s)
        return super().__call__(s - 0.5) * SQRT_PI_OVER_2


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

    Examples
    --------
    >>> from fftloggin.kernels import BesselJKernel, CombinedKernel
    >>> k1 = BesselJKernel(mu=0)
    >>> k2 = BesselJKernel(mu=1)
    >>> kernel = CombinedKernel([k1, k2])
    >>> # Transform uses both kernels, returning shape (2, n)
    """

    def __init__(self, kernels: Sequence[Kernel]) -> None:
        super().__init__()
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

    def __call__(self, s: npt.ArrayLike) -> npt.NDArray:
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
        results = np.broadcast_arrays(*[k(s) for k in self.kernels])
        res = np.stack(results, axis=0)

        # Scalar input needs shape (N, 1) for FFTLog broadcasting
        if np.ndim(s) == 0 and res.ndim == 1:
            res = res[..., np.newaxis]

        return res
