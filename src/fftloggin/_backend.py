"""
Vectorized FFTLog implementation for computing discrete Hankel transforms.

This module implements the FFTLog algorithm as described in Hamilton (2000),
with vectorization support for mu (order), offset, and bias parameters.
The API mirrors scipy.fft's implementation with optional numexpr optimization.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy.fft import irfft, rfft

LN_2 = np.log(2)


# ============================================================================
# Standard implementation (numpy-based)
# ============================================================================
def _fhtcoeff(
    n: int,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    inverse: bool = False,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    """
    Compute the coefficient array for the fast Mellin transform.

    This is the generalized version that accepts custom kernel functions.

    Parameters
    ----------
    n : int
        Number of points in the array
    dln : float
        Logarithmic spacing of the input array, dln = ln(r[1]/r[0])
    mu : array_like
        Order parameter for the kernel (interpretation depends on kernel)
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Default is 0.0.
    bias : float, optional
        Exponent of the power law bias. Default is 0.0.
    inverse: bool
        Whether coefficients are calculated for inverse transformation
    kernel : callable, optional
        Function with signature kernel(mu, y, q) -> complex
        Returns Mellin transform at frequency y.
        Either kernel or log_kernel must be provided (not both).
    log_kernel : callable, optional
        Function with signature log_kernel(mu, y, q) -> complex
        Returns LOG of Mellin transform (more numerically stable).
        Either kernel or log_kernel must be provided (not both).

    Returns
    -------
    u : ndarray
        The coefficient array with shape (..., n//2+1) where ... represents
        the broadcasted shape of mu, offset, and bias.

    Raises
    ------
    ValueError
        If neither kernel nor log_kernel is provided, or if both are provided.

    Notes
    -----
    The kernel callable should compute the Mellin transform of the desired
    integral kernel. For the standard Bessel kernel J_μ, use:
    `from fftloggin.kernels import bessel_mellin_log_kernel`

    The log_kernel version is preferred for numerical stability as it avoids
    overflow/underflow in intermediate calculations.
    """
    # Validate kernel parameters
    if kernel is None and log_kernel is None:
        raise ValueError(
            "Either 'kernel' or 'log_kernel' must be provided. "
            "For standard Hankel transforms, use: "
            "from fftloggin.kernels import bessel_mellin_log_kernel"
        )
    if kernel is not None and log_kernel is not None:
        raise ValueError("Cannot specify both 'kernel' and 'log_kernel'")

    # Ensure inputs are arrays and can broadcast
    mu = np.asarray(mu)
    offset = np.asarray(offset)

    q = bias

    # Frequency array (kernel-independent)
    xj = np.arange(0, n // 2 + 1)
    m = 2 * np.pi / (n * dln)
    y = 0.5j * xj * m

    # Compute kernel-specific Mellin transform
    if log_kernel is not None:
        log_u = log_kernel(mu, y, q)
    else:
        # Compute log from kernel for numerical stability
        kernel_vals = kernel(mu, y, q)
        log_u = np.log(kernel_vals)

    # Apply offset and bias (kernel-independent)
    log_u = log_u + q * LN_2 + 1j * xj * m * (LN_2 - offset)
    u = np.exp(log_u)

    # Handle Nyquist frequency for even n (kernel-independent)
    if n % 2 == 0:
        u[..., -1] = np.real(u[..., -1])

    # Check for special cases
    mask = np.isfinite(u[..., 0])
    if not mask.all():
        warnings.warn(
            "Non-finite kernel coefficients at zero frequency. "
            "This may indicate a singular transform or require "
            "kernel-specific special case handling.",
            stacklevel=3,
        )

    # check for singular transform or singular inverse transform
    if not inverse:
        singular = np.isinf(u[..., 0])
        if singular.any():
            warnings.warn(
                "singular transform; consider changing the bias", stacklevel=3
            )
            # fix coefficient to obtain (potentially correct) transform anyway
            u = np.copy(u)
            u[singular, 0] = 0
    else:
        singular = u[..., 0] == 0
        if singular.any():
            warnings.warn(
                "singular inverse transform; consider changing the bias", stacklevel=3
            )
            # fix coefficient to obtain (potentially correct) inverse anyway
            u = np.copy(u)
            u[singular, 0] = np.inf

    return u


def _fhtq(a: npt.NDArray, u: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """
    Computes the discrete Hankel transform of a real array a
    with pre-computed array of coefficients u.
    """
    n = a.shape[axis]
    a_tilde = rfft(a, axis=axis)
    ak = irfft(a_tilde * u, n, axis=axis)
    return np.flip(ak, axis=axis)


def _ifhtq(a: npt.NDArray, u: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """
    Computes the inverse discrete Hankel transform of a real array a
    with pre-computed array of coefficients u.
    """
    n = a.shape[axis]
    a_tilde = rfft(a, axis=axis)
    # Eq. 25
    ar = irfft(a_tilde / np.conjugate(u), n, axis=axis)
    return np.flip(ar, axis=axis)


def _fht(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    a *= np.exp(-bias * (j - jc) * dln)
    u = _fhtcoeff(
        n,
        dln,
        mu,
        offset=offset,
        bias=bias,
        inverse=False,
        kernel=kernel,
        log_kernel=log_kernel,
    )
    aq_tilde = _fhtq(a, u, axis=-1)
    aq_tilde *= np.exp(-bias * ((j - jc) * dln + offset))

    aq_tilde = np.moveaxis(aq_tilde, -1, axis)
    return aq_tilde


def _ifht(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    a *= np.exp(bias * ((j - jc) * dln + offset))
    u = _fhtcoeff(
        n,
        dln,
        mu,
        offset=offset,
        bias=bias,
        inverse=True,
        kernel=kernel,
        log_kernel=log_kernel,
    )
    aq_tilde = _ifhtq(a, u, axis=-1)
    aq_tilde *= np.exp(bias * (j - jc) * dln)

    aq_tilde = np.moveaxis(aq_tilde, -1, axis)
    return aq_tilde
