"""
Vectorized FFTLog implementation for computing discrete Hankel transforms.

This module implements the FFTLog algorithm as described in Hamilton (2000),
with vectorization support for mu (order), offset, and bias parameters.
The API mirrors scipy.fft's implementation with optional numexpr optimization.
"""

import warnings

import numpy as np
import numpy.typing as npt
from scipy import special
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
) -> np.ndarray:
    """
    Compute the coefficient array for the fast Hankel transform.

    This is the vectorized version of scipy's _fhtcoeff function.

    Parameters
    ----------
    n : int
        Number of points in the array
    dln : float
        Logarithmic spacing of the input array, dln = ln(r[1]/r[0])
    mu : array_like
        Order of the Hankel transform, mu=q+0.5 where q is the exponent of the
        power law bias (k*r)^q
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Default is 0.0.
    bias : float, optional
        Exponent of the power law bias. Default is 0.0.
    inverse: bool
        Whether coefficients are calculated for inverse transformation

    Returns
    -------
    u : ndarray
        The coefficient array with shape (..., n//2+1) where ... represents
        the broadcasted shape of mu, offset, and bias.
    """
    # Ensure inputs are arrays and can broadcast
    mu = np.asarray(mu)
    offset = np.asarray(offset)

    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2

    # Frequency array
    xj = np.arange(0, n // 2 + 1)
    m = 2 * np.pi / (n * dln)
    arg = xp + 0.5j * xj * m
    loggamma_res = np.empty_like(arg)
    u = np.zeros_like(arg)
    special.loggamma(arg, out=loggamma_res)
    u += loggamma_res
    arg = xm - 0.5j * xj * m
    special.loggamma(arg, out=loggamma_res)
    u -= loggamma_res
    u += q * LN_2
    u += 1j * xj * m * (LN_2 - offset)
    u = np.exp(u, out=u)

    # Handle Nyquist frequency for even n
    if n % 2 == 0:
        u[..., -1] = np.real(u[..., -1])

    # deal with special cases
    mask = np.isfinite(u[..., 0])
    if not mask.all():
        # write u_0 = 2^q Gamma(xp)/Gamma(xm) = 2^q poch(xm, xp-xm)
        # poch() handles special cases for negative integers correctly
        u[~mask, 0] = 2**q * special.poch(xm[~mask], q)
        # the coefficient may be inf or 0, meaning the transform or the
        # inverse transform, respectively, is singular

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
) -> np.ndarray:
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    a *= np.exp(-bias * (j - jc) * dln)
    u = _fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=False)
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
) -> np.ndarray:
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    a *= np.exp(bias * ((j - jc) * dln + offset))
    u = _fhtcoeff(n, dln, mu, offset=offset, bias=bias, inverse=True)
    aq_tilde = _ifhtq(a, u, axis=-1)
    aq_tilde *= np.exp(bias * (j - jc) * dln)

    aq_tilde = np.moveaxis(aq_tilde, -1, axis)
    return aq_tilde
