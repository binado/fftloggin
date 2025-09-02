import warnings

import numpy as np
import numpy.typing as npt
from scipy import special

from ._backend import _fht, _ifht
from ._backend_numexpr import NUMEXPR_AVAILABLE, _fht_numexpr, _ifht_numexpr

LN_2 = np.log(2)


def fhtoffset(
    dln: float,
    mu: npt.ArrayLike,
    initial: npt.ArrayLike = 0.0,
    bias: float = 0.0,
) -> np.ndarray:
    """
    Return optimal offset for the fast Hankel transform.

    Vectorized version of scipy.fft.fhtoffset that supports broadcasting
    of mu, offset, and bias parameters.

    Parameters
    ----------
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order of the Hankel transform, any positive or negative real number.
    initial : array_like, optional
        Original offset of the uniform logarithmic spacing.
        Default is 0.0.
    bias : array_like, optional
        Exponent of power law bias. Default is 0.0.

    Returns
    -------
    offset_opt : ndarray
        Optimal offset of the uniform logarithmic spacing of the output array.

    Notes
    -----
    This function computes an optimal value for the offset parameter that
    minimizes ringing in the transform by making the transform kernel periodic.
    This implements the "low-ringing" condition from equations (28)-(30) of [1]_.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    Examples
    --------
    >>> import numpy as np
    >>> from fftlog_vectorized import fhtoffset, fht
    >>> dln = 0.1
    >>> mu = 0.5
    >>> offset_opt = fhtoffset(dln, mu)
    >>> # Use optimal offset for transform
    >>> a = np.random.randn(128)
    >>> A = fht(a, dln, mu, offset=offset_opt)
    """
    mu = np.asarray(mu)
    lnkr = np.asarray(initial)
    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.pi / (2 * dln)
    zp = special.loggamma(xp + 1j * y)
    zm = special.loggamma(xm + 1j * y)
    arg = (LN_2 - lnkr) / dln + (zp.imag + zm.imag) / np.pi
    return lnkr + (arg - np.round(arg)) * dln


def fht(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    use_numexpr: bool = False,
) -> np.ndarray:
    """
    Compute the fast Hankel transform.

    Vectorized version of scipy.fft.fht that supports broadcasting
    of mu, offset, and bias parameters with optional numexpr optimization.

    Parameters
    ----------
    a : array_like
        Real input array to be transformed.
        Shape: (..., n) where n is the length along the transform axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order of the Hankel transform, any positive or negative real number.
        Can be scalar or array that broadcasts with offset and bias.
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Can be scalar or array that broadcasts with mu and bias.
        Default is 0.0, which is equivalent to offset = log(kr) for kr = 1.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.
        Can be scalar or array that broadcasts with mu and offset.
        Default is 0.0, which gives an unbiased transform.
    use_numexpr : bool
        Whether to use numexpr optimization if available.
        - True: Use numexpr if available
        - False: Use standard numpy implementation

    Returns
    -------
    A : ndarray
        The transformed output array. Shape depends on broadcasting of
        mu, offset, and bias with the input array.

    Notes
    -----
    This function computes a discrete version of the Hankel transform

        A(k) = \int_{0}^{\infty} a(r) J_{\mu}(kr) (kr)^{q} k dr

    where J_{\mu} is the Bessel function of order \mu. The index q
    is related to the bias parameter by q = bias.

    The algorithm is based on the FFTLog method presented in [1]_.

    Numexpr optimization provides significant speedup for large arrays
    (typically 2-5x for n > 10000) but may have slight overhead for
    small arrays (n < 1000).

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    Examples
    --------
    >>> import numpy as np
    >>> from fftlog_vectorized import fht
    >>> # Logarithmically spaced input
    >>> r = np.logspace(-3, 3, 128)
    >>> dln = np.log(r[1]/r[0])
    >>> a = np.exp(-(r/1.0)**2)  # Gaussian
    >>> # Single transform
    >>> A = fht(a, dln, mu=0)
    >>> # Multiple transforms with different orders
    >>> mu_vals = np.array([0, 0.5, 1.0])[:, np.newaxis]
    >>> A_multi = fht(a, dln, mu=mu_vals)  # Shape: (3, 128)
    >>> # Force numexpr usage for large arrays
    >>> A_fast = fht(a, dln, mu=0, use_numexpr=True)
    """
    # Use appropriate implementation
    if use_numexpr and NUMEXPR_AVAILABLE:
        return _fht_numexpr(a, dln, mu, offset, bias, axis=axis)
    elif use_numexpr and not NUMEXPR_AVAILABLE:
        warnings.warn(
            "numexpr requested but not available, using standard implementation",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fht(a, dln, mu, offset=offset, bias=bias, axis=axis)
    else:
        return _fht(a, dln, mu, offset=offset, bias=bias, axis=axis)


def ifht(
    A: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    use_numexpr: bool = False,
) -> np.ndarray:
    """
    Compute the inverse fast Hankel transform.

    Vectorized version of scipy.fft.ifht that supports broadcasting
    of mu, offset, and bias parameters with optional numexpr optimization.

    Parameters
    ----------
    A : array_like
        Real input array to be transformed back.
        Shape: (..., n) where n is the length along the transform axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order of the Hankel transform, any positive or negative real number.
        Can be scalar or array that broadcasts with offset and bias.
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Can be scalar or array that broadcasts with mu and bias.
        Default is 0.0.
    bias : array_like, optional
        Exponent of power law bias, any positive or negative real number.
        Can be scalar or array that broadcasts with mu and offset.
        Default is 0.0.
    use_numexpr : bool
        Whether to use numexpr optimization if available.
        - True: Use numexpr if available
        - False: Use standard numpy implementation

    Returns
    -------
    a : ndarray
        The inverse transformed output array.

    Notes
    -----
    This function computes the inverse of the transformation computed by `fht`.

    Examples
    --------
    >>> import numpy as np
    >>> from fftlog_vectorized import fht, ifht
    >>> r = np.logspace(-3, 3, 128)
    >>> dln = np.log(r[1]/r[0])
    >>> a = np.exp(-(r/1.0)**2)
    >>> A = fht(a, dln, mu=0)
    >>> a_reconstructed = ifht(A, dln, mu=0)
    >>> np.allclose(a, a_reconstructed)
    True
    """
    # Use appropriate implementation
    if use_numexpr and NUMEXPR_AVAILABLE:
        return _ifht_numexpr(A, dln, mu, offset, bias, axis=axis)
    elif use_numexpr and not NUMEXPR_AVAILABLE:
        warnings.warn(
            "numexpr requested but not available, using standard implementation",
            RuntimeWarning,
            stacklevel=2,
        )
        return _ifht(A, dln, mu, offset=offset, bias=bias, axis=axis)
    else:
        return _ifht(A, dln, mu, offset=offset, bias=bias, axis=axis)
