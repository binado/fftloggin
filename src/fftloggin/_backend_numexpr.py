import warnings

import numpy as np
import numpy.typing as npt
from scipy import special
from scipy.fft import irfft, rfft

try:
    import numexpr as ne

    NUMEXPR_AVAILABLE = True
except ImportError:
    NUMEXPR_AVAILABLE = False

LN_2 = np.log(2)


# ============================================================================
# Numexpr-optimized implementation
# ============================================================================


def fhtoffset_numexpr(
    dln: float,
    mu: npt.ArrayLike,
    initial: npt.ArrayLike = 0.0,
    bias: float = 0.0,
) -> np.ndarray:
    """
    Return optimal offset for the fast Hankel transform, using numexpr.

    Vectorized version of scipy.fft.fhtoffset that supports broadcasting
    of mu, offset, and bias parameters.
    """
    mu = np.asarray(mu)
    lnkr = np.asarray(initial)
    q = bias
    xp = ne.evaluate("(mu + 1 + q) / 2", local_dict={"mu": mu, "q": q})
    xm = ne.evaluate("(mu + 1 - q) / 2", local_dict={"mu": mu, "q": q})
    y = np.pi / (2 * dln)
    zp = special.loggamma(xp + 1j * y)
    zm = special.loggamma(xm + 1j * y)
    zp_imag = zp.imag
    zm_imag = zm.imag
    arg = ne.evaluate(
        "(LN_2 - lnkr) / dln + (zp_imag + zm_imag) / np.pi",
        local_dict=dict(
            LN_2=LN_2,
            lnkr=lnkr,
            dln=dln,
            zp_imag=zp_imag,
            zm_imag=zm_imag,
        ),
    )
    return ne.evaluate(
        "lnkr + (arg - round(arg)) * dln", local_dict=dict(lnkr=lnkr, arg=arg, dln=dln)
    )


def _fhtcoeff_numexpr(
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
    Compute the coefficient array for the fast Mellin transform using numexpr.

    Parameters
    ----------
    kernel : callable, optional
        Function with signature kernel(mu, y, q) -> complex
        Returns Mellin transform at frequency y.
    log_kernel : callable, optional
        Function with signature log_kernel(mu, y, q) -> complex
        Returns LOG of Mellin transform (more numerically stable).

    Notes
    -----
    numexpr optimization is primarily applied to the bias/offset operations.
    Custom kernels are computed using numpy/scipy (not optimized by numexpr).
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

    # Frequency array
    xj = np.arange(0, n // 2 + 1)
    m = 2 * np.pi / (n * dln)
    y = 0.5j * xj * m

    # Compute kernel-specific Mellin transform
    # Note: Custom kernels cannot be optimized by numexpr
    if log_kernel is not None:
        log_kernel_vals = log_kernel(mu, y, q)
    else:
        kernel_vals = kernel(mu, y, q)
        log_kernel_vals = np.log(kernel_vals)

    # Apply offset and bias using numexpr
    log_u = ne.evaluate(
        "log_kernel_vals + q * LN_2 + 1j * xj * m * (LN_2 - offset)",
        local_dict=dict(
            log_kernel_vals=log_kernel_vals,
            q=q,
            LN_2=LN_2,
            xj=xj,
            m=m,
            offset=offset,
        ),
    )
    u = ne.evaluate("exp(log_u)", local_dict=dict(log_u=log_u))

    # Handle Nyquist frequency for even n
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
            u = np.copy(u)
            u[singular, 0] = 0
    else:
        singular = u[..., 0] == 0
        if singular.any():
            warnings.warn(
                "singular inverse transform; consider changing the bias", stacklevel=3
            )
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
    a_tilde = ne.evaluate("a_tilde * u", local_dict=dict(a_tilde=a_tilde, u=u))
    ak = irfft(a_tilde, n, axis=axis)
    return np.flip(ak, axis=axis)


def _ifhtq(a: npt.NDArray, u: npt.NDArray, axis: int = -1) -> npt.NDArray:
    """
    Computes the inverse discrete Hankel transform of a real array a
    with pre-computed array of coefficients u.
    """
    n = a.shape[axis]
    a_tilde = rfft(a, axis=axis)
    u_conj = np.conjugate(u)
    a_tilde = ne.evaluate(
        "a_tilde / u_conj", local_dict=dict(a_tilde=a_tilde, u_conj=u_conj)
    )
    ar = irfft(a_tilde, n, axis=axis)
    return np.flip(ar, axis=axis)


def _fht_numexpr(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    """
    Compute the fast Mellin transform using numexpr optimization.
    """
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    ne.evaluate(
        "a * exp(-bias * (j - jc) * dln)",
        out=a,
        local_dict={
            "a": a,
            "bias": bias,
            "j": j,
            "jc": jc,
            "dln": dln,
        },
    )
    u = _fhtcoeff_numexpr(
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
    ne.evaluate(
        "aq_tilde * exp(-bias * ((j - jc) * dln + offset))",
        out=aq_tilde,
        local_dict={
            "aq_tilde": aq_tilde,
            "bias": bias,
            "j": j,
            "jc": jc,
            "dln": dln,
            "offset": offset,
        },
    )

    aq_tilde = np.moveaxis(aq_tilde, -1, axis)
    return aq_tilde


def _ifht_numexpr(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    """
    Compute the inverse fast Mellin transform using numexpr optimization.
    """
    a = np.asarray(a).copy()
    a = np.moveaxis(a, axis, -1)

    offset = np.atleast_1d(offset)
    n = a.shape[-1]
    jc = (n - 1) / 2
    j = np.arange(n).astype(a.dtype)

    ne.evaluate(
        "a * exp(bias * ((j - jc) * dln + offset))",
        out=a,
        local_dict=dict(
            a=a,
            bias=bias,
            j=j,
            jc=jc,
            dln=dln,
            offset=offset,
        ),
    )
    u = _fhtcoeff_numexpr(
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
    ne.evaluate(
        "aq_tilde * exp(bias * (j - jc) * dln)",
        out=aq_tilde,
        local_dict=dict(
            aq_tilde=aq_tilde,
            bias=bias,
            j=j,
            jc=jc,
            dln=dln,
        ),
    )

    aq_tilde = np.moveaxis(aq_tilde, -1, axis)
    return aq_tilde
