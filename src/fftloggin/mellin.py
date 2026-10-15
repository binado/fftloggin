"""Discrete Mellin transforms with positive-exponent analysis.

Samples are specified relative to their first coordinate: ``t_j / t_0 =
exp(j * dlog)``. Outputs are discrete coefficients, without quadrature or
absolute-origin factors. All transforms support ``jit``, ``vmap`` and ``grad``;
``n``, ``axis`` and ``norm`` must be static under JAX transformations.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Real

__all__ = (
    "imellin",
    "irmellin",
    "mellin",
    "mellinfreq",
    "rmellin",
    "rmellinfreq",
)


def _opposite_norm(norm: str | None) -> str:
    if norm is None or norm == "backward":
        return "forward"
    if norm == "forward":
        return "backward"
    if norm == "ortho":
        return "ortho"
    raise ValueError(f"invalid norm: {norm!r}")


def _weight(
    a: jax.Array, n: int, dlog: Real[ArrayLike, ""], c: Real[ArrayLike, ""]
) -> jax.Array:
    dlog, c = jnp.asarray(dlog), jnp.asarray(c)
    if dlog.ndim != 0 or c.ndim != 0:
        raise ValueError("dlog and c must be scalars")
    if jnp.iscomplexobj(dlog) or jnp.iscomplexobj(c):
        raise TypeError("dlog and c must be real")
    # A weak floating scalar preserves weak float64 inputs when X64 is enabled.
    dtype = jnp.result_type(a.real, dlog, c, 0.0)
    return jnp.exp(c * dlog * jnp.arange(n, dtype=dtype))


def _prepare(a: ArrayLike, n: int | None, axis: int) -> tuple[jax.Array, int]:
    a = jnp.moveaxis(jnp.asarray(a), axis, -1)
    n = a.shape[-1] if n is None else n
    if n < 1:
        raise ValueError("transform length must be positive")
    return a, n


def _resize_samples(a: jax.Array, n: int) -> jax.Array:
    a = a[..., :n]
    return jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, max(0, n - a.shape[-1]))])


def mellin(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    dlog: Real[ArrayLike, ""] = 1.0,
    c: Real[ArrayLike, ""] = 0.0,
) -> jax.Array:
    """Return positive-exponent discrete Mellin coefficients.

    Parameters
    ----------
    a : array_like
        Real or complex samples. Arbitrary batch dimensions are supported.
    n : int, optional
        Transform length. Crop or zero-pad samples before weighting.
    axis : int, optional
        Sample axis, defaulting to the last axis.
    norm : {None, "backward", "forward", "ortho"}, optional
        None and "backward" leave analysis unscaled and scale synthesis by
        1/n. "forward" reverses these factors; "ortho" uses 1/sqrt(n).
    dlog : real scalar, optional
        Positive logarithmic sample spacing, defaulting to one.
    c : real scalar, optional
        Real part of the Mellin contour, defaulting to zero.

    Returns
    -------
    jax.Array
        Complex coefficients in FFT frequency order. With default
        normalization, ``C[m] = sum_j a[j] * exp(c*j*dlog + 2j*pi*j*m/n)``.
        They correspond to ``s = c + 1j * mellinfreq(n, dlog)``.

    Notes
    -----
    For absolute coordinates ``t_j = t_0 * exp(j*dlog)``, continuous Mellin
    values are approximated by ``dlog * t_0**s * C`` with default normalization.
    Accuracy depends on sampling and truncation. No convergence-strip checks
    are performed. This operation does not include the absolute grid origin.
    """
    a, n = _prepare(a, n, axis)
    a = _resize_samples(a, n)
    result = jnp.fft.ifft(a * _weight(a, n, dlog, c), norm=_opposite_norm(norm))
    return jnp.moveaxis(result, -1, axis)


def imellin(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    dlog: Real[ArrayLike, ""] = 1.0,
    c: Real[ArrayLike, ""] = 0.0,
) -> jax.Array:
    """Reconstruct complex samples from discrete Mellin coefficients.

    Arguments follow :func:`mellin`. Use the same contour, spacing and
    normalization as analysis. ``n`` crops or pads the spectrum as in an FFT.
    Default synthesis has negative exponent and a 1/n factor, then removes
    the power-law weight. The return value is always complex.
    """
    a, n = _prepare(a, n, axis)
    result = jnp.fft.fft(a, n=n, norm=_opposite_norm(norm))
    result = result / _weight(result, n, dlog, c)
    return jnp.moveaxis(result, -1, axis)


def rmellin(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    dlog: Real[ArrayLike, ""] = 1.0,
    c: Real[ArrayLike, ""] = 0.0,
) -> jax.Array:
    """Transform real samples, storing only nonnegative Mellin frequencies.

    Arguments and normalization follow :func:`mellin`. Real samples and real
    ``c`` give conjugate symmetry. The output has ``n // 2 + 1`` complex
    coefficients along ``axis``; coordinates come from :func:`rmellinfreq`.
    Complex input is rejected rather than discarding its imaginary part.
    """
    a, n = _prepare(a, n, axis)
    if jnp.iscomplexobj(a):
        raise TypeError("rmellin requires real samples")
    a = _resize_samples(a, n)
    result = jnp.conj(jnp.fft.rfft(a * _weight(a, n, dlog, c), norm=norm))
    return jnp.moveaxis(result, -1, axis)


def irmellin(
    a: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: str | None = None,
    *,
    dlog: Real[ArrayLike, ""] = 1.0,
    c: Real[ArrayLike, ""] = 0.0,
) -> jax.Array:
    """Reconstruct real samples from a Hermitian Mellin half spectrum.

    Arguments follow :func:`mellin`. When ``n`` is omitted, infer the even
    length ``2 * (a.shape[axis] - 1)``. Pass the original ``n`` explicitly for
    odd lengths. Spectrum resizing and real DC/Nyquist treatment follow
    ``irfft``. Use the same contour, spacing and normalization as analysis.
    """
    a, length = _prepare(a, n, axis)
    n = 2 * (length - 1) if n is None else n
    if n < 1:
        raise ValueError("transform length must be positive")
    result = jnp.fft.irfft(jnp.conj(a), n=n, norm=norm)
    result = result / _weight(result, n, dlog, c)
    return jnp.moveaxis(result, -1, axis)


def mellinfreq(n: int, dlog: Real[ArrayLike, ""] = 1.0) -> jax.Array:
    """Angular log frequencies in FFT order, with spacing 2*pi/(n*dlog).

    The Mellin coordinates are ``c + 1j * eta``. ``n`` is static under JIT;
    ``dlog`` is a positive real scalar and may be traced.
    """
    return 2 * jnp.pi * jnp.fft.fftfreq(n, d=dlog)


def rmellinfreq(n: int, dlog: Real[ArrayLike, ""] = 1.0) -> jax.Array:
    """Nonnegative angular log frequencies, including Nyquist for even n."""
    return 2 * jnp.pi * jnp.fft.rfftfreq(n, d=dlog)
