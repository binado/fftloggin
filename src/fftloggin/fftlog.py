"""Pure, one-dimensional FFTLog transforms."""

import warnings

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float

from .exceptions import ArgumentOutOfDomainError, DomainCheckWarning
from .kernels import Kernel

__all__ = ("forward", "inverse", "lowring_log_kr", "validate_parameters")


def _samples(x: Float[ArrayLike, "n"], name: str) -> Float[Array, "n"]:
    x = jnp.asarray(x)
    if x.ndim != 1 or x.shape[0] < 2:
        raise ValueError(
            f"{name} must be a one-dimensional array with at least two samples"
        )
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"{name} must contain real floating-point samples")
    return x


def _coefficients(
    kernel: Kernel,
    n: int,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""],
    log_kr: Float[ArrayLike, ""],
) -> Complex[Array, "m"]:
    m = jnp.arange(n // 2 + 1)
    angle = 2j * jnp.pi * m / (n * dlog)
    coeffs = kernel(1 + bias + angle) * jnp.exp(-angle * log_kr)
    if n % 2 == 0:
        coeffs = coeffs.at[-1].set(jnp.real(coeffs[-1]))
    return coeffs


def _power_law(
    n: int,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""],
    sign: int,
) -> Float[Array, "m"]:
    return jnp.exp(sign * bias * dlog * (jnp.arange(n) - (n - 1) / 2))


def forward(
    a: Float[ArrayLike, "n"],
    kernel: Kernel,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> Float[Array, "n"]:
    """Transform one real sample array between logarithmically spaced grids.

    Parameters other than ``a`` are scalars; map over them with ``jax.vmap``.
    The sample count comes from ``a.shape[0]``.
    """
    a = _samples(a, "a")
    n = a.shape[0]
    before = _power_law(n, dlog, bias, -1)
    coeffs = _coefficients(kernel, n, dlog, bias, log_kr)
    result = jnp.fft.irfft(jnp.fft.rfft(a * before) * coeffs, n=n)[::-1]
    return result * before * jnp.exp(-jnp.asarray(bias) * log_kr)


def inverse(
    A: Float[ArrayLike, "n"],
    kernel: Kernel,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> Float[Array, "n"]:
    """Invert one real sample array from the output grid to the input grid."""
    A = _samples(A, "A")
    n = A.shape[0]
    power = _power_law(n, dlog, bias, 1)
    coeffs = _coefficients(kernel, n, dlog, bias, log_kr)
    result = jnp.fft.irfft(
        jnp.fft.rfft(A * power * jnp.exp(bias * log_kr)) / jnp.conj(coeffs), n=n
    )[::-1]
    return result * power


def lowring_log_kr(
    kernel: Kernel,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> Float[Array, ""]:
    """Snap ``log_kr`` to the nearest low-ringing value.

    The snap is piecewise constant in ``log_kr``. Apply it explicitly before
    ``forward`` or ``inverse`` when low ringing is wanted.
    """
    optimal = dlog * jnp.angle(kernel(1 + bias + 1j * jnp.pi / dlog)) / jnp.pi
    return optimal + jnp.round((log_kr - optimal) / dlog) * dlog


def validate_parameters(
    kernel: Kernel,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> None:
    """Validate concrete scalar parameters on the host before tracing.

    Raises for invalid spacing or center, and warns for a bias outside the
    kernel's Mellin strip. Do not call this function inside a JAX transform.
    """
    for name, value in (("dlog", dlog), ("bias", bias), ("log_kr", log_kr)):
        arr = jnp.asarray(value)
        if arr.ndim != 0 or not bool(jnp.isfinite(arr)):
            raise ValueError(f"{name} must be a finite scalar")
    if not bool(jnp.asarray(dlog) > 0):
        raise ValueError("dlog must be positive")
    if not bool(kernel.is_in_domain(1 + bias)):
        warnings.warn(
            str(ArgumentOutOfDomainError(1 + bias, kernel, "FFTLog bias parameter")),
            DomainCheckWarning,
            stacklevel=2,
        )
