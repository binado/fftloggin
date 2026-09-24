"""Pure, one-dimensional FFTLog transforms."""

import warnings
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, ArrayLike, Complex, Float

from .exceptions import ArgumentOutOfDomainError, DomainCheckWarning
from .kernels import Kernel

__all__ = (
    "Plan",
    "forward",
    "inverse",
    "lowring_log_kr",
    "plan",
    "validate_parameters",
)


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


def _apply_coefficients(
    a: Float[Array, "n"],
    coeffs: Complex[Array, "m ..."],
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""],
    log_kr: Float[ArrayLike, ""],
) -> Float[Array, "n ..."]:
    """Apply forward coefficients, broadcasting over their trailing axes."""
    n = a.shape[0]
    trailing = (1,) * (coeffs.ndim - 1)
    before = _power_law(n, dlog, bias, -1)
    spectrum = jnp.fft.rfft(a * before).reshape((-1, *trailing))
    result = jnp.fft.irfft(spectrum * coeffs, n=n, axis=0)[::-1]
    before = before.reshape((n, *trailing))
    return result * before * jnp.exp(-jnp.asarray(bias) * log_kr)


def _invert_coefficients(
    A: Float[Array, "n"],
    coeffs: Complex[Array, "m"],
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""],
    log_kr: Float[ArrayLike, ""],
) -> Float[Array, "n"]:
    n = A.shape[0]
    power = _power_law(n, dlog, bias, 1)
    spectrum = jnp.fft.rfft(A * power * jnp.exp(bias * log_kr))
    result = jnp.fft.irfft(spectrum / jnp.conj(coeffs), n=n)[::-1]
    return result * power


@partial(
    register_dataclass,
    data_fields=("coeffs", "dlog", "bias", "log_kr"),
    meta_fields=("n",),
)
@dataclass(frozen=True)
class Plan:
    """FFTLog coefficients for a fixed grid, reusable across input arrays.

    Build one with ``plan`` or with the constructors in
    ``fftloggin.cosmology`` and pass it to ``forward`` or ``inverse`` in
    place of a kernel. A plan is a pytree: ``coeffs``, ``dlog``, ``bias`` and
    ``log_kr`` are array leaves, while the sample count ``n`` is static.
    Plans built under ``jax.vmap`` carry a leading batch axis on every leaf;
    map ``forward`` over them with ``in_axes=(None, 0)``.

    Attributes
    ----------
    coeffs : array
        Complex coefficients with shape ``(n // 2 + 1, ...)``. Trailing axes,
        such as the ratio axis of a kernel-product plan, are carried to the
        transform output.
    dlog : scalar
        Logarithmic grid spacing.
    bias : scalar
        Power-law bias.
    log_kr : scalar
        Logarithm of the product of the geometric centers of the paired grids.
    n : int
        Number of samples of the arrays the plan transforms.
    """

    # Leading "*batch" axes appear on plans returned from ``jax.vmap``.
    coeffs: Complex[Array, "..."]
    dlog: Float[Array, "*batch"]
    bias: Float[Array, "*batch"]
    log_kr: Float[Array, "*batch"]
    n: int


def plan(
    kernel: Kernel,
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> Plan:
    """Evaluate the FFTLog coefficients of ``kernel`` once for reuse.

    Parameters
    ----------
    kernel : Kernel
        Mellin kernel used to calculate the transform coefficients.
    n : int
        Number of samples of the arrays to transform, at least two.
    dlog : scalar
        Positive spacing between adjacent input coordinates in log space.
    bias : scalar, optional
        Power-law bias used to control the periodic continuation. Defaults to
        zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.

    Returns
    -------
    Plan
        Coefficients with shape ``(n // 2 + 1,)`` and their grid parameters.

    Notes
    -----
    ``forward(a, plan(kernel, n, dlog=dlog))`` equals
    ``forward(a, kernel, dlog=dlog)`` but skips the Gamma-function
    evaluations, so a plan pays off when many arrays share one kernel and
    grid, for example with ``jax.vmap(forward, in_axes=(0, None))``.

    Gradients do not reach whatever built a precomputed plan. Precompute it
    when differentiating with respect to the input samples; build it inside
    the differentiated function when differentiating with respect to kernel
    or grid parameters.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    return Plan(
        coeffs=_coefficients(kernel, n, dlog, bias, log_kr),
        dlog=jnp.asarray(dlog),
        bias=jnp.asarray(bias),
        log_kr=jnp.asarray(log_kr),
        n=n,
    )


def _resolve(
    kernel: Kernel | Plan,
    n: int,
    dlog: Float[ArrayLike, ""] | None,
    bias: Float[ArrayLike, ""] | None,
    log_kr: Float[ArrayLike, ""] | None,
) -> Plan:
    if isinstance(kernel, Plan):
        if dlog is not None or bias is not None or log_kr is not None:
            raise TypeError(
                "a Plan carries its own grid parameters; "
                "do not pass dlog, bias or log_kr"
            )
        if kernel.n != n:
            raise ValueError(f"plan was built for n={kernel.n}, but got {n} samples")
        return kernel
    if dlog is None:
        raise TypeError("dlog is required when transforming with a Kernel")
    return plan(
        kernel,
        n,
        dlog=dlog,
        bias=0.0 if bias is None else bias,
        log_kr=0.0 if log_kr is None else log_kr,
    )


def forward(
    a: Float[ArrayLike, "n"],
    kernel: Kernel | Plan,
    *,
    dlog: Float[ArrayLike, ""] | None = None,
    bias: Float[ArrayLike, ""] | None = None,
    log_kr: Float[ArrayLike, ""] | None = None,
) -> Float[Array, "n ..."]:
    """Transform one real sample array between logarithmically spaced grids.

    The forward Bessel transform convention is ``k * integral(a(r) *
    J_mu(k*r), r)``. The sample count comes from ``a.shape[0]``.

    Parameters
    ----------
    a : array_like
        One-dimensional real floating-point samples, with at least two values.
    kernel : Kernel or Plan
        Mellin kernel used to calculate the transform coefficients, or a
        ``Plan`` holding precomputed coefficients. A plan carries its own
        grid parameters, so ``dlog``, ``bias`` and ``log_kr`` must then be
        omitted.
    dlog : scalar
        Positive spacing between adjacent input coordinates in log space.
        Required with a kernel.
    bias : scalar, optional
        Power-law bias used to control the periodic continuation. Defaults to
        zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.

    Returns
    -------
    array
        Transformed real samples with shape ``(n, *plan.coeffs.shape[1:])``,
        that is the shape of ``a`` for a kernel or single-kernel plan.

    Raises
    ------
    TypeError
        If ``dlog`` is missing with a kernel, or grid parameters are passed
        with a plan.
    ValueError
        If a plan was built for a different number of samples.

    Notes
    -----
    Map over scalar parameters with ``jax.vmap``. Concrete value and
    Mellin-domain checks are available through ``validate_parameters`` and
    should be performed outside JAX transformations. See ``plan`` for reusing
    coefficients across inputs.
    """
    a = _samples(a, "a")
    p = _resolve(kernel, a.shape[0], dlog, bias, log_kr)
    return _apply_coefficients(a, p.coeffs, p.dlog, p.bias, p.log_kr)


def inverse(
    A: Float[ArrayLike, "n"],
    kernel: Kernel | Plan,
    *,
    dlog: Float[ArrayLike, ""] | None = None,
    bias: Float[ArrayLike, ""] | None = None,
    log_kr: Float[ArrayLike, ""] | None = None,
) -> Float[Array, "n"]:
    """Invert one real sample array from the output grid to the input grid.

    Parameters
    ----------
    A : array_like
        One-dimensional real floating-point samples, with at least two values.
    kernel : Kernel or Plan
        Mellin kernel used for the inverse transform, or a single-kernel
        ``Plan``. Grid parameters must be omitted with a plan.
    dlog : scalar
        Positive logarithmic spacing of the paired grids. Required with a
        kernel.
    bias : scalar, optional
        Power-law bias used by the forward transform. Defaults to zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.

    Returns
    -------
    array
        Inverse-transformed real samples, with the same length as ``A``.

    Raises
    ------
    TypeError
        If ``dlog`` is missing with a kernel, or grid parameters are passed
        with a plan.
    ValueError
        If a plan was built for a different number of samples or holds more
        than one column of coefficients.

    Notes
    -----
    Use the same kernel, ``dlog``, ``bias`` and ``log_kr`` values, or the
    same plan, as the corresponding forward transform.
    """
    A = _samples(A, "A")
    p = _resolve(kernel, A.shape[0], dlog, bias, log_kr)
    if p.coeffs.ndim != 1:
        raise ValueError("inverse supports only single-kernel plans")
    return _invert_coefficients(A, p.coeffs, p.dlog, p.bias, p.log_kr)


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

    Parameters
    ----------
    kernel : Kernel
        Mellin kernel used to calculate the low-ringing center.
    dlog : scalar
        Positive logarithmic grid spacing.
    bias : scalar, optional
        Power-law bias used by the transform. Defaults to zero.
    log_kr : scalar, optional
        Requested log center. The result is the nearest low-ringing value to
        this request. Defaults to zero.

    Returns
    -------
    scalar
        Snapped log center, suitable for both the transform and paired-grid
        construction.
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

    Parameters
    ----------
    kernel : Kernel
        Mellin kernel whose convergence strip is checked.
    dlog : scalar
        Must be a finite positive scalar.
    bias : scalar, optional
        Must be finite. A value outside the open Mellin strip emits
        ``DomainCheckWarning``. Defaults to zero.
    log_kr : scalar, optional
        Must be finite. Defaults to zero.

    Raises
    ------
    ValueError
        If any parameter is not a finite scalar or if ``dlog`` is not
        positive.

    Notes
    -----
    This is an eager host-side helper. Do not call it inside a JAX transform.
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
