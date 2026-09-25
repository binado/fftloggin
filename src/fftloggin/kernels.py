"""Scalar Mellin kernels for JAX FFTLog transforms."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import loggamma as _jax_loggamma
from jax.tree_util import register_dataclass
from jaxtyping import Array, ArrayLike, Bool, Inexact, Real

__all__ = (
    "BesselJKernel",
    "Derivative",
    "Kernel",
    "PowerLaw",
    "Scale",
    "SphericalBesselJKernel",
    "Transform",
    "TransformedKernel",
)


def _complex_digamma(z: Inexact[Array, "..."]) -> Inexact[Array, "..."]:
    """Digamma via reflection and a fixed recurrence/asymptotic expansion."""
    reflected = jnp.real(z) < 0.5
    zp = jnp.where(reflected, 1 - z, z)
    shifted = zp + 30
    inv2 = 1 / (shifted * shifted)
    term = inv2
    correction = jnp.zeros_like(z)
    # B_(2k)/(2k), k=1..5; the series is asymptotic, not convergent.
    for c in (1 / 12, -1 / 120, 1 / 252, -1 / 240, 1 / 132):
        correction = correction + c * term
        term = term * inv2
    positive = jnp.log(shifted) - 0.5 / shifted - correction
    positive -= jnp.sum(1 / (zp[..., None] + jnp.arange(30)), axis=-1)
    return jnp.where(reflected, positive - jnp.pi / jnp.tan(jnp.pi * z), positive)


@jax.custom_jvp
def _loggamma(z: Inexact[Array, "..."]) -> Inexact[Array, "..."]:
    return _jax_loggamma(z)


@_loggamma.defjvp
def _loggamma_jvp(primals, tangents):
    (z,), (tangent,) = primals, tangents
    return _loggamma(z), _complex_digamma(z) * tangent


def _bessel_j_mellin(
    mu: Real[ArrayLike, ""], s: Inexact[ArrayLike, "..."]
) -> Inexact[Array, "..."]:
    s = jnp.asarray(s)
    log_value = (
        jnp.log(2.0) * (s - 1) + _loggamma((mu + s) / 2) - _loggamma((mu + 2 - s) / 2)
    )
    return jnp.exp(log_value)


class Kernel:
    """Base interface for scalar Mellin kernels.

    Custom kernels passed to ``jax.jit`` must also be registered as pytrees.
    """

    @property
    def domain(self) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        """Open interval of real Mellin arguments where the kernel converges."""
        return jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        """Evaluate the Mellin kernel at ``s``."""
        raise NotImplementedError

    def is_in_domain(self, s: Inexact[ArrayLike, "..."]) -> Bool[Array, ""]:
        """Return whether every real part in ``s`` lies inside ``domain``."""
        lower, upper = self.domain
        return jnp.all((jnp.real(s) > lower) & (jnp.real(s) < upper))

    def transform(self, op: "Transform", *ops: "Transform") -> "TransformedKernel":
        """Apply transforms to the kernel in pipeline order.

        ``k.transform(a, b)`` equals ``k.transform(a).transform(b)``: ``a``
        acts on the kernel first. For example,
        ``k.transform(PowerLaw(2), Derivative(1))`` is
        ``d/dx [x**2 K(x)]``, not ``x**2 K'(x)``.
        """
        kernel = TransformedKernel(self, op)
        for other in ops:
            kernel = TransformedKernel(kernel, other)
        return kernel


class Transform:
    """Base interface for linear operations on a kernel.

    A transform acts on the kernel's Mellin transform pointwise in ``s`` and
    maps its convergence strip. Apply it with ``Kernel.transform``. Custom
    transforms passed to ``jax.jit`` must also be registered as pytrees.
    """

    def domain(
        self, lower: Real[ArrayLike, "..."], upper: Real[ArrayLike, "..."]
    ) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        """Map the base kernel's strip to the transformed kernel's strip."""
        return jnp.asarray(lower), jnp.asarray(upper)

    def __call__(
        self, kernel: Kernel, s: Inexact[ArrayLike, "..."]
    ) -> Inexact[Array, "..."]:
        """Evaluate the Mellin transform of the transformed ``kernel`` at ``s``."""
        raise NotImplementedError


@partial(register_dataclass, data_fields=("nu",), meta_fields=())
@dataclass(frozen=True)
class PowerLaw(Transform):
    """Multiply the kernel by a power law, ``x**nu * K(x)``."""

    nu: Real[ArrayLike, ""]

    def domain(
        self, lower: Real[ArrayLike, "..."], upper: Real[ArrayLike, "..."]
    ) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        return jnp.asarray(lower) - self.nu, jnp.asarray(upper) - self.nu

    def __call__(
        self, kernel: Kernel, s: Inexact[ArrayLike, "..."]
    ) -> Inexact[Array, "..."]:
        return kernel(s + self.nu)


@partial(register_dataclass, data_fields=(), meta_fields=("order",))
@dataclass(frozen=True)
class Derivative(Transform):
    """Differentiate the kernel ``order`` times, ``d^n K / dx^n``."""

    order: int

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order must be a positive integer")

    def domain(
        self, lower: Real[ArrayLike, "..."], upper: Real[ArrayLike, "..."]
    ) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        return jnp.asarray(lower) + self.order, jnp.asarray(upper) + self.order

    def __call__(
        self, kernel: Kernel, s: Inexact[ArrayLike, "..."]
    ) -> Inexact[Array, "..."]:
        s = jnp.asarray(s)
        factor = jnp.prod(s[..., None] - jnp.arange(1, self.order + 1), axis=-1)
        return (-1) ** self.order * factor * kernel(s - self.order)


@partial(register_dataclass, data_fields=("factor",), meta_fields=())
@dataclass(frozen=True)
class Scale(Transform):
    """Rescale the kernel's argument, ``K(factor * x)``.

    ``factor`` must be positive. The strip is unchanged.
    """

    factor: Real[ArrayLike, ""]

    def __call__(
        self, kernel: Kernel, s: Inexact[ArrayLike, "..."]
    ) -> Inexact[Array, "..."]:
        s = jnp.asarray(s)
        return kernel(s) * jnp.exp(-s * jnp.log(self.factor))


@partial(register_dataclass, data_fields=("base", "op"), meta_fields=())
@dataclass(frozen=True)
class TransformedKernel(Kernel):
    """Kernel obtained by applying ``op`` to ``base``.

    Build it with ``Kernel.transform``.
    """

    base: Kernel
    op: Transform

    @property
    def domain(self) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        return self.op.domain(*self.base.domain)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        return self.op(self.base, s)


@partial(register_dataclass, data_fields=("mu",), meta_fields=())
@dataclass(frozen=True)
class BesselJKernel(Kernel):
    """Mellin kernel for the ordinary Bessel function ``J_mu``.

    Its open convergence strip is ``(-mu, 1.5)``.
    """

    mu: Real[ArrayLike, ""]

    @property
    def domain(self) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        return -jnp.asarray(self.mu), jnp.asarray(1.5)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        return _bessel_j_mellin(self.mu, s)


@partial(register_dataclass, data_fields=("ell",), meta_fields=())
@dataclass(frozen=True)
class SphericalBesselJKernel(Kernel):
    """Mellin kernel for the spherical Bessel function ``j_ell``.

    Its open convergence strip is ``(-ell, 2)``.
    """

    ell: Real[ArrayLike, ""]

    @property
    def domain(self) -> tuple[Real[Array, "..."], Real[Array, "..."]]:
        return -jnp.asarray(self.ell), jnp.asarray(2.0)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        # j_ell(x) = sqrt(pi/(2x)) J_(ell+1/2)(x).
        return jnp.sqrt(jnp.pi / 2) * _bessel_j_mellin(self.ell + 0.5, s - 0.5)
