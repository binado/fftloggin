"""Scalar Mellin kernels for JAX FFTLog transforms."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import loggamma as _jax_loggamma
from jax.tree_util import register_dataclass
from jaxtyping import Array, ArrayLike, Bool, Float, Inexact

__all__ = (
    "BesselJKernel",
    "Derivative",
    "Kernel",
    "ShiftedKernel",
    "SphericalBesselJKernel",
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
    mu: Float[ArrayLike, ""], s: Inexact[ArrayLike, "..."]
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
    def domain(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        raise NotImplementedError

    def is_in_domain(self, s: Inexact[ArrayLike, "..."]) -> Bool[Array, ""]:
        lower, upper = self.domain
        return jnp.all((jnp.real(s) > lower) & (jnp.real(s) < upper))


@partial(register_dataclass, data_fields=("base", "nu"), meta_fields=())
@dataclass(frozen=True)
class ShiftedKernel(Kernel):
    base: Kernel
    nu: Float[ArrayLike, ""]

    @property
    def domain(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        lower, upper = self.base.domain
        return lower - self.nu, upper - self.nu

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        return self.base(s + self.nu)


@partial(register_dataclass, data_fields=("base",), meta_fields=("order",))
@dataclass(frozen=True)
class Derivative(Kernel):
    base: Kernel
    order: int

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order must be a positive integer")

    @property
    def domain(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        lower, upper = self.base.domain
        return lower + self.order, upper + self.order

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        s = jnp.asarray(s)
        factor = jnp.prod(s[..., None] - jnp.arange(1, self.order + 1), axis=-1)
        return (-1) ** self.order * factor * self.base(s - self.order)


@partial(register_dataclass, data_fields=("mu",), meta_fields=())
@dataclass(frozen=True)
class BesselJKernel(Kernel):
    mu: Float[ArrayLike, ""]

    @property
    def domain(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return -jnp.asarray(self.mu), jnp.asarray(1.5)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        return _bessel_j_mellin(self.mu, s)


@partial(register_dataclass, data_fields=("ell",), meta_fields=())
@dataclass(frozen=True)
class SphericalBesselJKernel(Kernel):
    ell: Float[ArrayLike, ""]

    @property
    def domain(self) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return -jnp.asarray(self.ell), jnp.asarray(2.0)

    def __call__(self, s: Inexact[ArrayLike, "..."]) -> Inexact[Array, "..."]:
        # j_ell(x) = sqrt(pi/(2x)) J_(ell+1/2)(x).
        return jnp.sqrt(jnp.pi / 2) * _bessel_j_mellin(self.ell + 0.5, s - 0.5)
