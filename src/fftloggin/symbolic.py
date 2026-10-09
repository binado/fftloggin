"""Real-space kernel composition through known Mellin identities.

Derivative strips assume the boundary terms in repeated integration by parts
vanish. Sum strips are conservative intersections, even when terms cancel.
"""

from dataclasses import dataclass
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike

from .kernels import Kernel

__all__ = ("Coordinate", "KernelExpression", "diff")


def _real_scalar(value, operation):
    try:
        arr = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{operation} requires a real scalar") from error
    if arr.ndim != 0 or not (
        jnp.issubdtype(arr.dtype, jnp.integer)
        or jnp.issubdtype(arr.dtype, jnp.floating)
    ):
        raise TypeError(f"{operation} requires a real scalar")
    return value


def _same_coordinate(first, second):
    if first.name != second.name:
        raise TypeError("cannot combine different coordinates; explicitly rebind first")


class _Factor:
    name: str
    __array_priority__ = 1000

    def _monomial(self) -> "_Monomial":
        raise NotImplementedError

    def __mul__(self, other):
        first = self._monomial()
        if isinstance(other, KernelExpression):
            _same_coordinate(first, other)
            return _Weight(
                _Power(other, first.exponent, first.name), first.coefficient, first.name
            )
        if isinstance(other, _Factor):
            second = other._monomial()
            _same_coordinate(first, second)
            return _Monomial(
                first.coefficient * second.coefficient,
                first.exponent + second.exponent,
                first.name,
            )
        return _Monomial(
            first.coefficient * _real_scalar(other, "multiplication"),
            first.exponent,
            first.name,
        )

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        raise TypeError("coordinate translations are unsupported")

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__


@dataclass(frozen=True)
class Coordinate(_Factor):
    """A named real-space coordinate; equal names identify the same variable."""

    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("coordinate name must be a nonempty string")

    def _monomial(self):
        return _Monomial(1, 1, self.name)

    def __pow__(self, exponent):
        return _Monomial(1, _real_scalar(exponent, "coordinate power"), self.name)


@partial(
    register_dataclass, data_fields=("coefficient", "exponent"), meta_fields=("name",)
)
@dataclass(frozen=True)
class _Monomial(_Factor):
    coefficient: ArrayLike
    exponent: ArrayLike
    name: str

    def _monomial(self):
        return self


class KernelExpression(Kernel):
    """Frozen pytree kernel representing a bound real-space expression."""

    name: str
    __array_priority__ = 1000

    def __mul__(self, other):
        if isinstance(other, _Factor):
            return other * self
        if isinstance(other, Kernel):
            raise TypeError("products of complete kernels are unsupported")
        return _Weight(self, _real_scalar(other, "kernel weighting"), self.name)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, KernelExpression):
            raise TypeError("kernel addition requires another bound kernel expression")
        _same_coordinate(self, other)
        return _Sum(self, other, self.name)

    __radd__ = __add__

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        if not isinstance(other, KernelExpression):
            raise TypeError(
                "kernel subtraction requires another bound kernel expression"
            )
        return self + (-other)

    def __rsub__(self, other):
        raise TypeError("kernel subtraction requires another bound kernel expression")

    def __pow__(self, exponent):
        raise TypeError("powers of complete kernels are unsupported")


@partial(register_dataclass, data_fields=("base",), meta_fields=("name",))
@dataclass(frozen=True)
class _Bound(KernelExpression):
    base: Kernel
    name: str

    @property
    def domain(self):
        return self.base.domain

    def mellin(self, s: ArrayLike) -> jax.Array:
        return self.base.mellin(s)


@partial(register_dataclass, data_fields=("base", "factor"), meta_fields=("name",))
@dataclass(frozen=True)
class _Scaled(KernelExpression):
    base: Kernel
    factor: ArrayLike
    name: str

    @property
    def domain(self):
        return self.base.domain

    def mellin(self, s: ArrayLike) -> jax.Array:
        s = jnp.asarray(s)
        return jnp.exp(-s * jnp.log(self.factor)) * self.base.mellin(s)


@partial(register_dataclass, data_fields=("base", "exponent"), meta_fields=("name",))
@dataclass(frozen=True)
class _Power(KernelExpression):
    base: Kernel
    exponent: ArrayLike
    name: str

    @property
    def domain(self):
        lower, upper = self.base.domain
        return lower - self.exponent, upper - self.exponent

    def mellin(self, s: ArrayLike) -> jax.Array:
        return self.base.mellin(jnp.asarray(s) + self.exponent)


@partial(register_dataclass, data_fields=("base", "weight"), meta_fields=("name",))
@dataclass(frozen=True)
class _Weight(KernelExpression):
    base: Kernel
    weight: ArrayLike
    name: str

    @property
    def domain(self):
        return self.base.domain

    def mellin(self, s: ArrayLike) -> jax.Array:
        return jnp.asarray(self.weight) * self.base.mellin(s)


@partial(register_dataclass, data_fields=("first", "second"), meta_fields=("name",))
@dataclass(frozen=True)
class _Sum(KernelExpression):
    first: KernelExpression
    second: KernelExpression
    name: str

    @property
    def domain(self):
        lo1, hi1 = self.first.domain
        lo2, hi2 = self.second.domain
        return jnp.maximum(lo1, lo2), jnp.minimum(hi1, hi2)

    def mellin(self, s: ArrayLike) -> jax.Array:
        return self.first.mellin(s) + self.second.mellin(s)


@partial(register_dataclass, data_fields=("base",), meta_fields=("order", "name"))
@dataclass(frozen=True)
class _Differentiated(KernelExpression):
    base: KernelExpression
    order: int
    name: str

    @property
    def domain(self):
        lower, upper = self.base.domain
        return lower + self.order, upper + self.order

    def mellin(self, s: ArrayLike) -> jax.Array:
        s = jnp.asarray(s)
        factor = jnp.prod(s[..., None] - jnp.arange(1, self.order + 1), axis=-1)
        return (-1) ** self.order * factor * self.base.mellin(s - self.order)


def _bind(kernel, argument):
    if isinstance(argument, Coordinate):
        return _Bound(kernel, argument.name)
    # Exponents are inspected only for argument structure. Traced or non-unit
    # exponents are valid monomial factors but cannot be kernel arguments.
    if not isinstance(argument.exponent, Integral) or argument.exponent != 1:
        raise TypeError("kernel arguments must be linear coordinates")
    return _Scaled(kernel, argument.coefficient, argument.name)


def diff(
    expression: KernelExpression, coordinate: Coordinate, order=1
) -> KernelExpression:
    """Differentiate a bound kernel, assuming vanishing Mellin boundary terms.

    The order is a static nonnegative integer. No derivative expansion or
    cancellation-based widening of convergence strips is performed.
    """
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("order must be a static nonnegative integer")
    if not isinstance(expression, KernelExpression) or not isinstance(
        coordinate, Coordinate
    ):
        raise TypeError("diff requires a kernel expression and a Coordinate")
    _same_coordinate(expression, coordinate)
    return (
        expression
        if order == 0
        else _Differentiated(expression, int(order), expression.name)
    )


def _validate_expression(kernel):
    """Validate concrete expression data recursively, outside JAX tracing."""
    if isinstance(kernel, (_Scaled, _Power, _Weight)):
        attribute = {_Scaled: "factor", _Power: "exponent", _Weight: "weight"}[
            type(kernel)
        ]
        value = jnp.asarray(getattr(kernel, attribute))
        if not bool(jnp.isfinite(value)):
            raise ValueError(f"expression {attribute} must be finite")
        if isinstance(kernel, _Scaled) and not bool(value > 0):
            raise ValueError("argument scale must be strictly positive")
    if isinstance(kernel, _Sum):
        _validate_expression(kernel.first)
        _validate_expression(kernel.second)
    elif isinstance(kernel, KernelExpression):
        _validate_expression(kernel.base)
