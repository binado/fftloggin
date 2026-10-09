"""Public behavior of symbolic real-space kernels."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike
from numpy.testing import assert_allclose

from fftloggin import (
    BesselJKernel,
    Coordinate,
    Kernel,
    KernelExpression,
    diff,
    forward,
    inverse,
    lowring_log_kr,
    plan,
    product_plan,
    validate_parameters,
)


@pytest.fixture
def t():
    return Coordinate("t")


@pytest.fixture
def j():
    return BesselJKernel(0.5)


@pytest.mark.parametrize("scalar", [int, float, np.float64, np.asarray, jnp.asarray])
def test_reflected_scalars_and_monomials(x64, t, j, scalar):
    scalar = scalar(2)
    s = jnp.array([0.8 + 0.2j, 1.1 + 0.3j])
    expression = scalar * j(t) + j(t) * scalar - -j(t)
    assert isinstance(expression, KernelExpression)
    assert_allclose(expression(s), (2 * scalar + 1) * j(s), rtol=1e-11)
    monomial = (scalar * t) * t**0.2
    assert_allclose((monomial * j(t))(s), scalar * j(s + 1.2), rtol=1e-11)
    assert_allclose((j(t) * monomial)(s), scalar * j(s + 1.2), rtol=1e-11)
    assert_allclose(j(scalar * t)(s), j(s) * scalar**-s, rtol=1e-11)


@pytest.mark.parametrize("s", [0.8 + 0.3j, np.array([0.8 + 0.2j, 1.1 + 0.3j])])
def test_nested_composition_and_complete_rescaling(x64, t, j, s):
    expr = 2 * t**0.2 * j(1.7 * t) - diff(j(t), t)
    expected = 2 * 1.7 ** -(s + 0.2) * j(s + 0.2) + (s - 1) * j(s - 1)
    assert_allclose(expr(s), expected, rtol=1e-11, atol=1e-12)
    assert_allclose(expr(0.7 * t)(s), 0.7**-s * expected, rtol=1e-11)
    assert_allclose(expr.domain, (0.5, 1.3))


@pytest.mark.parametrize("order", [0, 1, 2, np.int64(3)])
def test_derivative_orders(x64, t, j, order):
    expr = j(t)
    derivative = diff(expr, Coordinate("t"), order=order)
    if order == 0:
        assert derivative is expr
    s = order + 0.8 + 0.2j
    expected = (
        (-1) ** order * np.prod([s - r for r in range(1, order + 1)]) * j(s - order)
    )
    assert_allclose(derivative(s), expected, rtol=1e-11)


@pytest.mark.parametrize("order", [True, False, -1, 1.5, jnp.array(1)])
def test_invalid_orders(t, j, order):
    with pytest.raises(ValueError, match="order"):
        diff(j(t), t, order=order)


def test_coordinate_identity_rebinding_and_empty_strip(t, j):
    u = Coordinate("u")
    expr = t**0.2 * j(t)
    rebound = expr(u)
    assert_allclose((rebound + j(u))(0.8), expr(0.8) + j(0.8))
    assert_allclose((expr + j(Coordinate("t"))).domain, (-0.5, 1.3))
    empty = j(t) + diff(j(t), t, order=3)
    assert_allclose(empty.domain, (2.5, 1.5))
    assert not bool(empty.is_in_domain(2))
    assert_allclose((expr - expr).domain, expr.domain)


@pytest.mark.parametrize(
    "operation,match",
    [
        (lambda t, j: j(t) + j(Coordinate("u")), "different coordinates"),
        (lambda t, j: t * j(Coordinate("u")), "different coordinates"),
        (lambda t, j: diff(j(t), Coordinate("u")), "different coordinates"),
        (lambda t, j: j(t + 1), "translations"),
        (lambda t, j: j(t**2), "linear coordinates"),
        (lambda t, j: j(t * t), "linear coordinates"),
        (lambda t, j: j(t) * j(t), "products"),
        (lambda t, j: j(t) ** 2, "powers"),
        (lambda t, j: j(t) + 2, "addition"),
        (lambda t, j: 2 + j(t), "addition"),
        (lambda t, j: 2 - j(t), "subtraction"),
        (lambda t, j: j(t) - 2, "subtraction"),
        (lambda t, j: j(j(t)), "linear coordinates"),
        (lambda t, j: diff(t, t), "kernel expression"),
    ],
)
def test_unsupported_operations(t, j, operation, match):
    with pytest.raises(TypeError, match=match):
        operation(t, j)


@pytest.mark.parametrize("value", [True, 1j, [1.0], np.ones(2), jnp.ones(2), "bad"])
@pytest.mark.parametrize(
    "operation",
    [
        lambda t, j, x: x * j(t),
        lambda t, j, x: t**x * j(t),
        lambda t, j, x: j(x * t),
    ],
)
def test_parameters_must_be_real_scalars(t, j, value, operation):
    with pytest.raises(TypeError, match="real scalar"):
        operation(t, j, value)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("scale", 0.0),
        ("scale", -1.0),
        ("scale", np.inf),
        ("weight", np.nan),
        ("power", np.inf),
    ],
)
def test_eager_validation_recurses_through_expression(t, j, kind, value):
    if kind == "scale":
        expr = j(value * t)
    elif kind == "weight":
        expr = value * j(t)
    else:
        expr = t**value * j(t)
    with pytest.raises(ValueError, match="finite|positive"):
        validate_parameters(diff(j(t) + expr, t), dlog=0.1)


def test_jax_construction_evaluation_and_parameter_gradients(x64, t):
    def evaluate(parameters, s):
        mu, weight, exponent, scale = parameters
        expr = weight * t**exponent * BesselJKernel(mu)(scale * t)
        return (expr + diff(expr, t))(s)

    parameters = jnp.array([0.5, 2.0, 0.2, 1.7])
    s = 1.1 + 0.3j
    compiled = jax.jit(evaluate)
    assert_allclose(compiled(parameters, s), evaluate(parameters, s), rtol=1e-11)
    actual = jax.jit(jax.grad(lambda p: jnp.real(evaluate(p, s))))(parameters)
    step = 1e-5
    expected = [
        (
            jnp.real(evaluate(parameters.at[i].add(step), s))
            - jnp.real(evaluate(parameters.at[i].add(-step), s))
        )
        / (2 * step)
        for i in range(4)
    ]
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    batch = jnp.stack([parameters, parameters * 1.1])
    samples = jnp.array([s, s + 0.1j])
    got = jax.jit(jax.vmap(lambda p: jax.vmap(lambda z: evaluate(p, z))(samples)))(
        batch
    )
    expected = jnp.stack([evaluate(p, samples) for p in batch])
    assert_allclose(got, expected, rtol=1e-11)


@partial(register_dataclass, data_fields=("rate",), meta_fields=())
@dataclass(frozen=True)
class CustomKernel(Kernel):
    rate: ArrayLike

    @property
    def domain(self):
        return jnp.asarray(-2.0), jnp.asarray(2.0)

    def mellin(self, s: ArrayLike) -> jax.Array:
        return jnp.exp(-jnp.asarray(self.rate) * jnp.asarray(s) ** 2)


def test_registered_custom_mellin_kernel(t):
    kernel = CustomKernel(0.2)
    expression = t**0.1 * kernel(1.7 * t)
    s = jnp.array([0.8 + 0.1j, 1.0 + 0.2j])
    expected = 1.7 ** -(s + 0.1) * kernel.mellin(s + 0.1)
    got = jax.jit(lambda k, z: k(z))(expression, s)
    assert_allclose(got, expected, rtol=1e-6)
    assert jnp.isfinite(
        jax.grad(lambda rate: jnp.real(CustomKernel(rate)(t)(0.8)))(0.2)
    )


def test_fftlog_expression_plans_linearity_and_roundtrip(x64, t, j):
    n, dlog, bias, log_kr = 64, 0.1, 0.0, 0.2
    a = jnp.exp(-(jnp.linspace(-3, 3, n) ** 2))
    first, second = t**0.1 * j(t), BesselJKernel(1.0)(1.2 * t)
    expr = 2 * first + second
    kwargs = {"dlog": dlog, "bias": bias, "log_kr": log_kr}
    p = plan(expr, n, **kwargs)
    expected = 2 * forward(a, first, **kwargs) + forward(a, second, **kwargs)
    assert_allclose(forward(a, p), expected, rtol=1e-11, atol=1e-12)
    assert_allclose(forward(a, expr, **kwargs), expected, rtol=1e-11)
    nonzero = plan(2 * j(t), n, **kwargs)
    assert_allclose(inverse(forward(a, nonzero), nonzero), a, rtol=1e-11, atol=1e-12)
    assert jnp.isfinite(lowring_log_kr(expr, dlog=dlog, bias=bias))
    symbolic_product = product_plan(
        first,
        second,
        n=n,
        max_offset=2,
        dlog=dlog,
        log_kr=log_kr,
        first_bias=bias,
        second_bias=bias,
    )
    explicit_product = product_plan(
        plan(first, n, **kwargs), plan(second, n, **kwargs), max_offset=2
    )
    assert_allclose(
        forward(a, symbolic_product), forward(a, explicit_product), rtol=1e-11
    )


@pytest.mark.parametrize(
    "dtype,rtol", [(jnp.int32, 1e-6), (jnp.float32, 1e-6), (jnp.bfloat16, 1e-2)]
)
def test_real_jax_scalar_dtypes(t, j, dtype, rtol):
    scalar = jnp.asarray(2, dtype=dtype)
    assert_allclose((scalar * j(t))(0.8), 2 * j(0.8), rtol=rtol)
