"""Optional SymPy generation and numerical JAX behavior."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from mellin_helpers import bessel_factory, bessel_kernel
from numpy.testing import assert_allclose
from scipy.special import loggamma

from fftloggin import (
    BesselJKernel,
    DomainCheckWarning,
    Kernel,
    forward,
    inverse,
    lowring_log_kr,
    plan,
    product_plan,
    validate_parameters,
)
from fftloggin.symbolic import from_expression, from_mellin


@pytest.fixture
def symbols():
    return (
        sp.Symbol("x", positive=True),
        sp.Symbol("s"),
        sp.Symbol("rate", positive=True),
    )


@pytest.mark.parametrize("s", [0.8, 0.8 + 0.3j, [0.8 + 0.3j, 1.1 + 0.5j]])
def test_laplace_transform_and_metadata(x64, symbols, s):
    x, z, rate = symbols
    factory = from_expression(sp.exp(-rate * x), x, z, parameters=(rate,))
    reported = sp.mellin_transform(sp.exp(-rate * x), x, z)
    assert (factory.expression, factory.strip, factory.conditions) == reported
    assert factory.parameters == (rate,)
    kernel = factory(rate=2)
    s = np.asarray(s)
    assert_allclose(kernel(s), np.exp(loggamma(s + 0j) - s * np.log(2)), rtol=1e-11)
    assert_allclose(kernel.domain, (0, np.inf))
    assert isinstance(kernel, Kernel)
    with pytest.raises(FrozenInstanceError):
        kernel.values = ()  # ty: ignore[invalid-assignment]


def test_mellin_scaling_property(x64, symbols):
    x, s, rate = symbols
    scale = sp.Symbol("scale", positive=True)
    factory = from_expression(sp.exp(-rate * scale * x), x, s, parameters=(rate, scale))

    rate_value, scale_value = 1.7, 0.6
    z = np.array([0.8 + 0.3j, 1.2 - 0.4j])
    expected = np.exp(loggamma(z) - z * np.log(rate_value * scale_value))
    assert_allclose(factory(rate_value, scale_value)(z), expected, rtol=1e-11)


def test_mellin_power_multiplication_shifts_argument(x64, symbols):
    x, s, rate = symbols
    power = sp.Symbol("power", positive=True)
    factory = from_expression(
        x**power * sp.exp(-rate * x), x, s, parameters=(rate, power)
    )

    rate_value, power_value = 1.7, 0.6
    z = np.array([0.8 + 0.3j, 1.2 - 0.4j])
    shifted = z + power_value
    expected = np.exp(loggamma(shifted) - shifted * np.log(rate_value))
    assert_allclose(factory(rate_value, power_value)(z), expected, rtol=1e-11)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_mellin_derivative_shifts_argument(x64, symbols, order):
    x, s, rate = symbols
    factory = from_expression(
        sp.diff(sp.exp(-rate * x), x, order), x, s, parameters=(rate,)
    )

    rate_value = 1.7
    z = order + np.array([0.8 + 0.3j, 1.2 - 0.4j])
    polynomial = (-1) ** order * np.prod([z - k for k in range(1, order + 1)], axis=0)
    shifted = z - order
    base = np.exp(loggamma(shifted) - shifted * np.log(rate_value))
    assert_allclose(factory(rate_value)(z), polynomial * base, rtol=1e-11)


def test_mellin_reciprocal_argument_reflection(x64, symbols):
    x, s, rate = symbols
    factory = from_expression(sp.exp(-rate / x), x, s, parameters=(rate,))

    rate_value = 1.7
    z = np.array([-0.8 + 0.3j, -1.2 - 0.4j])
    expected = np.exp(loggamma(-z) + z * np.log(rate_value))
    assert_allclose(factory(rate_value)(z), expected, rtol=1e-11)


@pytest.mark.parametrize("scale", [0.7, 1, 2])
@pytest.mark.parametrize("imaginary", [0.3, 100, 2000])
def test_bessel_gamma_ratio_avoids_underflow(x64, scale, imaginary):
    s = 0.8 + 1j * imaginary
    expected = scale**-s * np.exp(
        (s - 1) * np.log(2) + loggamma((0.5 + s) / 2) - loggamma((2.5 - s) / 2)
    )
    assert_allclose(bessel_kernel(0.5, scale=scale)(s), expected, rtol=1e-10)


@pytest.mark.parametrize("power", [1, 2, -1, -2])
def test_integer_gamma_powers(x64, symbols, power):
    _, s, _ = symbols
    formula = sp.gamma(s / 2) ** power * sp.gamma((2 - s) / 2) ** -power
    factory = from_mellin(formula, s, strip=(0, 2))
    z = 0.8 + 2000j
    expected = np.exp(power * (loggamma(z / 2) - loggamma((2 - z) / 2)))
    assert_allclose(factory()(z), expected, rtol=1e-10)


@pytest.mark.parametrize(
    "formula",
    [lambda s: sp.log(sp.gamma(s)), lambda s: sp.gamma(s) ** sp.Rational(1, 2)],
)
def test_branch_sensitive_expressions(x64, symbols, formula):
    _, s, _ = symbols
    z = -0.3 + 1.7j
    expected = complex(formula(s).subs(s, z).evalf(30))
    assert_allclose(
        from_mellin(formula(s), s, strip=(-sp.oo, sp.oo))()(z), expected, rtol=1e-11
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_real_space_derivatives_weights_and_powers(x64, symbols, order):
    x, s, rate = symbols
    expression = sp.diff(
        3 * x**2 * sp.exp(-rate * x) + 2 * sp.exp(-2 * rate * x), x, order
    )
    factory = from_expression(expression, x, s, parameters=(rate,))
    z = order + 1.3 + 0.2j
    factor = (-1) ** order * np.prod([z - i for i in range(1, order + 1)])
    shifted = z - order
    expected = factor * (
        3 * np.exp(loggamma(shifted + 2) - (shifted + 2) * np.log(1.7))
        + 2 * np.exp(loggamma(shifted) - shifted * np.log(3.4))
    )
    assert_allclose(factory(1.7)(z), expected, rtol=1e-10)


def test_inferred_bessel_strip_matches_sympy(symbols):
    x, s, _ = symbols
    expression = sp.besselj(sp.Rational(1, 2), x)
    factory = from_expression(expression, x, s)
    assert (
        factory.expression,
        factory.strip,
        factory.conditions,
    ) == sp.mellin_transform(expression, x, s)


def test_explicit_strip_override_retains_conditions(symbols):
    x, s, _ = symbols
    a = sp.Symbol("a")
    expr = sp.exp(-a * x)
    reported = sp.mellin_transform(expr, x, s)
    factory = from_expression(expr, x, s, parameters=(a,), strip=(1, 3))
    assert factory.strip == (1, 3)
    assert factory.conditions == reported[2]
    assert bool(factory(2).is_in_domain(2))
    assert not bool(factory(-2).is_in_domain(2))


def test_auxiliary_boolean_conditions(symbols):
    _, s, _ = symbols
    a = sp.Symbol("a", real=True)
    condition = sp.And(a > 0, sp.Or(sp.re(s) < 2, a < 1))
    factory = from_mellin(1, s, parameters=(a,), strip=(0, 4), conditions=condition)
    assert bool(
        jax.jit(lambda k: k.is_in_domain(jnp.array([1 + 1j, 3 + 1j])))(factory(0.5))
    )
    assert not bool(factory(2).is_in_domain(jnp.array([1, 3])))
    with pytest.warns(DomainCheckWarning):
        validate_parameters(factory(-1), dlog=0.1)


@pytest.mark.parametrize("formula", [0, 3, sp.pi])
def test_constant_formulas_broadcast_and_transform(symbols, formula):
    _, s, _ = symbols
    kernel = from_mellin(formula, s, strip=(-sp.oo, sp.oo))()
    assert kernel(1).shape == ()
    assert_allclose(jax.jit(kernel)(jnp.array([1, 2, 3])), float(formula))


def test_factory_source_returns_generated_evaluator():
    s = sp.Symbol("s")
    factory = from_mellin(sp.exp(s), s, strip=(0, 2))

    source = factory.source()

    assert "def _lambdifygenerated" in source
    assert "jax.numpy.exp" in source


def test_dynamic_binding_pytrees_grad_and_nested_vmap(x64):
    factory = bessel_factory()
    samples = jnp.array([0.8 + 0.2j, 1.0 + 0.3j])
    evaluate = lambda a, z: factory(mu=a, power=0.2, scale=1.7)(z)
    kernel = factory(0.5, 0.2, 1.7)
    assert_allclose(
        jax.jit(lambda k, s: k(s))(kernel, samples), kernel(samples), rtol=1e-11
    )
    actual = jax.jit(jax.vmap(lambda a: jax.vmap(lambda z: evaluate(a, z))(samples)))(
        jnp.array([0.5, 0.6])
    )
    assert_allclose(
        actual, jnp.stack([evaluate(a, samples) for a in [0.5, 0.6]]), rtol=1e-11
    )
    loss = lambda a: jnp.real(evaluate(a, samples)).sum()
    step = 1e-5
    assert_allclose(
        jax.jit(jax.grad(loss))(0.5),
        (loss(0.5 + step) - loss(0.5 - step)) / (2 * step),
        rtol=1e-6,
    )
    differentiated = jax.grad(lambda k: jnp.real(k(samples)).sum())(kernel)
    assert all(bool(jnp.isfinite(v)) for v in jax.tree.leaves(differentiated))


@pytest.mark.parametrize("samples", [0.8 + 0.3j, [0.8 + 0.3j, 1.1 + 0.5j]])
def test_check_jax_successful_paths(samples):
    _, s, rate = (sp.Symbol("x"), sp.Symbol("s"), sp.Symbol("rate"))
    factory = from_mellin(sp.gamma(s) * rate, s, parameters=(rate,), strip=(0, 2))
    assert factory.check_jax(samples, 2) is None
    assert factory.check_jax(samples, rate=2) is None


def test_check_jax_accepts_integer_parameter():
    _, s, rate = sp.Symbol("x"), sp.Symbol("s"), sp.Symbol("rate")
    factory = from_mellin(rate * s, s, parameters=(rate,), strip=(0, 2))
    assert factory.check_jax(jnp.array([0.8, 1.1]), 2) is None


def test_check_jax_identifies_jit_failure():
    from sympy.utilities.lambdify import implemented_function

    _, s, _ = sp.Symbol("x"), sp.Symbol("s"), sp.Symbol("rate")
    f = implemented_function("python_only", lambda z: float(z))
    factory = from_mellin(
        f(s), s, strip=(0, 2), functions={"python_only": lambda z: float(z)}
    )
    with pytest.raises(ValueError, match="jit"):
        factory.check_jax(1.0)


def test_check_jax_identifies_vmap_failure():
    from jax import ShapeDtypeStruct

    def callback(z):
        return jax.pure_callback(
            lambda x: np.asarray(x), ShapeDtypeStruct(z.shape, z.dtype), z
        )

    _, s, _ = sp.Symbol("x"), sp.Symbol("s"), sp.Symbol("rate")
    f = cast(Callable[..., sp.Expr], sp.Function("callback"))
    factory = from_mellin(f(s), s, strip=(0, 2), functions={"callback": callback})
    with pytest.raises(ValueError, match="vmap"):
        factory.check_jax(1.0)


def test_check_jax_identifies_grad_failure():
    @jax.custom_jvp
    def no_jvp(z):
        return jnp.sin(z)

    @no_jvp.defjvp
    def no_jvp_rule(primals, tangents):
        raise NotImplementedError("derivative unavailable")

    _, s, _ = sp.Symbol("x"), sp.Symbol("s"), sp.Symbol("rate")
    f = cast(Callable[..., sp.Expr], sp.Function("no_jvp"))
    factory = from_mellin(f(s), s, strip=(0, 2), functions={"no_jvp": no_jvp})
    with pytest.raises(ValueError, match="grad"):
        factory.check_jax(1.0)


def test_all_fftlog_entrypoints(x64):
    generated, direct = bessel_kernel(0.5), BesselJKernel(0.5)
    n, dlog = 32, 0.2
    a = jnp.exp(-(jnp.linspace(-2, 2, n) ** 2))
    snap = jax.jit(lambda k: lowring_log_kr(k, dlog=dlog))(generated)
    assert_allclose(snap, lowring_log_kr(direct, dlog=dlog), atol=1e-11)
    p = jax.jit(lambda k: plan(k, n, dlog=dlog, log_kr=snap))(generated)
    expected = forward(a, direct, dlog=dlog, log_kr=snap)
    assert_allclose(jax.jit(forward)(a, p), expected, rtol=1e-10, atol=1e-11)
    assert_allclose(inverse(forward(a, p), p), a, rtol=1e-10, atol=1e-11)
    pp = product_plan(
        generated,
        generated,
        n=n,
        dlog=dlog,
        first_bias=-0.5,
        second_bias=-0.5,
        max_offset=2,
    )
    reference = product_plan(
        direct, direct, n=n, dlog=dlog, first_bias=-0.5, second_bias=-0.5, max_offset=2
    )
    assert_allclose(forward(a, pp), forward(a, reference), rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "kind", ["undeclared", "integral", "derivative", "transform", "function"]
)
def test_generation_rejects_unresolved_or_unsupported(symbols, kind):
    x, s, _ = symbols
    f = cast(Callable[..., sp.Expr], sp.Function("f"))
    expressions = {
        "undeclared": sp.Symbol("other") + s,
        "integral": sp.Integral(sp.exp(-x), (x, 0, sp.oo)),
        "derivative": sp.Derivative(f(s), s),
        "transform": sp.MellinTransform(f(x), x, s),
        "function": sp.besselj(0, s),
    }
    with pytest.raises(ValueError):
        from_mellin(expressions[kind], s, strip=(0, 1))
    with pytest.raises(ValueError):
        from_expression(f(x), x, s)


@pytest.mark.parametrize(
    "parameters",
    [
        lambda a: [a],
        lambda a: (a, a),
        lambda a: (1,),
        lambda a: (sp.Symbol("s"),),
        lambda a: (a, sp.Symbol(a.name, real=True)),
    ],
)
def test_invalid_parameter_declarations(symbols, parameters):
    _, s, a = symbols
    with pytest.raises((TypeError, ValueError)):
        from_mellin(1, s, strip=(0, 1), parameters=parameters(a))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((1, 2), {}),
        ((), {}),
        ((1,), {"rate": 2}),
        ((), {"other": 1}),
        (([1, 2],), {}),
        ((True,), {}),
        (("2",), {}),
    ],
)
def test_binding_errors(symbols, args, kwargs):
    _, s, a = symbols
    factory = from_mellin(a * s, s, parameters=(a,), strip=(0, 1))
    with pytest.raises(TypeError):
        factory(*args, **kwargs)


@pytest.mark.parametrize(
    "assumption,value",
    [("positive", -1), ("real", 1j), ("integer", 1.5), ("prime", 2)],
)
def test_symbolic_assumptions_are_caller_responsibility(symbols, assumption, value):
    _, s, _ = symbols
    a = sp.Symbol("a", **{assumption: True})
    kernel = from_mellin(a * s, s, parameters=(a,), strip=(0, 2))(value)
    kernel.validate_parameters()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_eager_finiteness_validation(symbols, value):
    _, s, a = symbols
    kernel = from_mellin(a * s, s, parameters=(a,), strip=(0, 2))(value)
    with pytest.raises(ValueError, match="finite"):
        kernel.validate_parameters()


def test_jax_function_mapping(symbols):
    _, s, a = symbols
    f = cast(Callable[..., sp.Expr], sp.Function("custom"))
    factory = from_mellin(
        f(s) + a, s, parameters=(a,), strip=(0, 2), functions={"custom": jnp.sin}
    )
    assert_allclose(jax.jit(lambda a: factory(a)(1.2))(2.0), jnp.sin(1.2) + 2)
    assert_allclose(jax.grad(lambda a: factory(a)(1.2))(2.0), 1)


def test_piecewise_and_parameter_dependent_min_max_strips(symbols):
    _, s, _ = symbols
    a = sp.Symbol("a", real=True)
    expression = sp.Piecewise((sp.Max(sp.re(s), 1), sp.re(s) > a), (a, True))
    factory = from_mellin(
        expression, s, parameters=(a,), strip=(sp.Max(a, 0), sp.Min(a + 4, 5))
    )
    kernel = factory(0.5)
    samples = jnp.array([0.2 + 1j, 0.8 + 1j, 2 + 1j])
    assert_allclose(jax.jit(kernel)(samples), [0.5, 1, 2])
    assert_allclose(jax.jit(lambda k: k.domain)(kernel), (0.5, 4.5))
    assert bool(kernel.is_in_domain(1))
    assert not bool(kernel.is_in_domain(0.5))


@pytest.mark.parametrize(
    "kind",
    [
        "strip_symbol",
        "condition_symbol",
        "bad_strip",
        "complex_strip",
        "bad_conditions",
        "bad_mapping",
        "coordinate",
    ],
)
def test_generation_metadata_errors(symbols, kind):
    _, s, a = symbols
    # These cases deliberately pass invalid values to exercise runtime checks.
    options: dict[str, Any] = {"strip": (0, 2), "parameters": (a,)}
    coordinate: Any = s
    if kind == "strip_symbol":
        options["strip"] = (sp.Symbol("other"), 2)
    elif kind == "condition_symbol":
        options["conditions"] = sp.Symbol("other") > 0
    elif kind == "bad_strip":
        options["strip"] = (0,)
    elif kind == "complex_strip":
        options["strip"] = (sp.I, 2)
    elif kind == "bad_conditions":
        options["conditions"] = s + 1
    elif kind == "bad_mapping":
        options["functions"] = {"custom": 1}
    elif kind == "coordinate":
        coordinate = 1
    with pytest.raises((ValueError, TypeError)):
        from_mellin(1, cast(Any, coordinate), **options)


def test_undefined_function_never_uses_sympy_implementation(symbols):
    from sympy.utilities.lambdify import implemented_function

    _, s, _ = symbols
    f = implemented_function("not_jax", lambda z: float(z))
    with pytest.raises(ValueError, match="unsupported"):
        from_mellin(f(s), s, strip=(0, 2))
    kernel = from_mellin(f(s), s, strip=(0, 2), functions={"not_jax": jnp.sin})()
    assert_allclose(jax.jit(kernel)(1.0), jnp.sin(1.0))


def test_complex_constants_and_parameter_values(x64, symbols):
    _, s, _ = symbols
    a = sp.Symbol("a")
    factory = from_mellin(
        sp.exp(sp.I * s) * sp.gamma(s + a), s, parameters=(a,), strip=(0, 2)
    )
    z, value = 0.8 + 0.3j, 0.2 + 0.5j
    kernel = factory(value)
    kernel.validate_parameters()
    assert_allclose(
        jax.jit(kernel)(z), np.exp(1j * z + loggamma(z + value)), rtol=1e-11
    )
