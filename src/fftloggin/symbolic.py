"""Eager SymPy generation of numerical JAX Mellin kernel factories.

Install ``fftloggin[symbolic]`` to use this module. Create factories outside
JAX transformations; binding their scalar parameters is traceable.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

try:
    import sympy as sp
    from sympy.core.relational import Relational
    from sympy.functions.elementary.piecewise import ExprCondPair
    from sympy.integrals.transforms import IntegralTransform, IntegralTransformError
    from sympy.printing.numpy import JaxPrinter
except ImportError as error:
    raise ImportError(
        "fftloggin.symbolic requires SymPy; install `pip install "
        "'fftloggin[symbolic]'` or `uv add 'fftloggin[symbolic]'`."
    ) from error

from .kernels import GeneratedKernel, _loggamma

__all__ = ("KernelFactory", "from_expression", "from_mellin")


def _gamma(z: jax.typing.ArrayLike) -> jax.Array:
    return jnp.exp(_loggamma(jnp.asarray(z) + 0j))


def _complex_loggamma(z: jax.typing.ArrayLike) -> jax.Array:
    return _loggamma(jnp.asarray(z) + 0j)


_FUNCTIONS = {
    "gamma": _gamma,
    "loggamma": _complex_loggamma,
    "Abs": jnp.abs,
    "re": jnp.real,
    "im": jnp.imag,
    "arg": jnp.angle,
    "conjugate": jnp.conj,
    "exp": jnp.exp,
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "atan2": jnp.arctan2,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "asinh": jnp.arcsinh,
    "acosh": jnp.arccosh,
    "atanh": jnp.arctanh,
    "sign": jnp.sign,
    "floor": jnp.floor,
    "ceiling": jnp.ceil,
    "Min": jnp.minimum,
    "Max": jnp.maximum,
}


def _log_products(expression):
    """Combine integer gamma powers in each product without moving branches.

    Other factors stay outside exp: in particular, log(gamma(z)), fractional
    gamma powers and arbitrary complex powers are never expanded.
    """
    if not expression.args:
        return expression
    if isinstance(expression, (sp.Mul, sp.Pow)) or expression.func == sp.gamma:
        factors = sp.Mul.make_args(expression)
        logs, rest = [], []
        for factor in factors:
            base, exponent = factor.as_base_exp()
            if base.func == sp.gamma and isinstance(exponent, sp.Integer):
                logs.append(exponent * sp.loggamma(base.args[0]))
            else:
                rest.append(factor)
        if logs:
            return sp.Mul(*(_log_products(a) for a in rest)) * sp.exp(sp.Add(*logs))
    return expression.func(*(_log_products(a) for a in expression.args))


def _declarations(s, parameters, x=None):
    if not isinstance(s, sp.Symbol) or (x is not None and not isinstance(x, sp.Symbol)):
        raise TypeError("transform coordinates must be SymPy symbols")
    if not isinstance(parameters, tuple) or any(
        not isinstance(p, sp.Symbol) for p in parameters
    ):
        raise TypeError("parameters must be an ordered tuple of SymPy symbols")
    symbols = (s, *parameters) if x is None else (x, s, *parameters)
    if len(set(symbols)) != len(symbols) or len({p.name for p in symbols}) != len(
        symbols
    ):
        raise ValueError(
            "coordinates and parameters must have distinct symbols and names"
        )


def _check(expression, allowed):
    if expression.free_symbols - set(allowed):
        raise ValueError(
            f"undeclared symbols: {expression.free_symbols - set(allowed)}"
        )
    if expression.has(
        IntegralTransform, sp.Integral, sp.Derivative, sp.Sum, sp.Product
    ):
        raise ValueError(
            "unevaluated transforms, integrals, derivatives or sums are unsupported"
        )


class _Printer(JaxPrinter):
    """Broadcast scalar and array operands instead of stacking them."""

    def _fold(self, expression, function):
        function = self._module_format(function)
        parts = [self._print(arg) for arg in expression.args]
        result = parts[0]
        for part in parts[1:]:
            result = f"{function}({result}, {part})"
        return result

    def _print_And(self, expr):
        return self._fold(expr, "jax.numpy.logical_and")

    def _print_Or(self, expr):
        return self._fold(expr, "jax.numpy.logical_or")

    def _print_Min(self, expr):
        return self._fold(expr, "jax.numpy.minimum")

    def _print_Max(self, expr):
        return self._fold(expr, "jax.numpy.maximum")


def _compile(expression, arguments, functions, cse):
    expressions = expression if isinstance(expression, tuple) else (expression,)
    for expr in expressions:
        _check(expr, arguments)
        for node in sp.preorder_traversal(expr):
            if (
                isinstance(node, sp.Function) and not isinstance(node, sp.Piecewise)
            ) or node.func in (sp.Min, sp.Max):
                if node.func.__name__ not in functions:
                    raise ValueError(
                        f"unsupported numerical function: {node.func.__name__}"
                    )
            elif node != sp.I and not isinstance(
                node,
                (
                    sp.Symbol,
                    sp.Number,
                    sp.NumberSymbol,
                    sp.Add,
                    sp.Mul,
                    sp.Pow,
                    Relational,
                    sp.logic.boolalg.Boolean,
                    sp.Piecewise,
                    ExprCondPair,
                ),
            ):
                raise ValueError(f"unsupported numerical expression: {node}")
    # Explicit printer mappings prevent undefined functions from invoking Python
    # or SymPy evaluation, and override the backend's real-only special functions.
    printer = _Printer({"user_functions": {name: name for name in functions}})
    try:
        return sp.lambdify(
            arguments,
            expression,
            modules=[functions, "jax"],
            printer=printer,
            cse=cse,
            docstring_limit=0,
            use_imps=False,
            dummify=True,
        )
    except (NotImplementedError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported numerical expression: {expression}") from error


def _assumptions(parameter):
    if (
        parameter.is_finite is False
        or parameter.is_complex is False
        or parameter.is_commutative is False
    ):
        raise ValueError("parameters must admit finite numerical scalar values")
    predicates = {
        "real": lambda v: jnp.imag(v) == 0,
        "imaginary": lambda v: (jnp.real(v) == 0) & (jnp.imag(v) != 0),
        "positive": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) > 0),
        "negative": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) < 0),
        "nonnegative": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) >= 0),
        "nonpositive": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) <= 0),
        "nonzero": lambda v: (jnp.imag(v) == 0) & (v != 0),
        "zero": lambda v: v == 0,
        "integer": lambda v: (
            (jnp.imag(v) == 0) & (jnp.real(v) == jnp.floor(jnp.real(v)))
        ),
        "even": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) % 2 == 0),
        "odd": lambda v: (jnp.imag(v) == 0) & (jnp.real(v) % 2 == 1),
    }
    checks = []
    for name, predicate in predicates.items():
        assumed = parameter.assumptions0.get(name)
        if assumed is True:
            checks.append((name, predicate))
        elif assumed is False:
            checks.append((f"not {name}", lambda v, fn=predicate: ~fn(v)))
    unsupported = {
        "prime",
        "composite",
        "irrational",
        "rational",
        "algebraic",
        "transcendental",
    }
    # Derived assumptions (e.g. integer implies rational) need no extra checks.
    explicit = parameter._assumptions_orig
    if unsupported.intersection(explicit):
        raise ValueError("unsupported parameter assumptions")
    return parameter.name, tuple(checks)


@dataclass(frozen=True)
class KernelFactory:
    """Reusable factory with inspectable symbolic formula and convergence data.

    Bind every declared parameter positionally or by its symbol name. Binding
    checks scalar numeric shape/dtype; call ``kernel.validate_parameters()``
    eagerly to check finiteness and symbolic assumptions.
    """

    expression: object
    parameters: tuple
    strip: tuple
    conditions: object
    _evaluate: Callable[..., jax.typing.ArrayLike] = field(repr=False)
    _bounds: Callable[..., tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]] = field(
        repr=False
    )
    _condition: Callable[..., jax.typing.ArrayLike] = field(repr=False)
    _assumptions: tuple = field(repr=False)

    def __call__(self, *args, **kwargs) -> GeneratedKernel:
        if len(args) > len(self.parameters):
            raise TypeError("too many positional parameters")
        bound = dict(zip((p.name for p in self.parameters), args))
        for name, value in kwargs.items():
            if name not in {p.name for p in self.parameters}:
                raise TypeError(f"unexpected parameter: {name}")
            if name in bound:
                raise TypeError(f"multiple values for parameter: {name}")
            bound[name] = value
        values = []
        for p in self.parameters:
            if p.name not in bound:
                raise TypeError(f"missing parameter: {p.name}")
            try:
                value = jnp.asarray(bound[p.name])
            except (TypeError, ValueError) as error:
                raise TypeError(f"{p.name} must be a numeric scalar") from error
            if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.number):
                raise TypeError(f"{p.name} must be a numeric scalar")
            values.append(value)
        return GeneratedKernel(
            tuple(values),
            self._evaluate,
            self._bounds,
            self._condition,
            self._assumptions,
        )


def from_mellin(
    expression, s, *, parameters=(), strip, conditions=True, cse=True, functions=None
) -> KernelFactory:
    """Compile a Mellin formula and its required open convergence ``strip``.

    ``functions`` maps additional symbolic function names to JAX callables.
    Only supported JAX operations are generated; unresolved expressions fail
    at creation. Gamma products/ratios with integer powers use log space.
    """
    _declarations(s, parameters)
    expression = sp.sympify(expression)
    conditions = sp.sympify(conditions)
    if not isinstance(conditions, sp.logic.boolalg.Boolean):
        raise TypeError("conditions must be a symbolic Boolean expression")
    if not isinstance(strip, (tuple, list, sp.Tuple)) or len(strip) != 2:
        raise ValueError("strip must contain two endpoints")
    strip = tuple(sp.sympify(endpoint) for endpoint in strip)
    if any(
        endpoint.is_real is False and endpoint not in (-sp.oo, sp.oo)
        for endpoint in strip
    ):
        raise ValueError("strip endpoints must be real")
    mappings = dict(_FUNCTIONS)
    if functions is not None:
        if any(
            not isinstance(name, str) or not callable(fn)
            for name, fn in functions.items()
        ):
            raise TypeError("functions must map names to JAX callables")
        mappings.update(functions)
    _check(expression, (s, *parameters))
    evaluate = _compile(_log_products(expression), (s, *parameters), mappings, cse)
    bounds = _compile(strip, parameters, mappings, cse)
    condition = _compile(conditions, (s, *parameters), mappings, cse)
    assumptions = tuple(_assumptions(p) for p in parameters)
    return KernelFactory(
        expression,
        parameters,
        strip,
        conditions,
        evaluate,
        bounds,
        condition,
        assumptions,
    )


def from_expression(
    expression, x, s, *, parameters=(), strip=None, cse=True, functions=None
) -> KernelFactory:
    """Compute a SymPy Mellin transform, retaining its strip and conditions.

    An explicit ``strip`` overrides the inferred bounds but retains auxiliary
    conditions. Unsuccessful or unsupported transforms raise ``ValueError``.
    Differentiate real-space expressions with ``sympy.diff`` before generation.
    """
    _declarations(s, parameters, x)
    expression = sp.sympify(expression)
    _check(expression, (x, *parameters))
    try:
        result = sp.mellin_transform(expression, x, s)
    except IntegralTransformError as error:
        raise ValueError(
            "SymPy could not resolve the Mellin transform; use from_mellin"
        ) from error
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(
            "SymPy could not resolve the Mellin transform; use from_mellin"
        )
    formula, inferred, conditions = result
    return from_mellin(
        formula,
        s,
        parameters=parameters,
        strip=inferred if strip is None else strip,
        conditions=conditions,
        cse=cse,
        functions=functions,
    )
