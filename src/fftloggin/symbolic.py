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
    from sympy.integrals.transforms import IntegralTransformError
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
    "conjugate": jnp.conj,
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
    # Explicit user function names prevent undefined functions from invoking
    # attached SymPy implementations; numerical operations come from SymPy's
    # JAX backend.
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


@dataclass(frozen=True)
class KernelFactory:
    """Reusable factory with inspectable symbolic formula and convergence data.

    Bind every declared parameter positionally or by its symbol name. Binding
    checks scalar numeric shape/dtype; call ``kernel.validate_parameters()``
    eagerly to check finiteness. Callers are responsible for symbolic assumptions.
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
    parameter_names: tuple[str, ...] = field(repr=False)

    def _bind(self, args, kwargs) -> GeneratedKernel:
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
            self.parameter_names,
        )

    def __call__(self, *args, **kwargs) -> GeneratedKernel:
        return self._bind(args, kwargs)

    def check_jax(self, s: jax.typing.ArrayLike, /, *args, **kwargs) -> None:
        """Eagerly exercise JIT, batching, and gradients for supplied values."""
        kernel = self._bind(args, kwargs)
        sample = jnp.asarray(s)
        batch = sample[None] if sample.ndim == 0 else sample

        def stage(name, operation):
            try:
                result = operation()
                jax.tree.map(
                    lambda value: (
                        value.block_until_ready()
                        if callable(getattr(value, "block_until_ready", None))
                        else value
                    ),
                    result,
                )
            except Exception as error:
                raise ValueError(f"JAX compatibility check failed at {name}") from error

        stage("jit", lambda: jax.jit(lambda k, z: k(z))(kernel, sample))
        stage(
            "vmap",
            lambda: jax.jit(jax.vmap(lambda z, k: k(z), in_axes=(0, None)))(
                batch, kernel
            ),
        )

        def loss(values, z):
            candidate = GeneratedKernel(
                values,
                kernel.evaluate,
                kernel.bounds,
                kernel.condition,
                kernel.parameter_names,
            )
            return jnp.real(jnp.sum(candidate(z)))

        stage(
            "grad",
            lambda: jax.jit(jax.grad(loss, argnums=(0, 1), allow_int=True))(
                kernel.values, sample
            ),
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
    return KernelFactory(
        expression,
        parameters,
        strip,
        conditions,
        evaluate,
        bounds,
        condition,
        tuple(p.name for p in parameters),
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
