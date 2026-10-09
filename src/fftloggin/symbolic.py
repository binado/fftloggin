"""Eager SymPy generation of numerical JAX Mellin kernel factories.

Install ``fftloggin[symbolic]`` to use this module. Create factories outside
JAX transformations; binding their scalar parameters is traceable.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from inspect import getsource
from typing import Any, TypeAlias, cast

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

_SympyInput: TypeAlias = sp.Expr | int | float | complex
_StripInput: TypeAlias = tuple[_SympyInput, _SympyInput] | list[_SympyInput] | sp.Tuple
_FunctionMap: TypeAlias = Mapping[str, Callable[..., jax.typing.ArrayLike]]


def _gamma(z: jax.typing.ArrayLike) -> jax.Array:
    return jnp.exp(_loggamma(jnp.asarray(z) + 0j))


def _complex_loggamma(z: jax.typing.ArrayLike) -> jax.Array:
    return _loggamma(jnp.asarray(z) + 0j)


_FUNCTIONS: dict[str, Callable[..., jax.typing.ArrayLike]] = {
    "gamma": _gamma,
    "loggamma": _complex_loggamma,
    "conjugate": jnp.conj,
}


def _log_products(expression: sp.Basic) -> sp.Basic:
    """Combine integer gamma powers in each product without moving branches.

    Other factors stay outside exp: in particular, log(gamma(z)), fractional
    gamma powers and arbitrary complex powers are never expanded.
    """
    if not expression.args:
        return expression
    if isinstance(expression, (sp.Mul, sp.Pow)) or expression.func == sp.gamma:
        factors = sp.Mul.make_args(cast(sp.Expr, expression))
        logs, rest = [], []
        for factor in factors:
            base, exponent = factor.as_base_exp()
            if base.func == sp.gamma and isinstance(exponent, sp.Integer):
                logs.append(exponent * sp.loggamma(base.args[0]))
            else:
                rest.append(factor)
        if logs:
            return sp.Mul(*(cast(sp.Expr, _log_products(a)) for a in rest)) * sp.exp(
                sp.Add(*logs)
            )
    return expression.func(*(_log_products(a) for a in expression.args))


def _declarations(
    s: sp.Symbol, parameters: tuple[sp.Symbol, ...], x: sp.Symbol | None = None
) -> None:
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


def _check(expression: sp.Basic, allowed: tuple[sp.Symbol, ...]) -> None:
    if expression.free_symbols - set(allowed):
        raise ValueError(
            f"undeclared symbols: {expression.free_symbols - set(allowed)}"
        )


class _Printer(JaxPrinter):
    """Broadcast scalar and array operands instead of stacking them."""

    def _fold(self, expression: sp.Basic, function: str) -> str:
        function = self._module_format(function)
        parts = [self._print(arg) for arg in expression.args]
        result = parts[0]
        for part in parts[1:]:
            result = f"{function}({result}, {part})"
        return result

    def _print_And(self, expr: sp.Basic) -> str:
        return self._fold(expr, "jax.numpy.logical_and")

    def _print_Or(self, expr: sp.Basic) -> str:
        return self._fold(expr, "jax.numpy.logical_or")


def _compile(
    expression: sp.Basic | tuple[sp.Basic, ...],
    arguments: tuple[sp.Symbol, ...],
    functions: _FunctionMap,
    cse: bool,
) -> Callable[..., Any]:
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

    expression: sp.Basic
    parameters: tuple[sp.Symbol, ...]
    strip: tuple[sp.Basic, sp.Basic]
    conditions: sp.logic.boolalg.Boolean
    _evaluate: Callable[..., jax.typing.ArrayLike] = field(repr=False)
    _bounds: Callable[..., tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]] = field(
        repr=False
    )
    _condition: Callable[..., jax.typing.ArrayLike] = field(repr=False)
    parameter_names: tuple[str, ...] = field(repr=False)

    def _bind(
        self,
        *args: jax.typing.ArrayLike,
        **kwargs: jax.typing.ArrayLike,
    ) -> GeneratedKernel:
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

    def __call__(
        self,
        *args: jax.typing.ArrayLike,
        **kwargs: jax.typing.ArrayLike,
    ) -> GeneratedKernel:
        return self._bind(*args, **kwargs)

    def source(self) -> str:
        """Return the generated Python source for the Mellin evaluator."""
        return getsource(self._evaluate)

    def check_jax(
        self,
        s: jax.typing.ArrayLike,
        /,
        *args: jax.typing.ArrayLike,
        **kwargs: jax.typing.ArrayLike,
    ) -> None:
        """Eagerly exercise JIT, batching, and gradients for supplied values."""
        kernel = self._bind(*args, **kwargs)
        sample = jnp.asarray(s)
        batch = sample[None] if sample.ndim == 0 else sample

        def stage(name: str, operation: Callable[[], Any]) -> None:
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

        def loss(
            values: tuple[jax.typing.ArrayLike, ...], z: jax.typing.ArrayLike
        ) -> jax.Array:
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
    expression: _SympyInput,
    s: sp.Symbol,
    *,
    parameters: tuple[sp.Symbol, ...] = (),
    strip: _StripInput,
    conditions: bool | sp.logic.boolalg.Boolean = True,
    cse: bool = True,
    functions: _FunctionMap | None = None,
) -> KernelFactory:
    """Compile a Mellin formula and its required open convergence ``strip``.

    ``functions`` maps additional symbolic function names to JAX callables.
    Only supported JAX operations are generated; unresolved expressions fail
    at creation. Gamma products/ratios with integer powers use log space.
    """
    _declarations(s, parameters)
    symbolic_conditions = sp.sympify(conditions)
    if not isinstance(symbolic_conditions, sp.logic.boolalg.Boolean):
        raise TypeError("conditions must be a symbolic Boolean expression")
    if not isinstance(strip, (tuple, list, sp.Tuple)) or len(strip) != 2:
        raise ValueError("strip must contain two endpoints")
    strip = cast(
        tuple[sp.Expr, sp.Expr],
        tuple(sp.sympify(endpoint) for endpoint in strip),
    )
    if any(
        endpoint.is_real is False and endpoint not in (-sp.oo, sp.oo)
        for endpoint in strip
    ):
        raise ValueError("strip endpoints must be real")
    symbolic_expression = sp.sympify(expression)
    _check(symbolic_expression, (s, *parameters))
    mappings = dict(_FUNCTIONS)
    if functions is not None:
        if any(
            not isinstance(name, str) or not callable(fn)
            for name, fn in functions.items()
        ):
            raise TypeError("functions must map names to JAX callables")
        mappings.update(functions)
    evaluate = _compile(
        _log_products(symbolic_expression), (s, *parameters), mappings, cse
    )
    bounds = _compile(strip, parameters, mappings, cse)
    condition = _compile(symbolic_conditions, (s, *parameters), mappings, cse)
    return KernelFactory(
        symbolic_expression,
        parameters,
        strip,
        symbolic_conditions,
        evaluate,
        bounds,
        condition,
        tuple(p.name for p in parameters),
    )


def from_expression(
    expression: _SympyInput,
    x: sp.Symbol,
    s: sp.Symbol,
    *,
    parameters: tuple[sp.Symbol, ...] = (),
    strip: _StripInput | None = None,
    cse: bool = True,
    functions: _FunctionMap | None = None,
) -> KernelFactory:
    """Compute a SymPy Mellin transform, retaining its strip and conditions.

    An explicit ``strip`` overrides the inferred bounds but retains auxiliary
    conditions. Unsuccessful or unsupported transforms raise ``ValueError``.
    Differentiate real-space expressions with ``sympy.diff`` before generation.
    """
    _declarations(s, parameters, x)
    symbolic_expression = sp.sympify(expression)
    _check(symbolic_expression, (x, *parameters))
    try:
        result = sp.mellin_transform(symbolic_expression, x, s)
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
