"""Tests for the opt-in jaxtyping runtime checks."""

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
RUNTIME_CHECKS_ENV = "FFTLOGGIN_RUNTIME_TYPECHECK"


def _run_python(code: str, *, runtime_checking: str | None) -> None:
    env = os.environ.copy()
    env.pop(RUNTIME_CHECKS_ENV, None)
    if runtime_checking is not None:
        env[RUNTIME_CHECKS_ENV] = runtime_checking
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_checks_are_opt_in():
    _run_python(
        "import sys; import fftloggin; assert 'beartype' not in sys.modules",
        runtime_checking=None,
    )
    _run_python(
        "import sys; import fftloggin; assert 'beartype' not in sys.modules",
        runtime_checking="0",
    )


def test_runtime_checks_validate_shapes_and_compose_with_jax_transforms():
    _run_python(
        """
import jax
import jax.numpy as jnp
from jaxtyping import TypeCheckError
from fftloggin import BesselJKernel, forward

samples = jnp.linspace(0.2, 1.0, 8)
kernel = BesselJKernel(0.5)

try:
    forward(samples, kernel, dlog=jnp.array([0.1, 0.2]))
except TypeCheckError as error:
    assert "dlog" in str(error)
else:
    raise AssertionError("vector dlog was not rejected by runtime checking")

transform = lambda dlog: forward(samples, kernel, dlog=dlog)
batched = jax.jit(jax.vmap(transform))(jnp.array([0.1, 0.2]))
assert batched.shape == (2, samples.shape[0])
gradient = jax.grad(lambda dlog: jnp.sum(transform(dlog)))(jnp.array(0.1))
assert bool(jnp.isfinite(gradient))
""",
        runtime_checking="1",
    )


def test_runtime_checks_accept_integer_kernels_and_derivative_domains():
    _run_python(
        """
from fftloggin import BesselJKernel, Coordinate, diff
t = Coordinate("t")

kernel = diff(BesselJKernel(0)(t), t)
lower, upper = kernel.domain
assert float(lower) == 1.0
assert float(upper) == 2.5

shifted = t**1 * BesselJKernel(0)(t)
shift_lower, shift_upper = shifted.domain
assert float(shift_lower) == -1.0
assert float(shift_upper) == 0.5
""",
        runtime_checking="1",
    )


def test_runtime_checks_support_symbolic_jax_composition():
    _run_python(
        """
import jax
import jax.numpy as jnp
from fftloggin import BesselJKernel, Coordinate, diff, forward, plan
t = Coordinate("t")
def make(mu, weight, power, scale):
    expr = weight * t**power * BesselJKernel(mu)(scale * t)
    return expr + diff(expr, t)
expr = make(0.5, 2.0, 0.2, 1.7)
s = jnp.array([1.0 + 0.3j, 1.2 + 0.4j])
assert bool(jnp.allclose(jax.jit(lambda e, z: e(z))(expr, s), expr(s)))
evaluate = lambda a: jnp.real(make(a, a, a, a)(s)).sum()
assert bool(jnp.isfinite(jax.jit(jax.grad(evaluate))(0.5)))
assert jax.jit(jax.vmap(evaluate))(jnp.array([0.5, 0.6])).shape == (2,)
p = plan(expr, 16, dlog=0.1, bias=0.0)
assert forward(jnp.ones(16), p).shape == (16,)
""",
        runtime_checking="1",
    )


def test_import_preserves_process_jax_configuration():
    _run_python(
        """
import importlib
import jax
for enabled in (False, True):
    with jax.enable_x64(enabled):
        before = dict(jax.config.values)
        import fftloggin
        importlib.reload(fftloggin)
        assert jax.config.values == before
""",
        runtime_checking=None,
    )
