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
