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
import sympy as sp
from fftloggin import BesselJKernel
from fftloggin.symbolic import from_mellin
s = sp.Symbol("s")
m = lambda z: 2**(z-1)*sp.gamma(z/2)/sp.gamma((2-z)/2)
kernel = from_mellin(-(s-1)*m(s-1), s, strip=(1, 2.5))()
lower, upper = kernel.domain
assert float(lower) == 1.0
assert float(upper) == 2.5
shifted = from_mellin(m(s+1), s, strip=(-1, 0.5))()
assert tuple(map(float, shifted.domain)) == (-1.0, 0.5)
assert BesselJKernel(0)(1.0).shape == ()
""",
        runtime_checking="1",
    )


def test_runtime_checks_support_symbolic_jax_composition():
    _run_python(
        """
import jax
import jax.numpy as jnp
import sympy as sp
from fftloggin import forward, plan
from fftloggin.symbolic import from_mellin
s, mu, weight, power = sp.symbols("s mu weight power")
scale = sp.Symbol("scale", positive=True)
def m(z):
    return 2**(z-1)*sp.gamma((mu+z)/2)/sp.gamma((mu+2-z)/2)
def weighted(z):
    return weight*scale**(-(z+power))*m(z+power)
factory = from_mellin(weighted(s)-(s-1)*weighted(s-1), s,
    parameters=(mu, weight, power, scale), strip=(1-mu-power, 1.5-power))
make = lambda mu, weight, power, scale: factory(mu, weight, power, scale)
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
        import fftloggin.symbolic
        importlib.reload(fftloggin.symbolic)
        assert jax.config.values == before
""",
        runtime_checking=None,
    )


def test_core_works_without_sympy():
    _run_python(
        """
import sys
import importlib.abc
class BlockSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sympy" or fullname.startswith("sympy."):
            raise ImportError("SymPy unavailable")
sys.meta_path.insert(0, BlockSympy())
import jax
import jax.numpy as jnp
import fftloggin as f
assert "sympy" not in sys.modules
assert "fftloggin.symbolic" not in sys.modules
kernel = f.BesselJKernel(0)
f.validate_parameters(kernel, dlog=0.1)
p = f.plan(kernel, 16, dlog=0.1)
assert jax.jit(f.forward)(jnp.ones(16), p).shape == (16,)
class Custom(f.Kernel):
    def mellin(self, s):
        return jnp.ones_like(jnp.asarray(s))
f.validate_parameters(Custom(), dlog=0.1)
assert f.forward(jnp.ones(16), Custom(), dlog=0.1).shape == (16,)
try:
    import fftloggin.symbolic
except ImportError as error:
    assert "fftloggin[symbolic]" in str(error)
else:
    raise AssertionError("missing optional dependency was not reported")
""",
        runtime_checking=None,
    )
