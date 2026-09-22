"""Functions for paired logarithmic FFTLog coordinate arrays."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ("infer_dlog", "get_array_center", "get_other_array", "infer_log_kr")


def infer_dlog(x: jax.Array, *, rtol: float = 1e-5) -> jax.Array:
    """Infer and eagerly validate the logarithmic spacing of a 1-D grid.

    This host-side convenience function is not intended for ``jax.jit``.
    """
    x = jnp.asarray(x)
    if x.ndim != 1 or x.shape[0] < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    if not bool(jnp.all(jnp.isfinite(x) & (x > 0))):
        raise ValueError("grid values must be finite and positive")
    logx = jnp.log(x)
    dlog = (logx[-1] - logx[0]) / (x.shape[0] - 1)
    if not bool(jnp.allclose(jnp.diff(logx), dlog, rtol=rtol)):
        raise ValueError("x must be uniformly spaced in the logarithm")
    return dlog


def get_array_center(x: jax.Array) -> jax.Array:
    """Geometric center of a one-dimensional logarithmic grid."""
    x = jnp.asarray(x)
    return jnp.sqrt(x[0] * x[-1])


def get_other_array(x: jax.Array, log_kr: jax.Array) -> jax.Array:
    """Return the paired grid, ``exp(log_kr) / x[::-1]``.

    This pure operation works inside ``jit`` and ``vmap``.
    """
    x = jnp.asarray(x)
    return jnp.exp(log_kr) / x[::-1]


def infer_log_kr(
    x: jax.Array,
    *,
    ycenter: jax.Array | None = None,
    ymax: jax.Array | None = None,
    ymin: jax.Array | None = None,
) -> jax.Array:
    """Infer ``log(k_center * r_center)`` from one paired-grid anchor.

    Exactly one of ``ycenter``, ``ymax`` or ``ymin`` is required. The choice
    is a Python-level structural argument; the selected calculation is pure.
    """
    selected = sum(value is not None for value in (ycenter, ymax, ymin))
    if selected != 1:
        raise ValueError("provide exactly one of ycenter, ymax or ymin")
    x = jnp.asarray(x)
    if ycenter is not None:
        return jnp.log(ycenter * get_array_center(x))
    if ymax is not None:
        return jnp.log(ymax * x[0])
    return jnp.log(ymin * x[-1])
