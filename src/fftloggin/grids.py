"""Functions for paired logarithmic FFTLog coordinate arrays."""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

__all__ = ("get_array_center", "get_paired_grids", "infer_dlog", "infer_log_kr")


def infer_dlog(x: Float[ArrayLike, "n"], *, rtol: float = 1e-5) -> Float[Array, ""]:
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


def get_array_center(x: Float[ArrayLike, "n"]) -> Float[Array, ""]:
    """Geometric center of a one-dimensional logarithmic grid."""
    x = jnp.asarray(x)
    return jnp.exp(0.5 * (jnp.log(x[0]) + jnp.log(x[-1])))


def get_paired_grids(
    *,
    r: Float[ArrayLike, "n"] | None = None,
    k: Float[ArrayLike, "n"] | None = None,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """Return paired ``(r, k)`` grids from exactly one supplied grid.

    The missing grid is ``exp(log_kr) / x[::-1]``. This pure operation works
    inside ``jit`` and ``vmap``.
    """
    if (r is None) == (k is None):
        raise ValueError("provide exactly one of r or k")

    if r is not None:
        r = jnp.asarray(r)
        k = jnp.exp(log_kr - jnp.log(r[::-1]))
    else:
        k = jnp.asarray(k)
        r = jnp.exp(log_kr - jnp.log(k[::-1]))
    return r, k


def infer_log_kr(
    x: Float[ArrayLike, "n"],
    *,
    ycenter: Float[ArrayLike, ""] | None = None,
    ymax: Float[ArrayLike, ""] | None = None,
    ymin: Float[ArrayLike, ""] | None = None,
) -> Float[Array, ""]:
    """Infer ``log(k_center * r_center)`` from one paired-grid anchor.

    Exactly one of ``ycenter``, ``ymax`` or ``ymin`` is required. The choice
    is a Python-level structural argument; the selected calculation is pure.
    """
    selected = sum(value is not None for value in (ycenter, ymax, ymin))
    if selected != 1:
        raise ValueError("provide exactly one of ycenter, ymax or ymin")
    x = jnp.asarray(x)
    if ycenter is not None:
        return jnp.log(ycenter) + 0.5 * (jnp.log(x[0]) + jnp.log(x[-1]))
    if ymax is not None:
        return jnp.log(ymax) + jnp.log(x[0])
    assert ymin is not None
    return jnp.log(ymin) + jnp.log(x[-1])
