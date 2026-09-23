"""Functions for paired logarithmic FFTLog coordinate arrays."""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

__all__ = ("get_array_center", "get_paired_grids", "infer_dlog", "infer_log_kr")


def infer_dlog(x: Float[ArrayLike, "n"], *, rtol: float = 1e-5) -> Float[Array, ""]:
    """Infer and eagerly validate the logarithmic spacing of a 1-D grid.

    This host-side convenience function is not intended for ``jax.jit``.

    Parameters
    ----------
    x : array_like
        Positive, finite, one-dimensional grid with at least two points.
    rtol : float, optional
        Relative tolerance for checking uniform spacing in log space.
        Defaults to ``1e-5``.

    Returns
    -------
    scalar
        Mean step between adjacent log-grid coordinates.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional, has fewer than two points, contains
        non-positive or non-finite values, or is not uniformly spaced in its
        logarithm.
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
    """Return the geometric center of a one-dimensional grid.

    The center is computed from the first and last values, which is the grid
    center for a uniformly logarithmic array.

    Parameters
    ----------
    x : array_like
        Grid values ordered from the first endpoint to the last.

    Returns
    -------
    scalar
        Geometric mean of the endpoint values.
    """
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

    Parameters
    ----------
    r, k : array_like, optional
        Exactly one input grid. The supplied grid is returned unchanged as its
        corresponding element in the output pair.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of ``r`` and ``k``.
        Defaults to zero.

    Returns
    -------
    tuple of arrays
        The paired ``(r, k)`` grids, in that order.

    Raises
    ------
    ValueError
        If neither grid or both grids are supplied.
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

    Parameters
    ----------
    x : array_like
        One grid from a paired ``r``/``k`` pair.
    ycenter : scalar, optional
        Value on the other grid at its geometric center.
    ymax : scalar, optional
        Maximum value on the other grid, paired with ``x[0]``.
    ymin : scalar, optional
        Minimum value on the other grid, paired with ``x[-1]``.

    Returns
    -------
    scalar
        Logarithm of the product of the two grid centers.

    Raises
    ------
    ValueError
        If zero or more than one anchor value is supplied.
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
