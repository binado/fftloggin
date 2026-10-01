"""Paired logarithmic FFTLog grids and the treatment of their edges."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, ArrayLike, Float, Real

__all__ = (
    "Padding",
    "get_array_center",
    "get_paired_grids",
    "infer_dlog",
    "infer_log_kr",
    "taper",
)


def infer_dlog(x: Float[ArrayLike, "n"], *, rtol: float = 1e-5) -> Float[Array, ""]:
    """Infer and eagerly validate the logarithmic spacing of a 1-D grid.

    This host-side convenience function is not intended for ``jax.jit``.

    Parameters
    ----------
    x : array_like
        Positive, finite, one-dimensional grid with at least two points.
    rtol : float, optional
        Relative tolerance for checking uniform spacing in log space.
        Defaults to ``1e-5``. An absolute allowance for the rounding of
        ``log(x)`` at the precision of ``x`` is added, so float32 grids
        such as ``jnp.logspace`` output pass.

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
    # Rounding of log(x) grows with |log(x)|, not with the spacing.
    atol = 64 * jnp.finfo(logx.dtype).eps * jnp.max(jnp.abs(logx))
    if not bool(jnp.allclose(jnp.diff(logx), dlog, rtol=rtol, atol=atol)):
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


@partial(register_dataclass, data_fields=(), meta_fields=("width",))
@dataclass(frozen=True)
class Padding:
    """Symmetric padding of logarithmic grids and their samples.

    FFTLog treats its input as periodic, so the samples at one end of the
    array leak into the output at the other end. Padding moves the periodic
    images apart: pad the samples, transform them with a plan built for the
    padded length, and crop the result back to the original grid.

    Parameters
    ----------
    width : int
        Number of points added at each end. It is static under JAX
        transformations because it sets array shapes.

    Raises
    ------
    TypeError
        If ``width`` is not an integer.
    ValueError
        If ``width`` is negative.

    Notes
    -----
    The padding is the same on both sides, so the padded grid keeps the
    geometric centre and the spacing of the original one. ``dlog``, ``bias``
    and ``log_kr``, including a value from ``lowring_log_kr``, carry over
    unchanged, and cropping the paired output grid gives back
    ``get_paired_grids(r=x, log_kr=log_kr)``.

    Padding suppresses the wrap-around between the two ends of the array,
    which dominates the error near the ends of the output. It does not remove
    ringing from a step or a kink inside the input: zeros next to a step are
    still a step. For inputs that vanish beyond the grid, the ``width``
    output points dropped at each end by ``crop`` are valid transform values
    on the extended output grid.

    Map over a stack of arrays with ``axis`` or ``jax.vmap(padding)``, and
    over other collections with ``jax.tree.map(padding, ...)``.

    Examples
    --------
    Zero-pad a window that vanishes beyond the grid::

        padding = Padding(n // 2)
        padded = padding(window)
        p = plan(kernel, padded.shape[0], dlog=dlog, bias=bias, log_kr=log_kr)
        result = padding.crop(forward(padded, p))  # on get_paired_grids(r=chi)

    Evaluate a window that does not vanish at the grid ends on the extended
    grid instead of padding it with zeros::

        chi_ext = padding.grid(chi)
        p = plan(kernel, chi_ext.size, dlog=dlog, bias=bias, log_kr=log_kr)
        result = padding.crop(forward(window(chi_ext), p))
    """

    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise TypeError("width must be an integer")
        if self.width < 0:
            raise ValueError("width must be non-negative")

    def __call__(
        self, a: Float[ArrayLike, "..."], *, axis: int = 0
    ) -> Float[Array, "..."]:
        """Pad ``a`` with ``width`` zeros at each end of ``axis``.

        Parameters
        ----------
        a : array_like
            Samples to pad.
        axis : int, optional
            Sample axis. Defaults to the first axis.

        Returns
        -------
        array
            Padded samples, ``2 * width`` longer along ``axis``.
        """
        a = jnp.asarray(a)
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (self.width, self.width)
        return jnp.pad(a, pad_width)

    def crop(self, a: Float[ArrayLike, "..."], *, axis: int = 0) -> Float[Array, "..."]:
        """Drop ``width`` points from each end of ``axis``.

        Parameters
        ----------
        a : array_like
            Padded samples or transform output, such as the result of
            ``forward`` with a plan built for the padded length.
        axis : int, optional
            Sample axis. Defaults to the first axis, which also holds the
            samples of a product-plan output.

        Returns
        -------
        array
            Cropped array, ``2 * width`` shorter along ``axis``.
        """
        a = jnp.asarray(a)
        return jax.lax.slice_in_dim(
            a, self.width, a.shape[axis] - self.width, axis=axis
        )

    def grid(self, x: Float[ArrayLike, "n"]) -> Float[Array, "m"]:
        """Extend a logarithmic grid by ``width`` points at each end.

        The spacing is taken from the endpoints of ``x`` without checking
        that the grid is uniform in its logarithm, so this works under
        ``jit``. Check concrete grids with ``infer_dlog`` first.

        Parameters
        ----------
        x : array_like
            Positive grid, uniformly spaced in its logarithm, with at least
            two points.

        Returns
        -------
        array
            Grid with ``2 * width`` more points, the same logarithmic spacing
            and the same geometric centre. Its middle points equal ``x``.
        """
        x = jnp.asarray(x)
        n = x.shape[0]
        log_first = jnp.log(x[0])
        dlog = (jnp.log(x[-1]) - log_first) / (n - 1)
        index = jnp.arange(-self.width, n + self.width, dtype=dlog.dtype)
        extended = jnp.exp(log_first + dlog * index)
        # Keep the original samples exact, without exp(log(x)) rounding.
        return jax.lax.dynamic_update_slice_in_dim(extended, x, self.width, axis=0)


def _taper_ramp(
    t: Float[Array, "n"], shape: Literal["cosine", "planck"]
) -> Float[Array, "n"]:
    """Rise from 0 at ``t <= 0`` to 1 at ``t >= 1``."""
    t = jnp.clip(t, 0.0, 1.0)
    if shape == "cosine":
        return jnp.sin(0.5 * jnp.pi * t) ** 2
    # Evaluate the Planck ramp only inside (0, 1) so gradients stay finite at
    # the ends, where 1/t and 1/(1 - t) diverge.
    inside = (t > 0) & (t < 1)
    t_inside = jnp.where(inside, t, 0.5)
    ramp = jax.nn.sigmoid(1 / (1 - t_inside) - 1 / t_inside)
    return jnp.where(inside, ramp, jnp.where(t >= 1, 1.0, 0.0))


def taper(
    x: Float[ArrayLike, "n"],
    width: Real[ArrayLike, ""],
    *,
    lo: Real[ArrayLike, ""] | None = None,
    hi: Real[ArrayLike, ""] | None = None,
    side: Literal["lo", "hi", "both"] = "both",
    shape: Literal["cosine", "planck"] = "cosine",
) -> Float[Array, "n"]:
    """Weights that take samples smoothly to zero at the edges of their support.

    Multiply samples on ``x`` by these weights to remove a step or a kink at an
    edge of their support, which FFTLog would otherwise turn into ringing.

    Parameters
    ----------
    x : array_like
        Positive grid of the samples. It need not be uniform in its logarithm.
    width : scalar
        Length of each ramp in ``log(x)``, assumed positive. A width in
        ``log(x)`` keeps the smoothing scale independent of the sample count.
    lo, hi : scalar, optional
        Edges of the support, in the units of ``x``. They default to ``x[0]``
        and ``x[-1]``. Give them explicitly when the support ends inside the
        grid.
    side : {"lo", "hi", "both"}, optional
        Edges to taper. Defaults to both.
    shape : {"cosine", "planck"}, optional
        Ramp shape in ``t``, the distance from the edge in units of ``width``:
        ``"cosine"`` is ``sin(pi * t / 2)**2``, continuous with its first
        derivative; ``"planck"`` is the Planck-taper ramp
        ``1 / (1 + exp(1/t - 1/(1 - t)))``, smooth to all orders.
        Defaults to ``"cosine"``.

    Returns
    -------
    array
        Weights with the shape of ``x``: 0 at and beyond a tapered edge, rising
        to 1 at a distance ``width`` in ``log(x)`` inside it. With ``"both"``
        they are the product of the two ramps.

    Raises
    ------
    ValueError
        If ``side`` or ``shape`` is unknown, or if ``lo`` or ``hi`` is given
        for an edge that ``side`` does not taper.

    Notes
    -----
    Tapering changes the input, so it trades ringing for a bias in the
    transform. Keep ``width`` small compared with the scale over which the
    samples vary.

    The weights are a function of ``x`` and the edges only, so they commute
    with ``Padding``. With the default edges, taper on the grid whose ends are
    the edges of the support: the unpadded grid for zero padding. On an
    extended grid, the ends lie beyond the support, so pass the edges
    explicitly.

    Examples
    --------
    Smooth the upper edge of a number-count bin, then zero-pad it::

        w = window(chi) * taper(chi, 0.05, hi=chi_max, side="hi")
        result = padding.crop(forward(padding(w), p))

    Evaluate a lensing window, which does not vanish at small ``chi``, on the
    extended grid and smooth only its kink at ``chi_star``::

        chi_ext = padding.grid(chi)
        w = window(chi_ext) * taper(chi_ext, 0.05, hi=chi_star, side="hi")
        result = padding.crop(forward(w, p))
    """
    if side not in ("lo", "hi", "both"):
        raise ValueError(f"side must be 'lo', 'hi' or 'both', got {side!r}")
    if shape not in ("cosine", "planck"):
        raise ValueError(f"shape must be 'cosine' or 'planck', got {shape!r}")
    if lo is not None and side == "hi":
        raise ValueError("lo is given but side='hi' does not taper the lower edge")
    if hi is not None and side == "lo":
        raise ValueError("hi is given but side='lo' does not taper the upper edge")
    logx = jnp.log(jnp.asarray(x))
    weights = jnp.ones_like(logx)
    if side in ("lo", "both"):
        log_lo = logx[0] if lo is None else jnp.log(lo)
        weights = weights * _taper_ramp((logx - log_lo) / width, shape)
    if side in ("hi", "both"):
        log_hi = logx[-1] if hi is None else jnp.log(hi)
        weights = weights * _taper_ramp((log_hi - logx) / width, shape)
    return weights
