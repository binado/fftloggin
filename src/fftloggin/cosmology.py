"""Double spherical Bessel FFTLog transforms for unequal-time kernels.

The transform ``chi * integral(a(k) * j_ell(k*chi) * j_ell(k*t*chi), k)`` is
evaluated for all ``chi`` of the output grid and for ratios ``t`` on the same
logarithmic spacing, by passing the plans built here to ``forward``. The plan
multiplies the discrete kernels of two single-kernel FFTLog transforms, see
``fftloggin.product_plan``.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from .fftlog import Plan, plan, product_plan
from .kernels import SphericalBesselJKernel

__all__ = ("double_spherical_bessel_plan",)


def double_spherical_bessel_plan(
    ell: Real[ArrayLike, ""],
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
    max_offset: int,
) -> Plan:
    """Plan the FFTLog transform with the kernel pair ``j_ell(x) * j_ell(t*x)``.

    ``forward(a, plan)`` returns an array ``K`` of shape
    ``(n, 2 * max_offset + 1)`` whose entry ``[i, j]`` is
    ``chi_i * integral(a(k) * j_ell(k*chi_i) * j_ell(k*t_j*chi_i), k)``
    with ``t_j = exp((j - max_offset) * dlog)``. Since the ratios share the
    grid spacing, ``t_j * chi_i = chi_(i + j - max_offset)`` and ``K`` is a
    band of the matrix ``K(chi, chi') = chi * I(chi, chi')`` with
    ``I(chi, chi') = integral(a(k) * j_ell(k*chi) * j_ell(k*chi'), k)``.
    Only the first coordinate multiplies the integral, so ``K`` is not
    symmetric: ``I`` is, which gives ``K(chi, chi') = (chi / chi') *
    K(chi', chi)``.

    Parameters
    ----------
    ell : scalar
        Order of the spherical Bessel functions.
    n : int
        Number of samples of the transformed array.
    dlog : scalar
        Positive logarithmic spacing of the grids.
    bias : scalar, optional
        Power-law bias seen by the input ``a(k)``. ``1 + bias`` must lie in
        ``(-2*ell, 2)``. Defaults to zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.
    max_offset : int
        Number of ratios ``t`` on each side of ``t = 1``. The plan covers
        ``|log(t)| <= max_offset * dlog`` in ``2 * max_offset + 1`` columns,
        so the transform holds ``K(chi, chi')`` for ``chi'`` within
        ``max_offset`` grid points of ``chi``. It sets which pairs are
        available, not the accuracy of each value. The kernel falls off like
        ``t**ell`` away from ``t = 1``, so higher orders need a narrower
        band. Cost grows linearly with it.

    Returns
    -------
    Plan
        Plan with coefficients of shape ``(n // 2 + 1, 2 * max_offset + 1)``.

    Notes
    -----
    The plan is ``product_plan`` of two ``SphericalBesselJKernel(ell)``
    plans, with biases ``q - 1`` and ``bias - q``. Both ``q`` and
    ``1 + bias - q`` must lie in the strip ``(-ell, 2)``, and ``q`` is placed
    in the middle of the interval where they do. Keep ``1 + bias`` well above
    ``-2*ell`` at low ``ell``, so that neither transform sits near the edge
    of its strip.

    Its coefficients are the Mellin transform ``I_ell(s, t) / (4*pi)`` of
    Assassi et al. (2017) band-limited to the grid, so they approximate it
    rather than equal it. Contracting the transform with windows on the same
    grid equals the corresponding single-kernel FFTLog calculation to
    rounding. Only ``forward`` accepts the plan; ``inverse`` rejects it.

    Examples
    --------
    Double integral ``integral(wa(chi) * wb(chi') * integral(a(k) *
    j_ell(k*chi) * j_ell(k*chi'), k), chi, chi')`` for windows sampled on
    ``chi``, with ``m = max_offset`` and rows whose partner leaves the grid
    dropped::

        pp = double_spherical_bessel_plan(ell, n, dlog=dlog, bias=0.5, max_offset=m)
        kern = forward(a, pp)
        i = jnp.arange(m, n - m)
        band = (wb * chi)[i[:, None] + jnp.arange(-m, m + 1)]
        cl = dlog**2 * jnp.einsum("i,it,it->", wa[i], kern[i], band)
    """
    kernel = SphericalBesselJKernel(ell)
    lower, upper = kernel.domain
    real_s = 1 + jnp.asarray(bias)
    # Centre both transforms in their strips.
    q = (jnp.maximum(lower, real_s - upper) + jnp.minimum(upper, real_s - lower)) / 2
    first = plan(kernel, n, dlog=dlog, bias=q - 1, log_kr=log_kr)
    second = plan(kernel, n, dlog=dlog, bias=real_s - 1 - q, log_kr=log_kr)
    return product_plan(first, second, max_offset=max_offset)
