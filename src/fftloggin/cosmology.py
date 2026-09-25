"""Double Bessel FFTLog transforms for unequal-time kernels.

The transform ``chi * integral(a(k) * K1(k*chi) * K2(k*t*chi), k)`` is
evaluated for all ``chi`` of the output grid and for ratios ``t`` on the same
logarithmic spacing, by passing the plans built here to ``forward``. The Mellin transform of ``K1(x) * K2(t*x)`` is obtained
as a Mellin convolution of the two kernels' Mellin transforms, which avoids
hypergeometric functions with complex parameters.
"""

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from .fftlog import Plan
from .kernels import Kernel, SphericalBesselJKernel

__all__ = ("double_spherical_bessel_plan", "kernel_product_plan")


def kernel_product_plan(
    first: Kernel,
    second: Kernel,
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
    half_width: int,
) -> Plan:
    """Plan the FFTLog transform with the kernel pair ``first(x) * second(t*x)``.

    ``forward(a, plan)`` returns an array ``K`` of shape
    ``(n, 2 * half_width + 1)`` whose entry ``[i, j]`` is
    ``chi_i * integral(a(k) * first(k*chi_i) * second(k*t_j*chi_i), k)``
    with ``t_j = exp((j - half_width) * dlog)``. Since the ratios share the
    grid spacing, ``t_j * chi_i = chi_(i + j - half_width)`` and ``K`` is a
    band of the matrix ``K(chi, chi') = chi * I(chi, chi')`` with
    ``I(chi, chi') = integral(a(k) * first(k*chi) * second(k*chi'), k)``.
    Only the first coordinate multiplies the integral, so ``K`` is not
    symmetric even for equal kernels: ``I`` is, which gives
    ``K(chi, chi') = (chi / chi') * K(chi', chi)``. Swapping the kernels
    gives ``K_21(chi, chi') = (chi / chi') * K_12(chi', chi)``.

    Row ``m`` of ``plan.coeffs`` holds the Mellin transform
    ``integral(x**(s-1) * first(x) * second(t*x), x)`` at
    ``s_m = 1 + bias + 2j*pi*m/(n*dlog)``, the frequencies used by
    ``forward`` for ``n`` samples, times the phase ``exp(-2j*pi*m*log_kr/(n*dlog))``.
    Column ``j`` holds ``t_j``.

    Parameters
    ----------
    first, second : Kernel
        Mellin kernels of the two factors, for example
        ``SphericalBesselJKernel(ell)`` or
        ``SphericalBesselJKernel(ell).transform(Derivative(2))``.
        Swapping them swaps the roles of ``chi`` and ``chi'``.
    n : int
        Number of samples of the transformed array.
    dlog : scalar
        Positive logarithmic spacing of the grids.
    bias : scalar, optional
        Power-law bias of the transform. ``1 + bias`` must lie in the
        convergence strip of ``first(x) * second(t*x)``; with strips
        ``(a1, b1)`` and ``(a2, b2)`` of the two kernels this is at most
        ``(a1 + a2, b1 + b2)``. Defaults to zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.
    half_width : int
        Number of ratios ``t`` on each side of ``t = 1``. The plan covers
        ``|log(t)| <= half_width * dlog`` in ``2 * half_width + 1`` columns,
        so the transform holds ``K(chi, chi')`` for
        ``chi'`` within ``half_width`` grid points of ``chi``. It sets which
        pairs are available, not the accuracy of each value. Choose it to
        span the separations covered by your windows; Bessel kernels fall
        off like ``t**ell`` away from ``t = 1``, so higher orders need a
        narrower band. Cost grows linearly with it.

    Returns
    -------
    Plan
        Plan with coefficients of shape ``(n // 2 + 1, 2 * half_width + 1)``.

    Notes
    -----
    The convolution runs along ``Re(w) = q``, which must lie in
    ``(a1, b1)`` while ``1 + bias - q`` lies in ``(a2, b2)``. The contour is
    placed in the middle of that interval, whose half-width ``r`` sets the
    aliasing error of the periodic convolution, ``exp(-r * n * dlog)``.
    Choose ``bias`` to keep the interval wide; derivative kernels at low
    orders leave it narrow.

    The plan depends only on the kernels and the grid, so it can be computed
    once and reused for every input array. The convolution uses the
    transform's own frequencies, so contracting the kernel with windows on
    the same grid equals the corresponding single-kernel FFTLog calculation
    to rounding. Pointwise values carry a truncation error near ``t = 1``.
    Memory scales as ``n**2 / 2``. Only ``forward`` accepts a kernel-pair
    plan; ``inverse`` rejects it.

    Examples
    --------
    Double integral ``integral(wa(chi) * wb(chi') * integral(a(k) *
    j_ell(k*chi) * j_ell(k*chi'), k), chi, chi')`` for windows sampled on
    ``chi``, with ``m = half_width`` and rows whose partner leaves the grid
    dropped::

        pp = kernel_product_plan(j, j, n, dlog=dlog, bias=0.5, half_width=m)
        kern = forward(a, pp)
        i = jnp.arange(m, n - m)
        band = (wb * chi)[i[:, None] + jnp.arange(-m, m + 1)]
        cl = dlog**2 * jnp.einsum("i,it,it->", wa[i], kern[i], band)
    """
    if half_width < 0:
        raise ValueError("half_width must be >= 0")
    real_s = 1 + jnp.asarray(bias)
    lower1, upper1 = first.domain
    lower2, upper2 = second.domain
    # Centring the contour in its interval maximizes the aliasing decay rate.
    lower = jnp.maximum(lower1, real_s - upper2)
    upper = jnp.minimum(upper1, real_s - lower2)
    q = (lower + upper) / 2
    omega = 2 * jnp.pi * jnp.arange(n // 2 + 1) / (n * dlog)
    # Convolution frequencies in FFT order, with period n * dlog in log(t).
    conv = 2 * jnp.pi * jnp.fft.fftfreq(n, dlog)
    integrand = first(q + 1j * conv) * second(real_s - q + 1j * (omega[:, None] - conv))
    offsets = jnp.arange(-half_width, half_width + 1)
    series = jnp.fft.ifft(integrand, axis=-1)[:, offsets % n]
    log_t = offsets * dlog
    coeffs = series * jnp.exp(-(real_s - q + 1j * omega[:, None]) * log_t) / dlog
    coeffs = coeffs * jnp.exp(-1j * omega * log_kr)[:, None]
    if n % 2 == 0:
        coeffs = coeffs.at[-1].set(jnp.real(coeffs[-1]))
    return Plan(
        coeffs=coeffs,
        dlog=jnp.asarray(dlog),
        bias=jnp.asarray(bias),
        log_kr=jnp.asarray(log_kr),
        n=n,
    )


def double_spherical_bessel_plan(
    ell: Real[ArrayLike, ""],
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
    half_width: int,
) -> Plan:
    """Plan the FFTLog transform with the kernel pair ``j_ell(x) * j_ell(t*x)``.

    Equivalent to ``kernel_product_plan`` with two
    ``SphericalBesselJKernel(ell)``. The coefficients at ``log_kr = 0`` equal ``I_ell(s, t) / (4*pi)``
    in the notation of Assassi et al. (2017). ``1 + bias`` must lie in
    ``(-2*ell, 2)``. The aliasing error decays like ``exp(-r * n * dlog)``
    with ``r = min(ell + 2, 2*ell + 1 + bias, 3 - bias) / 2``, so keep
    ``1 + bias`` well above ``-2*ell`` at low ``ell``. See
    ``kernel_product_plan`` for the parameters and the transform layout.
    """
    kernel = SphericalBesselJKernel(ell)
    return kernel_product_plan(
        kernel, kernel, n, dlog=dlog, bias=bias, log_kr=log_kr, half_width=half_width
    )
