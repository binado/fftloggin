"""Double Bessel FFTLog transforms for unequal-time kernels.

The transform ``chi * integral(a(k) * K1(k*chi) * K2(k*t*chi), k)`` is
evaluated for all ``chi`` of the output grid and for ratios ``t`` on the same
logarithmic spacing. The Mellin transform of ``K1(x) * K2(t*x)`` is obtained
as a Mellin convolution of the two kernels' Mellin transforms, which avoids
hypergeometric functions with complex parameters.
"""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex, Float, Real

from .fftlog import _apply_coefficients, _samples
from .kernels import Kernel, SphericalBesselJKernel

__all__ = (
    "double_spherical_bessel_table",
    "kernel_product_table",
    "unequal_time_kernel",
)


def kernel_product_table(
    first: Kernel,
    second: Kernel,
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    half_width: int,
) -> Complex[Array, "m t"]:
    """Tabulate the Mellin transform of ``first(x) * second(t*x)``.

    Row ``m`` holds the arguments ``s_m = 1 + bias + 2j*pi*m/(n*dlog)``, the
    frequencies used by ``forward`` for ``n`` samples. Column ``j`` holds
    ``t = exp((j - half_width) * dlog)``. The value is
    ``integral(x**(s-1) * first(x) * second(t*x), x)``.

    Parameters
    ----------
    first, second : Kernel
        Mellin kernels of the two factors, for example
        ``SphericalBesselJKernel(ell)`` or ``Derivative(SphericalBesselJKernel(ell), 2)``.
        The table for ``(second, first)`` holds the same kernel with ``chi``
        and ``chi'`` swapped.
    n : int
        Number of samples of the transformed array.
    dlog : scalar
        Positive logarithmic spacing of the grids.
    bias : scalar, optional
        Power-law bias of the transform. ``1 + bias`` must lie in the
        convergence strip of ``first(x) * second(t*x)``; with strips
        ``(a1, b1)`` and ``(a2, b2)`` of the two kernels this is at most
        ``(a1 + a2, b1 + b2)``. Defaults to zero.
    half_width : int
        Number of ratios ``t`` on each side of ``t = 1``. The table covers
        ``|log(t)| <= half_width * dlog`` in ``2 * half_width + 1`` columns,
        so the output of ``unequal_time_kernel`` holds ``K(chi, chi')`` for
        ``chi'`` within ``half_width`` grid points of ``chi``. It sets which
        pairs are available, not the accuracy of each value. Choose it to
        span the separations covered by your windows; Bessel kernels fall
        off like ``t**ell`` away from ``t = 1``, so higher orders need a
        narrower band. Cost grows linearly with it.

    Returns
    -------
    array
        Complex table with shape ``(n // 2 + 1, 2 * half_width + 1)``.

    Notes
    -----
    The convolution runs along ``Re(w) = q``, which must lie in
    ``(a1, b1)`` while ``1 + bias - q`` lies in ``(a2, b2)``. The contour is
    placed in the middle of that interval, whose half-width ``r`` sets the
    aliasing error of the periodic convolution, ``exp(-r * n * dlog)``.
    Choose ``bias`` to keep the interval wide; derivative kernels at low
    orders leave it narrow.

    The table depends only on the kernels and the grid, so it can be computed
    once and reused for every input array. The convolution uses the
    transform's own frequencies, so contracting the kernel with windows on
    the same grid equals the corresponding single-kernel FFTLog calculation
    to rounding. Pointwise values carry a truncation error near ``t = 1``.
    Memory scales as ``n**2 / 2``.
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
    table = series * jnp.exp(-(real_s - q + 1j * omega[:, None]) * log_t) / dlog
    if n % 2 == 0:
        table = table.at[-1].set(jnp.real(table[-1]))
    return table


def double_spherical_bessel_table(
    ell: Real[ArrayLike, ""],
    n: int,
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    half_width: int,
) -> Complex[Array, "m t"]:
    """Tabulate the Mellin transform of ``j_ell(x) * j_ell(t*x)``.

    Equivalent to ``kernel_product_table`` with two
    ``SphericalBesselJKernel(ell)``. The values equal ``I_ell(s, t) / (4*pi)``
    in the notation of Assassi et al. (2017). ``1 + bias`` must lie in
    ``(-2*ell, 2)``. The aliasing error decays like ``exp(-r * n * dlog)``
    with ``r = min(ell + 2, 2*ell + 1 + bias, 3 - bias) / 2``, so keep
    ``1 + bias`` well above ``-2*ell`` at low ``ell``. See
    ``kernel_product_table`` for the parameters and the returned layout.
    """
    kernel = SphericalBesselJKernel(ell)
    return kernel_product_table(
        kernel, kernel, n, dlog=dlog, bias=bias, half_width=half_width
    )


def unequal_time_kernel(
    a: Float[ArrayLike, "n"],
    table: Complex[ArrayLike, "m t"],
    *,
    dlog: Float[ArrayLike, ""],
    bias: Float[ArrayLike, ""] = 0.0,
    log_kr: Float[ArrayLike, ""] = 0.0,
) -> Float[Array, "n t"]:
    """Transform ``a`` with a pair of kernels.

    Entry ``[i, j]`` is ``chi_i * integral(a(k) * K1(k*chi_i) *
    K2(k*t_j*chi_i), k)`` with ``t_j`` the ratios of ``table``. Since the
    ratios share the grid spacing, ``t_j * chi_i = chi_(i + j - half_width)``
    and the result is a band of the matrix ``K(chi, chi')``, which is symmetric
    when both kernels are equal.

    Parameters
    ----------
    a : array_like
        One-dimensional real samples on a logarithmic grid.
    table : array_like
        Output of ``kernel_product_table`` or
        ``double_spherical_bessel_table`` for ``a.shape[0]``
        samples and the same ``dlog`` and ``bias``.
    dlog : scalar
        Positive logarithmic spacing of the grids.
    bias : scalar, optional
        Power-law bias used to build ``table``. Defaults to zero.
    log_kr : scalar, optional
        Logarithm of the product of the geometric centers of the paired grids.
        Defaults to zero.

    Returns
    -------
    array
        Real samples with shape ``(n, table.shape[1])``.

    Examples
    --------
    Double integral ``integral(wa(chi) * wb(chi') * integral(a(k) *
    j_ell(k*chi) * j_ell(k*chi'), k), chi, chi')`` for windows sampled on
    ``chi``, with ``m = half_width`` and rows whose partner leaves the grid
    dropped::

        kern = unequal_time_kernel(a, table, dlog=dlog)
        i = jnp.arange(m, n - m)
        band = (wb * chi)[i[:, None] + jnp.arange(-m, m + 1)]
        cl = dlog**2 * jnp.einsum("i,it,it->", wa[i], kern[i], band)
    """
    a = _samples(a, "a")
    table = jnp.asarray(table)
    if table.ndim != 2 or table.shape[0] != a.shape[0] // 2 + 1:
        raise ValueError(
            f"table has shape {table.shape}, but {a.shape[0]} samples need "
            f"{a.shape[0] // 2 + 1} rows; build it with n={a.shape[0]}"
        )
    omega = 2 * jnp.pi * jnp.arange(table.shape[0]) / (a.shape[0] * dlog)
    coeffs = table * jnp.exp(-1j * omega * log_kr)[:, None]
    return _apply_coefficients(a, coeffs, dlog, bias, log_kr)
