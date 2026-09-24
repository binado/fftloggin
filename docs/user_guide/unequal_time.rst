Unequal-time kernels
====================

Angular power spectra beyond the Limber approximation involve two spherical
Bessel functions of the same wavenumber,

.. math::

   C_\ell^{ab} = \frac{2}{\pi} \int d\chi\, W_a(\chi) \int d\chi'\, W_b(\chi')
   \int_0^\infty dk\, k^2 P(k; \chi, \chi')\,
   j_\ell(k\chi)\, j_\ell(k\chi') .

The usual approach integrates over :math:`\chi` first, with one FFTLog
transform per window. Integrating over :math:`k` first instead gives the
*unequal-time kernel*

.. math::

   K_\ell(\chi, \chi') = \chi \int_0^\infty dk\, a(k)\,
   j_\ell(k\chi)\, j_\ell(k\chi') ,

which does not depend on the windows. You need one kernel per pair of
transfer-function types (for example density and redshift-space
distortions), however many tomographic bins there are. The module
:mod:`fftloggin.cosmology` computes it with FFTLog.

The Mellin transform of two Bessel functions
--------------------------------------------

Write :math:`\chi' = t\chi`. For fixed :math:`t`, the kernel is an ordinary
FFTLog transform (see :doc:`concepts`) with the integral kernel
:math:`j_\ell(x)\, j_\ell(tx)`. Its Mellin transform (see :doc:`kernels`) is

.. math::

   \mathcal{M}_\ell(s, t) = \int_0^\infty x^{s-1} j_\ell(x)\, j_\ell(tx)\, dx
   = \frac{\mathsf{I}_\ell(s, t)}{4\pi} ,

where :math:`\mathsf{I}_\ell` is the function of
`Assassi et al. (2017) <https://arxiv.org/abs/1705.05022>`_. It converges for
:math:`-2\ell < \operatorname{Re} s < 2`. Assassi et al. give it as a Gauss
hypergeometric function of :math:`t^2` with complex parameters. For the large
imaginary parts FFTLog needs, the terms of that series grow to tens of orders
of magnitude above its sum, so it is unusable in double precision.

``fftloggin`` avoids the hypergeometric function. By the Parseval formula for
Mellin transforms, applied to :math:`j_\ell(x)` and :math:`j_\ell(tx)`,

.. math::

   \mathcal{M}_\ell(s, t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega\;
   U_\ell(q + i\omega)\, U_\ell(s - q - i\omega)\, t^{-(s - q - i\omega)} ,

where :math:`U_\ell` is the Mellin transform of a single spherical Bessel
function, already provided by
:class:`~fftloggin.kernels.SphericalBesselJKernel`, and :math:`q` is the real
part of the integration contour. Both :math:`q` and
:math:`\operatorname{Re} s - q` must lie in the strip :math:`(-\ell, 2)`.
The factor :math:`t^{i\omega}` makes the integral a Fourier transform in
:math:`\ln t`, so one FFT gives :math:`\mathcal{M}_\ell` at every
:math:`t` at once, and only gamma functions are evaluated.

The banded grid
---------------

:func:`~fftloggin.cosmology.double_spherical_bessel_table` evaluates
:math:`\mathcal{M}_\ell` at the FFTLog frequencies
:math:`s_m = 1 + q_{\rm bias} + 2\pi i m / (n\,\Delta)`, the ones
:func:`~fftloggin.fftlog.forward` uses, and at ratios on the grid's own
spacing,

.. math::

   t_j = e^{j\Delta}, \qquad j = -M, \dots, M .

Because :math:`t_j \chi_i = \chi_{i+j}`, every pair lands on the output grid
and no interpolation is needed.
:func:`~fftloggin.cosmology.unequal_time_kernel` applies the table to the
input samples and returns an array of shape ``(n, 2M + 1)`` whose entry
``[i, M + j]`` is :math:`K_\ell(\chi_i, \chi_{i+j})`: a band of the kernel
around the diagonal :math:`\chi = \chi'`.

``half_width``
   The number :math:`M` of ratios on each side of :math:`t = 1`. It decides
   which pairs :math:`(\chi, \chi')` are available, namely
   :math:`|\ln(\chi'/\chi)| \le M\Delta`, not how accurate each value is.
   Pick it to span the separations your windows cover. The kernel falls off
   like :math:`t^{\ell}` away from the diagonal, so high multipoles need a
   narrower band. Memory and time grow linearly with it.

Frequency cutoff
   The Fourier integral above runs over the transform's own frequencies,
   :math:`|\omega| \le \pi/\Delta`, so the table uses exactly the
   frequencies the transforms do. The contraction with windows on the same
   grid then equals the bins-first calculation to rounding (see
   :doc:`tutorial_cl`). Individual values of :math:`K_\ell` carry a
   truncation error, largest near :math:`t = 1`, that scales like
   :math:`(\pi/\Delta)^{q_{\rm bias} - 1}`. A finer convolution would
   reduce it, but the exact kernel has structure on scales of
   :math:`1/k_{\max}` that the :math:`\chi` grid cannot resolve, so it
   would make the sum over the grid *less* accurate.

Choosing the bias
-----------------

As for any FFTLog transform, choose the bias so that :math:`a(k)\,k^{-q}`
decays at both ends of the :math:`k` grid; otherwise the periodic
continuation adds ringing, which is most visible at low :math:`\ell`.

The FFT also makes the convolution periodic in :math:`\ln t` with period
:math:`n\Delta`. Its aliasing error decays like
:math:`\exp(-r\, n\Delta)` with
:math:`r = \min(\ell + q, \ell + 1 + q_{\rm bias} - q)`, so the contour must
stay well inside its strip. A more negative bias speeds up the decay in
:math:`\omega` but brings the contour closer to the lower edge at low
:math:`\ell`. For a Gaussian test input with :math:`n = 512` and
:math:`\Delta = 0.02`, the largest absolute errors against direct quadrature
were:

=====  ======  =========
ell    bias    error
=====  ======  =========
1      -2      6e-3
1      -1      4e-5
1       0      2e-7
2      -2      1e-7
2      -1      8e-10
=====  ======  =========

Differentiation and batching
----------------------------

The table depends only on :math:`\ell`, :math:`n`, :math:`\Delta`, the bias
and the band, not on the input. Compute it once per multipole, outside
``jax.grad``, and pass it to
:func:`~fftloggin.cosmology.unequal_time_kernel`, which is differentiable in
its input. Derivatives with respect to cosmological parameters, for example
for a Fisher matrix, then flow only through :math:`a(k)` and the windows.
The order is a data leaf, so ``jax.vmap`` batches tables over :math:`\ell`:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin.cosmology import double_spherical_bessel_table, unequal_time_kernel

   make = jax.vmap(
       lambda ell: double_spherical_bessel_table(
           ell, n, dlog=dlog, bias=-1.0, half_width=M
       )
   )
   tables = make(jnp.arange(2.0, 100.0))  # (n_ell, n // 2 + 1, 2 M + 1)
   kernels = jax.vmap(lambda table: unequal_time_kernel(a, table, dlog=dlog, bias=-1.0))(
       tables
   )  # (n_ell, n, 2 M + 1)

Building a table needs about :math:`n^2 / 2`
complex values of temporary memory, so batch large ranges of :math:`\ell` in
chunks with ``jax.lax.map``.

From kernel to power spectrum
-----------------------------

With windows sampled on the output grid :math:`\chi_i`, the double integral
over :math:`\chi` and :math:`\chi'` becomes a sum over the band. Rows whose
partner leaves the grid are dropped:

.. code-block:: python

   kern = unequal_time_kernel(a, table, dlog=dlog, bias=-1.0)
   i = jnp.arange(M, n - M)
   band = (wb * chi)[i[:, None] + jnp.arange(-M, M + 1)]
   cl = dlog**2 * jnp.einsum("i,it,it->", wa[i], kern[i], band)

Here ``a`` holds :math:`k^2 P(k)` times any transfer functions and the
:math:`2/\pi` prefactor, sampled on the paired :math:`k` grid.
