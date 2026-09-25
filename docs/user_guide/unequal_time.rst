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

``fftloggin`` never evaluates :math:`\mathcal{M}_\ell` directly. On the
paired grids, one FFTLog transform with the kernel :math:`j_\ell` is a matrix
:math:`M` whose entries depend only on the sum of the output index :math:`i`
and the input index :math:`m`. The kernel on the grid is then

.. math::

   K_\ell(\chi_i, \chi_j) = \frac{1}{\chi_j}
   \Big[ M\, \operatorname{diag}\!\Big(\frac{a_m}{k_m \Delta}\Big) M^{\mathsf T}
   \Big]_{ij} ,

a two-dimensional FFTLog transform of :math:`a` placed on the diagonal. Along
the diagonal :math:`j = i + d`, its entries form a one-dimensional FFTLog
transform whose discrete kernel is the product of the two transforms' discrete
kernels, shifted by :math:`d`. The band therefore costs one FFT per ratio, and
only the gamma functions of two single-kernel transforms are evaluated. Its
coefficients are :math:`\mathcal{M}_\ell` band-limited to the grid.

The two transforms share the bias. With :math:`q` in the middle of the
interval where both :math:`q` and :math:`1 + q_{\rm bias} - q` lie in the
strip :math:`(-\ell, 2)` of :math:`j_\ell`, they use the biases :math:`q - 1`
and :math:`q_{\rm bias} - q`, which keeps each as far from the edges of its
strip as possible.

The banded grid
---------------

:func:`~fftloggin.cosmology.double_spherical_bessel_plan` builds the
band-limited :math:`\mathcal{M}_\ell` at the FFTLog frequencies
:math:`s_m = 1 + q_{\rm bias} + 2\pi i m / (n\,\Delta)`, the ones
:func:`~fftloggin.fftlog.forward` uses, and at ratios on the grid's own
spacing,

.. math::

   t_j = e^{j\Delta}, \qquad j = -M, \dots, M .

Because :math:`t_j \chi_i = \chi_{i+j}`, every pair lands on the output grid
and no interpolation is needed. The result is a
:class:`~fftloggin.fftlog.Plan` whose coefficients have one column per ratio.
Passing it to :func:`~fftloggin.fftlog.forward` transforms the input samples
and returns an array of shape ``(n, 2M + 1)`` whose entry
``[i, M + j]`` is :math:`K_\ell(\chi_i, \chi_{i+j})`: a band of the kernel
around the diagonal :math:`\chi = \chi'`.

``half_width``
   The number :math:`M` of ratios on each side of :math:`t = 1`. It decides
   which pairs :math:`(\chi, \chi')` are available, namely
   :math:`|\ln(\chi'/\chi)| \le M\Delta`, not how accurate each value is.
   Pick it to span the separations your windows cover. The kernel falls off
   like :math:`t^{\ell}` away from the diagonal, so high multipoles need a
   narrower band. Memory and time grow linearly with it.

Exactness on the grid
   The band equals the two-dimensional transform above at every entry whose
   partner lies on the grid, so the contraction with windows on the same grid
   equals the bins-first calculation to rounding (see :doc:`tutorial_cl`).
   Rows whose partner leaves the grid wrap around periodically and must be
   dropped. Individual values of :math:`K_\ell` carry a truncation error,
   largest near :math:`t = 1`, because the exact kernel has structure on
   scales of :math:`1/k_{\max}` that the :math:`\chi` grid cannot resolve.

Choosing the bias
-----------------

As for any FFTLog transform, choose the bias so that :math:`a(k)\,k^{-q_{\rm bias}}`
decays at both ends of the :math:`k` grid; otherwise the periodic
continuation adds ringing, which is most visible at low :math:`\ell`.

Each of the two transforms also needs its own bias inside the strip of
:math:`j_\ell`. With the split above, the distance to the nearest edge is
:math:`r = \min(\ell + 2,\, 2\ell + 1 + q_{\rm bias},\, 3 - q_{\rm bias})/2`,
so :math:`1 + q_{\rm bias}` must stay well above :math:`-2\ell`: a more
negative bias brings both transforms closer to the lower edge of their strips
at low :math:`\ell`. For a Gaussian test input with :math:`n = 512` and
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

Other kernels
-------------

Nothing in the construction is specific to :math:`j_\ell`.
:func:`~fftloggin.fftlog.product_plan` combines any two single-kernel plans
built on the same grid, for example a spherical Bessel function and its
second derivative for redshift-space distortions:

.. code-block:: python

   from fftloggin import Derivative, SphericalBesselJKernel, plan, product_plan

   j = SphericalBesselJKernel(ell)
   p1 = plan(j, n, dlog=dlog, bias=-0.25)
   p2 = plan(j.transform(Derivative(2)), n, dlog=dlog, bias=-0.25)
   pp = product_plan(p1, p2, half_width=M)

:func:`~fftloggin.cosmology.double_spherical_bessel_plan` is the special
case of two :math:`j_\ell` with the bias split described above. For a general
pair:

- Each plan's :math:`1 + q` must lie in its own kernel's strip, and the bias
  seen by :math:`a(k)` is the sum of the two biases plus one.
  :class:`~fftloggin.kernels.Derivative` shifts a strip up by its order, so
  pairs of derivatives at low :math:`\ell` need carefully chosen biases.
- The integral :math:`\int dk\, a(k) K_1(k\chi) K_2(k\chi')` is no longer
  symmetric in :math:`\chi \leftrightarrow \chi'`. Including the
  prefactor :math:`\chi` of :math:`K`, the plan for :math:`(K_2, K_1)`
  gives :math:`K_{21}(\chi, \chi') = (\chi/\chi')\, K_{12}(\chi', \chi)`,
  so pass the plans in the order of the windows they multiply.
- The plans must share ``n``, ``dlog`` and ``log_kr``. The low-ringing
  ``log_kr`` of each kernel generally differs, so choose one shared value.

The contraction on the grid still equals the corresponding single-kernel
FFTLog calculation to rounding.

Differentiation and batching
----------------------------

The plan depends only on :math:`\ell`, :math:`n`, :math:`\Delta`, the bias,
:math:`\ln(k_c r_c)` and the band, not on the input. It carries these grid
parameters itself, so :func:`~fftloggin.fftlog.forward` takes no ``dlog``,
``bias`` or ``log_kr`` with a plan and cannot be called with values that
disagree with the coefficients. Compute it once per multipole, outside
``jax.grad``; ``forward`` is differentiable in its input. Derivatives with
respect to cosmological parameters, for example for a Fisher matrix, then
flow only through :math:`a(k)` and the windows. The order is a data leaf,
so ``jax.vmap`` batches plans over :math:`\ell`:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import forward
   from fftloggin.cosmology import double_spherical_bessel_plan

   make = jax.vmap(
       lambda ell: double_spherical_bessel_plan(
           ell, n, dlog=dlog, bias=-1.0, half_width=M
       )
   )
   plans = make(jnp.arange(2.0, 100.0))  # coeffs: (n_ell, n // 2 + 1, 2 M + 1)
   kernels = jax.vmap(forward, in_axes=(None, 0))(a, plans)  # (n_ell, n, 2 M + 1)

From kernel to power spectrum
-----------------------------

With windows sampled on the output grid :math:`\chi_i`, the double integral
over :math:`\chi` and :math:`\chi'` becomes a sum over the band. Rows whose
partner leaves the grid are dropped:

.. code-block:: python

   kern = forward(a, pp)
   i = jnp.arange(M, n - M)
   band = (wb * chi)[i[:, None] + jnp.arange(-M, M + 1)]
   cl = dlog**2 * jnp.einsum("i,it,it->", wa[i], kern[i], band)

Here ``a`` holds :math:`k^2 P(k)` times any transfer functions and the
:math:`2/\pi` prefactor, sampled on the paired :math:`k` grid.
