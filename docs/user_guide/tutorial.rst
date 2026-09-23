Cosmological correlation functions
==================================

The two-point correlation function :math:`\xi(r)` and the power spectrum
:math:`P(k)` describe the same clustering of matter, one in configuration
space and one in Fourier space. Theory codes such as CAMB compute
:math:`P(k)`, while galaxy surveys often measure :math:`\xi(r)`, so turning
one into the other is a routine step. This is the application Hamilton (2000)
built FFTLog for. This tutorial computes :math:`\xi(r)`, including the baryon
acoustic oscillation (BAO) peak, and checks the result against direct
integration.

The tutorial needs CAMB, SciPy and Matplotlib, which are in the ``docs``
dependency group (``uv sync --group docs``). The notebook
``notebooks/tutorial.ipynb`` in the repository covers the same example
interactively.

From ξ(r) to a Hankel transform
-------------------------------

For an isotropic field, the three-dimensional Fourier integral reduces to a
single integral over :math:`k`:

.. math::

   \xi(r) = \frac{1}{2\pi^2} \int_0^\infty P(k)\, j_0(kr)\, k^2\, dk .

Writing the spherical Bessel function as
:math:`j_0(x) = (\pi/2x)^{1/2} J_{1/2}(x)` (Hamilton §B.1) and collecting
powers of :math:`r` and :math:`k`,

.. math::

   \xi(r) = \frac{1}{(2\pi r)^{3/2}}
   \int_0^\infty \left[k^{3/2} P(k)\right] J_{1/2}(kr)\; r\, dk ,

using :math:`\sqrt{\pi/2}\,/\,(2\pi^2) = (2\pi)^{-3/2}`. This is Hamilton's
inverse transform (159) with :math:`\mu = 1/2`,

.. math::

   \tilde{A}(k) = k^{3/2} P(k), \qquad A(r) = (2\pi r)^{3/2}\, \xi(r),

so ``inverse(k**1.5 * P, BesselJKernel(0.5), ...)`` returns
:math:`(2\pi r)^{3/2} \xi(r)` on the paired :math:`r` grid. For
:math:`\mu = 1/2` the bias must satisfy :math:`-3/2 < q < 1/2`
(see :doc:`kernels`).

The power spectrum
------------------

CAMB computes the nonlinear matter power spectrum at three redshifts on a grid
that is uniform in :math:`\ln k`, as FFTLog requires. The grid runs to
:math:`k = 10\,h/\mathrm{Mpc}`; the accuracy section below shows why the upper
limit matters.

.. plot::
   :context:
   :nofigs:
   :include-source: true

   import camb
   import numpy as np

   h = 0.675
   minkh, maxkh, npoints = 1e-4, 10.0, 1024
   dlog = np.log(maxkh / minkh) / (npoints - 1)

   params = camb.set_params(
       H0=100 * h, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,
       As=2e-9, ns=0.965, halofit_version="mead", lmax=3000,
   )
   params.set_matter_power(
       redshifts=[0, 1, 2], kmax=maxkh / h, k_per_logint=int(1 / dlog)
   )
   results = camb.get_results(params)
   kh, redshifts, power = results.get_matter_power_spectrum(
       minkh=minkh, maxkh=maxkh, npoints=npoints
   )

.. plot::
   :context: close-figs
   :caption: Nonlinear matter power spectra from CAMB.
   :alt: Matter power spectra against wavenumber for redshifts 0, 1 and 2.

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6.5, 4))
   for z, spectrum in zip(redshifts, power):
       ax.loglog(kh, spectrum, label=f"$z={z:g}$")
   ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
   ax.set_ylabel(r"$P(k)\;[h^{-3}\,\mathrm{Mpc}^{3}]$")
   ax.legend()
   ax.grid(alpha=0.3)
   fig.tight_layout()

Transform to real space
-----------------------

Choose a bias inside the strip and the nearest low-ringing ``log_kr``, build
the paired :math:`r` grid from that same value, and undo the
:math:`(2\pi r)^{3/2}` factor:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source: true

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, get_paired_grids, inverse, lowring_log_kr

   jax.config.update("jax_enable_x64", True)  # or export JAX_ENABLE_X64=1

   kernel = BesselJKernel(0.5)

   def correlation(k, spectrum, *, bias=-0.3):
       log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias)
       r, _ = get_paired_grids(k=k, log_kr=log_kr)
       transformed = inverse(k**1.5 * spectrum, kernel, dlog=dlog, bias=bias,
                             log_kr=log_kr)
       return np.asarray(r), np.asarray(transformed / (2 * jnp.pi * r) ** 1.5)

   r, xi = correlation(kh, power[0])

On large scales, :math:`\xi(r)` falls below zero near
:math:`120\,h^{-1}\mathrm{Mpc}`, so a log-log plot hides the most interesting
part. The usual way to show it is :math:`r^2 \xi(r)` on linear axes, where
the BAO peak near :math:`100\,h^{-1}\mathrm{Mpc}` stands out:

.. plot::
   :context: close-figs
   :caption: r^2 xi(r) at three redshifts. The BAO peak sits near
             100 Mpc/h at every redshift, while its amplitude grows toward
             z = 0 as structure forms.
   :alt: r squared times the correlation function against r for three
         redshifts, showing the BAO peak near 100 Mpc/h.

   fig, ax = plt.subplots(figsize=(6.5, 4))
   for z, spectrum in zip(redshifts, power):
       r, xi = correlation(kh, spectrum)
       window = (r > 20) & (r < 200)
       ax.plot(r[window], r[window] ** 2 * xi[window], label=f"$z={z:g}$")
   ax.axhline(0, color="0.7", lw=0.8)
   ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
   ax.set_ylabel(r"$r^2\,\xi(r)\;[h^{-2}\,\mathrm{Mpc}^{2}]$")
   ax.legend()
   fig.tight_layout()

How accurate is it?
-------------------

To check the transform, compare against the same integral evaluated
directly. The integrand oscillates rapidly at large :math:`kr`, so an
ordinary trapezoid rule on the FFTLog grid is not accurate enough. Instead,
write :math:`j_0(kr) = \sin(kr)/kr`, interpolate :math:`\ln P` with a cubic
spline in :math:`\ln k`, and use QUADPACK's oscillatory-weight quadrature
through :func:`scipy.integrate.quad`:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source: true

   import warnings

   import scipy.integrate
   import scipy.interpolate

   def reference(r_values, k, spectrum):
       log_power = scipy.interpolate.CubicSpline(np.log(k), np.log(spectrum))

       def integrand(kk):
           return kk * np.exp(log_power(np.log(kk)))

       with warnings.catch_warnings():
           warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
           return np.array([
               scipy.integrate.quad(integrand, k[0], k[-1], weight="sin",
                                    wvar=rv, limit=5000)[0] / (2 * np.pi**2 * rv)
               for rv in r_values
           ])

Both methods use exactly the same tabulated spectrum, so any difference comes
from FFTLog's assumption that the input is periodic in :math:`\ln k`. The
figure compares three set-ups:

.. plot::
   :context: close-figs
   :caption: Absolute difference between FFTLog and direct quadrature, in
             units of r^2 xi. Grey band: the BAO range. Raising k_max fixes
             the BAO range; the bias fixes large r.
   :alt: Difference between FFTLog and quadrature against r for three
         choices of k_max and bias.

   probe = np.geomspace(0.5, 3000, 120)
   setups = [
       (r"$k_{\max} = 1$, $q = 0$", kh <= 1.0 + 1e-9, 0.0),
       (r"$k_{\max} = 10$, $q = 0$", np.ones_like(kh, dtype=bool), 0.0),
       (r"$k_{\max} = 10$, $q = -0.3$", np.ones_like(kh, dtype=bool), -0.3),
   ]
   fig, ax = plt.subplots(figsize=(6.5, 4))
   ax.axvspan(20, 200, color="0.9")
   for label, mask, bias in setups:
       k, spectrum = kh[mask], power[0][mask]
       r, xi = correlation(k, spectrum, bias=bias)
       error = probe**2 * np.abs(np.interp(probe, r, xi) - reference(probe, k, spectrum))
       ax.loglog(probe, error, label=label)
   ax.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$")
   ax.set_ylabel(r"$r^2\,|\Delta\xi|\;[h^{-2}\,\mathrm{Mpc}^{2}]$")
   ax.legend(fontsize=8)
   fig.tight_layout()

Three lessons follow, each an instance of the periodicity argument in
:doc:`concepts`:

1. **The k range sets the accuracy at the BAO scale.** Cutting the spectrum
   at :math:`k = 1\,h/\mathrm{Mpc}` leaves a large, abrupt step in
   :math:`k^{3/2}P(k)` where the periodic copies meet. Across the BAO range
   the error is a few units of :math:`r^2\xi`, comparable to the BAO
   feature itself (:math:`r^2\xi \approx 16` at the peak). Extending to :math:`k = 10\,h/\mathrm{Mpc}` shrinks the step and the
   error by more than an order of magnitude.
2. **The bias sets the accuracy at large r.** In the inverse transform, the
   sequence treated as periodic is :math:`\tilde{A}(k)\,k^{q} =
   k^{3/2 + q} P(k)` (157). Its high-:math:`k` end is the larger of the two
   steps, so a negative :math:`q` that steepens that tail helps. A positive
   :math:`q` makes it worse: :math:`q = +0.3` raises the large-:math:`r`
   error roughly thirtyfold.
3. **Low ringing changes little here.** For a spectrum this smooth, the
   Nyquist component that condition (184) protects is tiny. The low-ringing
   ``log_kr`` costs nothing, though, so it remains a sensible default.

The same transform with a spherical Bessel kernel
-------------------------------------------------

``SphericalBesselJKernel`` evaluates the original integral directly. Reading
:math:`\xi(r) = (2\pi^2)^{-1} \int_0^\infty P(k)\, j_0(kr)\, k^2\, dk` as a
forward transform (158) with the roles of :math:`r` and :math:`k` exchanged:

.. code-block:: python

   from fftloggin import SphericalBesselJKernel, forward

   kernel_j0 = SphericalBesselJKernel(0)
   log_kr = lowring_log_kr(kernel_j0, dlog=dlog, bias=0.5)
   r, _ = get_paired_grids(k=kh, log_kr=log_kr)
   xi_j0 = forward(
       kh**2 * power[0], kernel_j0, dlog=dlog, bias=0.5, log_kr=log_kr
   ) / (2 * np.pi**2 * r)

With ``bias=0.5``, this agrees with the :math:`J_{1/2}` route at ``bias=0``
to rounding error, and this is expected. The sequence that enters the FFT is
:math:`k^{2}P(k)\,k^{-1/2} = k^{3/2}P(k)` in both cases. The spherical kernel
simply absorbs the factor :math:`k^{1/2}` into its Mellin transform. With
``bias=0`` it would instead put :math:`k^2 P(k)` into the FFT, whose
high-:math:`k` step is much larger. Choosing a kernel and choosing a bias are
two views of the same freedom. (At a non-low-ringing ``log_kr`` the two
routes differ slightly, about :math:`10^{-5}` here: one is a forward and the
other an inverse transform, and these coincide only under condition (185).)

Next steps
----------

- Compute the quadrupole of the redshift-space correlation function with
  ``SphericalBesselJKernel(2)``.
- Differentiate :math:`\xi(r)` with respect to a spectral parameter by writing
  the spectrum as a JAX function and applying ``jax.grad``
  (see :doc:`jax`).
- Batch the three redshifts in one call with ``jax.vmap``.
