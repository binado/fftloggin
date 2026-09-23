Cosmological correlation functions
===================================

This tutorial adapts the repository's notebook example: transform a matter
power spectrum into the real-space correlation function. The notebook remains
available at ``notebooks/tutorial.ipynb`` with plotting, timing, and CAMB
calculations.

The radial correlation function is

.. math::

   \xi(r) = \frac{1}{2\pi^2}\int_0^\infty dk\,k^2P(k)j_0(kr).

Using
``j_\ell(x) = \sqrt{\pi/(2x)}J_{\ell+1/2}(x)``, this becomes an inverse
Hankel transform with a Bessel kernel of order ``1/2``. The sample passed to
``inverse`` is ``k**(3/2) * P(k)``; its output is divided by
``(2*pi*r)**(3/2)`` to recover ``xi(r)``.

Generate an input spectrum
--------------------------

The example uses CAMB to obtain a spectrum on a uniform logarithmic grid. The
figures below are generated during the Sphinx build using the same CAMB,
FFTLog and direct-integration calculations. The ``docs`` dependency group
includes CAMB, NumPy, SciPy and Matplotlib for this purpose.

.. code-block:: python

   import camb
   import numpy as np

   minkh, maxkh, npoints = 1e-4, 1.0, 512
   dlog = np.log(maxkh / minkh) / (npoints - 1)
   params = camb.set_params(
       H0=67.5,
       ombh2=0.022,
       omch2=0.122,
       mnu=0.06,
       omk=0,
       tau=0.06,
       As=2e-9,
       ns=0.965,
       halofit_version="mead",
       lmax=3000,
   )
   params.set_matter_power(
       redshifts=[0], kmax=maxkh / 0.675,
       k_per_logint=int(1 / dlog),
   )
   results = camb.get_results(params)
   kh, redshifts, power = results.get_matter_power_spectrum(
       minkh=minkh, maxkh=maxkh, npoints=npoints
   )

Prepare the data used by the figures. Sphinx runs this setup once and shares
its variables with the plot directives below.

.. plot::
   :context:
   :nofigs:

   import camb
   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import numpy as np
   import scipy.special

   from fftloggin import BesselJKernel, get_paired_grids, inverse, lowring_log_kr

   jax.config.update("jax_enable_x64", True)

   h = 0.675
   minkh, maxkh, npoints = 1e-4, 1.0, 512
   dlog = np.log(maxkh / minkh) / (npoints - 1)
   redshift_slices = [0, 1, 2]
   params = camb.set_params(
       H0=100 * h,
       ombh2=0.022,
       omch2=0.122,
       mnu=0.06,
       omk=0,
       tau=0.06,
       As=2e-9,
       ns=0.965,
       halofit_version="mead",
       lmax=3000,
   )
   params.set_matter_power(
       redshifts=redshift_slices,
       kmax=maxkh / h,
       k_per_logint=int(1 / dlog),
   )
   results = camb.get_results(params)
   kh, redshifts, power = results.get_matter_power_spectrum(
       minkh=minkh, maxkh=maxkh, npoints=npoints
   )

   kernel = BesselJKernel(0.5)

   def correlation(spectrum, *, bias=0.0, log_kr=0.0):
       r, _ = get_paired_grids(k=jnp.asarray(kh), log_kr=log_kr)
       r = np.asarray(jax.device_get(r))
       samples = jnp.asarray(kh ** (3 / 2) * spectrum)
       xi = inverse(
           samples,
           kernel,
           dlog=dlog,
           bias=bias,
           log_kr=log_kr,
       )
       xi = np.asarray(jax.device_get(xi)) / (2 * np.pi * r) ** (3 / 2)
       return r, xi

   def direct_correlation(r, spectrum):
       integrand = spectrum * kh**3 * scipy.special.spherical_jn(0, np.outer(r, kh))
       return np.trapezoid(integrand, x=np.log(kh), axis=-1) / (2 * np.pi**2)

Matter power spectrum
---------------------

.. plot::
   :context: close-figs
   :caption: CAMB matter power spectra at redshifts 0, 1 and 2.
   :alt: Matter power spectra as a function of wavenumber for three redshifts.

   fig, ax = plt.subplots()
   for i, redshift in enumerate(redshifts):
       ax.loglog(kh, power[i], label=rf"$z={redshift}$")
   ax.set_xlabel(r"$k\;[h/\mathrm{Mpc}]$")
   ax.set_ylabel(r"$P(k)\;[(\mathrm{Mpc}/h)^3]$")
   ax.set_title("Matter Power Spectrum")
   ax.legend()
   ax.grid()

Transform to real space
-----------------------

Use a low-ringing center consistently for both the inverse transform and its
paired grid. Enable JAX 64-bit mode before importing JAX when comparing with
high-precision quadrature.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import (
       BesselJKernel,
       get_paired_grids,
       inverse,
       lowring_log_kr,
   )

   kernel = BesselJKernel(0.5)
   bias = -0.3
   log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias)
   r, _ = get_paired_grids(k=jnp.asarray(kh), log_kr=log_kr)
   xi = inverse(
       jnp.asarray(kh ** (3 / 2) * power[0]),
       kernel,
       dlog=dlog,
       bias=bias,
       log_kr=log_kr,
   ) / (2 * jnp.pi * r) ** (3 / 2)

Compare against direct integration by integrating ``P(k) * k**3 * j_0(kr)``
over ``log(k)`` and dividing by ``2*pi**2``. The notebook contains the full
comparison and discusses the effect of the low-ringing center.

At the default center, FFTLog and direct integration differ most near the
ends of the finite input grid:

.. plot::
   :context: close-figs
   :caption: Default-center FFTLog transform compared with direct integration.
   :alt: Correlation function from FFTLog and direct integration at log_kr zero.

   r, xi_fftlog = correlation(power[0])
   xi_direct = direct_correlation(r, power[0])

   fig, ax = plt.subplots()
   ax.loglog(r, xi_fftlog, label="FFTLog")
   ax.loglog(r, xi_direct, label="Direct Integration")
   ax.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$")
   ax.set_ylabel(r"$\xi(r)$")
   ax.legend()
   ax.grid()

Choosing a low-ringing center and a bias of ``-0.3`` reduces ringing in the
FFTLog result:

.. plot::
   :context: close-figs
   :caption: Low-ringing FFTLog transform compared with direct integration.
   :alt: Correlation function with a low-ringing FFTLog center and direct integration.

   bias = -0.3
   log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias)
   r, xi_fftlog = correlation(power[0], bias=bias, log_kr=log_kr)
   xi_direct = direct_correlation(r, power[0])

   fig, ax = plt.subplots()
   ax.loglog(r, xi_fftlog, label="FFTLog")
   ax.loglog(r, xi_direct, label="Direct Integration")
   ax.set_xlabel(r"$r\;[\mathrm{Mpc}/h]$")
   ax.set_ylabel(r"$\xi(r)$")
   ax.set_title(f"Low-ringing center, bias={bias:.1f}")
   ax.legend()
   ax.grid()
