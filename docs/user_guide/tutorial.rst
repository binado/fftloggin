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
``tutorial`` dependency group in this repository includes CAMB, NumPy, SciPy,
Matplotlib and Jupyter support. For a pip environment, install CAMB,
Matplotlib, SciPy and NumPy in addition to ``fftloggin``.

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
