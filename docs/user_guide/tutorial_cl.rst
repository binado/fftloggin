Angular power spectra beyond Limber
===================================

This tutorial computes the angular power spectrum of galaxy number counts in
two tomographic bins without the Limber approximation, in two ways:

- **bins first**, one :func:`~fftloggin.fftlog.forward` transform per bin
  followed by an integral over :math:`k`;
- **k first**, one :func:`~fftloggin.cosmology.unequal_time_kernel` per
  multipole, contracted with every pair of bins (see :doc:`unequal_time`).

Both are checked against CAMB's own line-of-sight calculation,
:meth:`camb.results.CAMBdata.get_source_cls_dict`. The tutorial needs CAMB
and Matplotlib, which are in the ``docs`` dependency group
(``uv sync --group docs``).

The model
---------

With linear, scale-independent growth, the matter power spectrum factorizes
as :math:`P(k; \chi, \chi') = D(\chi) D(\chi') P_0(k)`. For tracers with
unit bias and redshift distributions :math:`n_a(z)`, the density term of the
number counts is

.. math::

   C_\ell^{ab} = \frac{2}{\pi} \int d\chi\, w_a(\chi) \int d\chi'\, w_b(\chi')
   \int_0^\infty dk\, k^2 P_0(k)\, j_\ell(k\chi)\, j_\ell(k\chi') ,
   \qquad
   w_a(\chi) = n_a(z)\, \frac{dz}{d\chi}\, D(\chi) .

To compare like with like, CAMB is run with the linear power spectrum,
massless neutrinos (so growth is scale-independent), only the density source
term, and Limber switched off. Both bins are Gaussian in redshift, as CAMB's
:class:`~camb.sources.GaussianSourceWindow` defines them, and they overlap so
that the cross-spectrum is sizeable.

.. plot::
   :context:
   :nofigs:
   :include-source: true

   import camb
   import numpy as np
   from camb.sources import GaussianSourceWindow

   bins = [(0.5, 0.05), (0.6, 0.05)]  # (mean redshift, width)
   lmax = 200

   params = camb.set_params(
       H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.0, omk=0, tau=0.06,
       As=2e-9, ns=0.965, lmax=lmax,
   )
   params.NonLinear = camb.model.NonLinear_none
   params.Want_CMB = False
   params.SourceTerms.limber_windows = False
   for term in ("redshift", "lensing", "velocity", "radial", "timedelay",
                "ISW", "potential"):
       setattr(params.SourceTerms, f"counts_{term}", False)
   params.SourceWindows = [
       GaussianSourceWindow(redshift=z, sigma=s, source_type="counts", bias=1.0)
       for z, s in bins
   ]
   params.set_accuracy(lSampleBoost=50)  # every multipole, no interpolation
   results = camb.get_results(params)
   camb_cls = results.get_source_cls_dict(raw_cl=True)

Grids and windows
-----------------

Both methods use one pair of logarithmic grids: :math:`k` from
:math:`10^{-5}` to :math:`2\,\mathrm{Mpc}^{-1}`, and the paired
:math:`\chi` grid centred near :math:`2500\,\mathrm{Mpc}`. The growth factor
comes from the ratio of linear power spectra on large scales.

.. plot::
   :context:
   :nofigs:
   :include-source: true

   import jax
   import jax.numpy as jnp
   from fftloggin import SphericalBesselJKernel, forward, get_paired_grids
   from fftloggin.cosmology import double_spherical_bessel_table, unequal_time_kernel

   jax.config.update("jax_enable_x64", True)  # or export JAX_ENABLE_X64=1

   kmin, kmax, n = 1e-5, 2.0, 1200
   dlog = np.log(kmax / kmin) / (n - 1)
   k = kmin * np.exp(dlog * np.arange(n))
   log_kr = np.log(np.sqrt(kmin * kmax) * 2500.0)
   chi = np.asarray(get_paired_grids(k=k, log_kr=log_kr)[0])

   power = camb.get_matter_power_interpolator(
       params, nonlinear=False, hubble_units=False, k_hunit=False, kmax=10, zmax=3
   )
   p0 = power.P(0.0, k)

   inside = chi < results.comoving_radial_distance(3.0)
   z = np.zeros_like(chi)
   z[inside] = results.redshift_at_comoving_radial_distance(chi[inside])
   growth = np.sqrt(power.P(z, 1e-3) / power.P(0.0, 1e-3))

   def window(mean, width):
       nz = np.exp(-((z - mean) ** 2) / (2 * width**2)) / np.sqrt(2 * np.pi) / width
       return np.where(inside, nz * results.h_of_z(z) * growth, 0.0)

   windows = [window(*b) for b in bins]
   pairs = [(0, 0), (0, 1), (1, 1)]
   ells = np.unique(np.geomspace(2, lmax, 25).astype(int))

Bins first
----------

For each bin, :math:`F_a(k) = \int d\chi\, w_a(\chi)\, j_\ell(k\chi)` is one
forward transform, since ``forward`` returns
:math:`k \int w_a(\chi) j_\ell(k\chi)\, d\chi`. The remaining integral is a
sum on the :math:`k` grid.

.. plot::
   :context:
   :nofigs:
   :include-source: true

   def cls_bins_first(ell):
       kernel = SphericalBesselJKernel(float(ell))
       f = [np.asarray(forward(w, kernel, dlog=dlog, log_kr=log_kr)) / k
            for w in windows]
       return [2 / np.pi * dlog * np.sum(k**3 * p0 * f[a] * f[b]) for a, b in pairs]

k first
-------

The unequal-time kernel takes :math:`a(k) = (2/\pi) k^2 P_0(k)` and is
contracted with each pair of windows over a band of ``half_width = 100``
grid points, :math:`|\ln(\chi'/\chi)| \le 0.96`, which covers both bins.
With ``bias = 0.5``, :math:`a(k)\, k^{-0.5}` decays at both ends of the
:math:`k` grid.

The table's convolution uses the same frequencies as the transforms, so the
contraction on the grid equals the bins-first sum to rounding.

.. plot::
   :context:
   :nofigs:
   :include-source: true

   bias, half_width = 0.5, 100
   rows = np.arange(half_width, n - half_width)
   partners = rows[:, None] + np.arange(-half_width, half_width + 1)

   def cls_k_first(ell):
       table = double_spherical_bessel_table(
           float(ell), n, dlog=dlog, bias=bias, half_width=half_width
       )
       kern = np.asarray(unequal_time_kernel(
           2 / np.pi * k**2 * p0, table, dlog=dlog, bias=bias, log_kr=log_kr
       ))[rows]
       return [dlog**2 * np.einsum("i,it,it->", windows[a][rows], kern,
                                   (windows[b] * chi)[partners])
               for a, b in pairs]

   keys = ["W1xW1", "W1xW2", "W2xW2"]
   reference = np.array([[camb_cls[key][ell] for key in keys] for ell in ells])
   bins_first = np.array([cls_bins_first(ell) for ell in ells])
   k_first = np.array([cls_k_first(ell) for ell in ells])

Comparison
----------

.. plot::
   :context: close-figs
   :caption: Top: angular power spectra from CAMB (lines) and from the
             k-first unequal-time kernel (dots). Middle: fractional
             difference of each FFTLog method from CAMB; the two methods
             coincide. Bottom: absolute fractional difference between the
             k-first and bins-first results.
   :alt: Three-panel plot against multipole: angular power spectra for two
         auto-spectra and one cross-spectrum, residuals relative to CAMB,
         and the tiny difference between the two FFTLog methods.

   import matplotlib.pyplot as plt

   fig, (top, middle, bottom) = plt.subplots(
       3, 1, figsize=(6.5, 7), sharex=True, height_ratios=(2, 1, 1)
   )
   scale = ells * (ells + 1) / (2 * np.pi)
   labels = ["1 × 1", "1 × 2", "2 × 2"]
   for column, (label, color) in enumerate(zip(labels, ["C0", "C1", "C2"])):
       top.plot(ells, scale * reference[:, column], color=color, label=label)
       top.plot(ells, scale * k_first[:, column], "o", color=color, ms=3)
       middle.plot(ells, bins_first[:, column] / reference[:, column] - 1,
                   color=color, lw=2, alpha=0.4)
       middle.plot(ells, k_first[:, column] / reference[:, column] - 1,
                   "o", color=color, ms=3)
       bottom.plot(ells, np.abs(k_first[:, column] / bins_first[:, column] - 1),
                   "o-", color=color, ms=3, lw=1)
   top.set_xscale("log")
   top.set_yscale("log")
   top.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
   top.legend(title="bins")
   middle.axhline(0, color="k", lw=0.8)
   middle.set_ylim(-0.12, 0.02)
   middle.set_ylabel("vs CAMB")
   bottom.set_yscale("log")
   bottom.set_ylabel("k first vs\nbins first")
   bottom.set_xlabel(r"$\ell$")
   for ax in (top, middle, bottom):
       ax.grid(alpha=0.3)
   fig.tight_layout()

The two FFTLog methods agree to better than :math:`10^{-9}` at every
multipole, because on the same grid they compute the same discrete sum.

Both agree with CAMB to about 0.2% for :math:`\ell \gtrsim 20`, where the
gap is at the level of CAMB's default accuracy settings. Toward
:math:`\ell = 2` the difference grows to 2% in the auto-spectra and 10% in
the smaller cross-spectrum. That gap is shared by both methods, so it comes
from the separable model :math:`D(\chi) D(\chi') P_0(k)`, not from either
transform. The growth factor is scale-independent to :math:`2\times10^{-4}`
over the relevant wavenumbers, so the likely source is how CAMB evaluates the
density source term on scales approaching the horizon.
