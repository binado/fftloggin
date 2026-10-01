Logarithmic grids
=================

Every FFTLog transform maps samples on one logarithmic grid to samples on
another. This page describes how the two grids are related and which helpers
build them. Symbols follow Hamilton (2000); see the notation table in
:doc:`concepts`.

Input and output grids
----------------------

FFTLog samples the input at :math:`N` points evenly spaced in :math:`\ln r`
and returns the output at :math:`N` points evenly spaced in :math:`\ln k`,
with the **same** spacing :math:`L/N` (Hamilton §B.3):

.. math::

   r_n = r_0\, e^{nL/N}, \qquad k_n = k_0\, e^{nL/N}.

Both grids run in increasing order. Only three numbers are free: the spacing
:math:`L/N` (``dlog``), the input centre :math:`r_0`, and the product of the
centres, :math:`k_0 r_0`, whose logarithm is ``log_kr``. Given the input grid
and ``log_kr``, the output grid is fixed.

A useful way to read the relation: points mirrored about the centres always
multiply to the same value,

.. math::

   k_n\, r_{-n} = k_0 r_0 = e^{\texttt{log\_kr}} .

The largest :math:`k` pairs with the smallest :math:`r`, as expected for a
transform whose kernel depends only on :math:`kr`. In ``fftloggin``'s
0-based indexing this reads ``k[j] * r[N - 1 - j] == exp(log_kr)``.

.. plot::
   :caption: A nine-point input grid r and its paired output grid k for
             log_kr = ln 10. Grey lines join mirrored points, whose product
             is always exp(log_kr), so every line crosses at sqrt(k0 r0).
             Open markers show the centres r0 and k0.
   :alt: Two rows of logarithmically spaced points, r and k, joined by lines
         connecting mirrored points.

   import matplotlib.pyplot as plt
   import numpy as np

   from fftloggin import get_paired_grids

   r = np.geomspace(1e-2, 1.0, 9)
   log_kr = np.log(10.0)
   _, k = get_paired_grids(r=r, log_kr=log_kr)
   k = np.asarray(k)

   fig, ax = plt.subplots(figsize=(8, 2.4))
   for rj, kj in zip(r[::-1], k):
       ax.plot([rj, kj], [1, 0], color="0.8", lw=0.8, zorder=0)
   ax.plot(r, np.ones_like(r), "o", color="C0", label=r"$r_n$")
   ax.plot(k, np.zeros_like(k), "s", color="C1", label=r"$k_n$")
   r0, k0 = np.sqrt(r[0] * r[-1]), np.sqrt(k[0] * k[-1])
   ax.plot([r0], [1], "o", ms=14, mfc="none", color="C0")
   ax.plot([k0], [0], "s", ms=14, mfc="none", color="C1")
   ax.annotate(r"$r_0$", (r0, 1), xytext=(0, 12), textcoords="offset points",
               ha="center")
   ax.annotate(r"$k_0 = e^{\mathrm{log\_kr}} / r_0$", (k0, 0), xytext=(0, -22),
               textcoords="offset points", ha="center")
   ax.set_xscale("log")
   ax.set_yticks([0, 1], ["$k$", "$r$"])
   ax.set_ylim(-0.6, 1.5)
   ax.spines[["left", "right", "top"]].set_visible(False)
   fig.tight_layout()

Building grids
--------------

:func:`~fftloggin.grids.infer_dlog` reads the spacing from a grid and checks
that it really is uniform in :math:`\ln r`.
:func:`~fftloggin.grids.get_paired_grids` builds the missing grid from the
one you have. It always returns ``(r, k)``, whichever one you pass:

.. code-block:: python

   import jax.numpy as jnp
   from fftloggin import get_paired_grids, infer_dlog

   r = jnp.geomspace(1e-2, 1e2, 64)
   dlog = infer_dlog(r)
   _, k = get_paired_grids(r=r, log_kr=0.0)

With ``log_kr=0`` and an input grid centred on :math:`r_0 = 1`, the output
grid also spans :math:`10^{-2}` to :math:`10^{2}`. Changing ``log_kr`` slides
the output grid along :math:`\ln k` without changing its spacing or length.

Choosing log_kr
---------------

Usually you know the range of :math:`k` you need rather than the product of
centres. :func:`~fftloggin.grids.infer_log_kr` converts one anchor on the
output grid into ``log_kr``:

- ``ycenter``: the output centre :math:`k_0`;
- ``ymax``: the largest output value, which pairs with ``r[0]``;
- ``ymin``: the smallest output value, which pairs with ``r[-1]``.

.. code-block:: python

   from fftloggin import infer_log_kr

   log_kr = infer_log_kr(r, ymax=50.0)
   _, k = get_paired_grids(r=r, log_kr=log_kr)  # k[-1] == 50

The output range is always as wide as the input range in :math:`\ln`, so
fixing one end fixes the other.

If you also want low ringing, pass the anchored value through
:func:`~fftloggin.fftlog.lowring_log_kr`. It moves ``log_kr`` by at most half
a notch, so the anchor moves by at most a factor :math:`e^{\texttt{dlog}/2}`.
Build the paired grid from the *snapped* value, and pass the same value to the
transform:

.. code-block:: python

   from fftloggin import BesselJKernel, forward, lowring_log_kr

   kernel = BesselJKernel(0.0)
   log_kr = lowring_log_kr(kernel, dlog=dlog, log_kr=infer_log_kr(r, ymax=50.0))
   _, k = get_paired_grids(r=r, log_kr=log_kr)
   samples = r * jnp.exp(-r**2 / 2)
   result = forward(samples, kernel, dlog=dlog, log_kr=log_kr)

Padding
-------

FFTLog treats its input as periodic, so samples at one end of the array leak
into the output at the other end. The error is largest near the ends of the
output grid. :class:`~fftloggin.grids.Padding` moves the periodic images apart:
pad the samples, transform with a plan built for the padded length, and crop
the result back.

.. code-block:: python

   from fftloggin import Padding, plan

   padding = Padding(r.size // 2)
   padded = padding(samples)                       # zeros at both ends
   p = plan(kernel, padded.shape[0], dlog=dlog, log_kr=log_kr)
   result = padding.crop(forward(padded, p))       # on get_paired_grids(r=r)

The padding is the same at both ends, so the padded grid keeps the spacing and
the centre of ``r``. ``dlog``, ``bias`` and ``log_kr``, including a low-ringing
value, carry over unchanged, and the cropped result lies on the original paired
grid.

Zero padding suits inputs that vanish beyond the grid. If the input does not,
such as a lensing window at small :math:`\chi`, zeros would introduce a step of
their own. Evaluate the input on the extended grid instead:

.. code-block:: python

   chi_ext = padding.grid(chi)
   p = plan(kernel, chi_ext.size, dlog=dlog, log_kr=log_kr)
   result = padding.crop(forward(window(chi_ext), p))

Padding only removes wrap-around between the two ends. Ringing from a step or
a kink inside the input, such as a window with a sharp edge, remains.

Tapering
--------

:func:`~fftloggin.grids.taper` returns weights that take the samples smoothly
to zero at the edges of their support. Each ramp has a length ``width`` in
:math:`\ln x`, so the smoothing scale does not depend on the number of
samples. Multiply the weights into the samples before padding:

.. code-block:: python

   from fftloggin import taper

   w = window(chi) * taper(chi, 0.05, hi=chi_max, side="hi")
   result = padding.crop(forward(padding(w), p))

The edges default to the ends of the grid passed in, which is right for the
unpadded grid. On an extended grid, the ends lie beyond the support, so pass
the edge where the samples are not smooth. For example, smooth only the kink
of the CMB-lensing window at :math:`\chi_*`:

.. code-block:: python

   chi_ext = padding.grid(chi)
   w = window(chi_ext) * taper(chi_ext, 0.05, hi=chi_star, side="hi")
   result = padding.crop(forward(w, p))

``shape="cosine"`` (the default) is continuous with its first derivative;
``shape="planck"`` is smooth to all orders. Tapering changes the input, so it
trades ringing for a bias in the result. Keep ``width`` small compared with
the scale over which the samples vary.

The figure below transforms :math:`a(r) = r` on :math:`r \le 1`, which steps
to zero at the last sample, with :math:`\mu = 0` and padding. Without a taper
the step rings across the whole output; a taper with ``width=0.3`` lowers the
error by three orders of magnitude or more.

.. plot::
   :caption: Left: the padded input near its upper edge, without a taper and
             with each ramp shape. Right: absolute error of the cropped
             transform against the exact transform of the same input:
             J1(k) without a taper, and quadrature of the tapered input
             otherwise. The taper changes the input, so the tapered curves
             measure ringing, not the bias from tapering.
   :alt: A step-edged input and its tapered versions, and the transform
         errors, which drop from about 1e-2 without a taper to below 1e-4
         with one.

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import numpy as np
   from scipy.integrate import simpson
   from scipy.special import j0, j1

   from fftloggin import (
       BesselJKernel,
       Padding,
       forward,
       get_paired_grids,
       infer_dlog,
       lowring_log_kr,
       plan,
       taper,
   )

   jax.config.update("jax_enable_x64", True)

   n = 512
   r = jnp.logspace(-4.0, 0.0, n)
   dlog = infer_dlog(r)
   kernel = BesselJKernel(0.0)
   log_kr = lowring_log_kr(kernel, dlog=dlog)
   _, k = get_paired_grids(r=r, log_kr=log_kr)
   k = np.asarray(k)
   padding = Padding(n // 2)
   p = plan(kernel, n + 2 * padding.width, dlog=dlog, log_kr=log_kr)
   log_r_ext = np.log(padding.grid(r))

   width = 0.3
   r_dense = np.linspace(0.0, 1.0, 2**17 + 1)[1:]
   every = slice(None, None, 4)

   fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
   untapered = np.asarray(padding.crop(forward(padding(r), p)))
   left.plot(log_r_ext, padding(r), color="C3", label="no taper")
   right.loglog(k, np.abs(untapered - j1(k)), color="C3", label="no taper")
   for shape, color in [("cosine", "C0"), ("planck", "C2")]:
       samples = r * taper(r, width, side="hi", shape=shape)
       result = np.asarray(padding.crop(forward(padding(samples), p)))
       integrand = r_dense * np.asarray(
           taper(r_dense, width, side="hi", shape=shape)
       )
       exact = np.array(
           [q * simpson(integrand * j0(q * r_dense), x=r_dense) for q in k[every]]
       )
       left.plot(log_r_ext, padding(samples), color=color, label=shape)
       right.loglog(k[every], np.abs(result[every] - exact), color=color,
                    label=shape)
   left.set_xlim(-2.0, 0.5)
   left.set_ylim(0.0, 1.05)
   left.set_xlabel(r"$\ln r$")
   left.set_ylabel(r"$a(r)$")
   left.legend(loc="upper left")
   right.set_xlabel(r"$k$")
   right.set_ylabel("absolute error")
   right.legend()
   fig.tight_layout()

Helpers inside JAX
------------------

``get_paired_grids``, ``infer_log_kr``, ``Padding``, ``taper`` and
:func:`~fftloggin.grids.get_array_center` are pure JAX and work under ``jit``
and ``vmap``. ``infer_dlog`` checks concrete values in Python and must run
outside JAX transformations. If you already know ``dlog`` from how the grid
was built, pass it directly instead.
