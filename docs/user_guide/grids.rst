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

Helpers inside JAX
------------------

``get_paired_grids``, ``infer_log_kr`` and
:func:`~fftloggin.grids.get_array_center` are pure JAX and work under ``jit``
and ``vmap``. ``infer_dlog`` checks concrete values in Python and must run
outside JAX transformations. If you already know ``dlog`` from how the grid
was built, pass it directly instead.
