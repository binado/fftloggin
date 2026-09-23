Quickstart
==========

This page transforms a function whose Hankel transform is known exactly, so
you can see both how to call ``fftloggin`` and how accurate the result is.

For :math:`\mu = 0`, the forward transform
:math:`\tilde{A}(k) = \int_0^\infty A(r)\, J_0(kr)\, k\, dr` maps

.. math::

   A(r) = r\, e^{-r^2/2} \quad\longmapsto\quad \tilde{A}(k) = k\, e^{-k^2/2}.

A first transform
-----------------

The examples assume 64-bit JAX. Set this before starting Python:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

JAX defaults to 32-bit floats. In 32-bit, the peak error below grows to
about :math:`10^{-6}`, and rounding in :math:`\ln r`
can make :func:`~fftloggin.grids.infer_dlog` reject a wide grid as not
uniform. If that happens, compute the spacing directly as
``jnp.log(r[-1] / r[0]) / (len(r) - 1)``.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, forward, get_paired_grids, infer_dlog

   r = jnp.geomspace(1e-4, 1e4, 256)
   samples = r * jnp.exp(-r**2 / 2)

   dlog = infer_dlog(r)
   kernel = BesselJKernel(0.0)
   log_kr = 0.0

   _, k = get_paired_grids(r=r, log_kr=log_kr)
   result = jax.jit(forward)(samples, kernel, dlog=dlog, log_kr=log_kr)
   exact = k * jnp.exp(-k**2 / 2)

Here is what each piece does:

- ``r`` must be evenly spaced in :math:`\ln r`. :func:`~fftloggin.grids.infer_dlog`
  checks this and returns the spacing ``dlog``.
- ``kernel`` selects the integral kernel, here :math:`J_0`.
- ``log_kr`` is the logarithm of the product of the input and output grid
  centres. With ``log_kr=0`` and an input centred on :math:`r = 1`, the output
  grid covers the same range, :math:`10^{-4} \le k \le 10^{4}`.
- :func:`~fftloggin.grids.get_paired_grids` returns ``(r, k)``. ``result[i]``
  is the transform at ``k[i]``.

The input and output have the same length. :func:`~fftloggin.fftlog.forward`
takes one real sample array; use ``jax.vmap`` to batch it
(see :doc:`/user_guide/jax`).

How accurate is it?
-------------------

.. plot::
   :caption: Top: FFTLog (dots, every fourth sample) against the exact
             transform (line). Bottom: absolute error for log_kr = 0 and for
             the nearest low-ringing value.
   :alt: FFTLog and exact Hankel transform of r exp(-r^2/2), with absolute
         errors for two choices of log_kr below.

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import numpy as np

   from fftloggin import (
       BesselJKernel, forward, get_paired_grids, infer_dlog, lowring_log_kr,
   )

   jax.config.update("jax_enable_x64", True)

   r = jnp.geomspace(1e-4, 1e4, 256)
   samples = r * jnp.exp(-r**2 / 2)
   dlog = infer_dlog(r)
   kernel = BesselJKernel(0.0)

   fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True,
                                     height_ratios=[2, 1])
   choices = [("log_kr = 0", 0.0, "C0"),
              ("low-ringing", lowring_log_kr(kernel, dlog=dlog), "C1")]
   for label, log_kr, color in choices:
       _, k = get_paired_grids(r=r, log_kr=log_kr)
       result = forward(samples, kernel, dlog=dlog, log_kr=log_kr)
       k, result = np.asarray(k), np.asarray(result)
       exact = k * np.exp(-k**2 / 2)
       bottom.loglog(k, np.abs(result - exact), color=color, label=label)
       if log_kr == 0.0:
           top.semilogx(k, exact, color="0.3", label="exact")
           top.semilogx(k[::4], result[::4], "o", ms=3, color=color,
                        label="FFTLog")
   top.set_ylabel(r"$\tilde A(k)$")
   top.legend()
   bottom.set_ylabel("absolute error")
   bottom.set_xlabel("$k$")
   bottom.legend(fontsize=8)
   fig.tight_layout()

Near the peak the error is about :math:`2 \times 10^{-7}` with
``log_kr=0``, and about :math:`10^{-8}` with the low-ringing value introduced
below. Toward either end of the grid it grows to a few times
:math:`10^{-5}`, whichever ``log_kr`` you choose. The reason is that FFTLog
treats the input as periodic in :math:`\ln r`. As :math:`r \to 0` the input
behaves like :math:`r`, so its value at the first sample, about
:math:`10^{-4}`, does not match the zero at the other end, and the periodic
copies meet with a small jump. Extending the grid by a decade at each end
shrinks the jump, and the edge error, by a factor of ten. For inputs whose
ends are far from zero, the ``bias`` parameter is the better tool;
:ref:`what-the-bias-does` shows an example.

Reducing ringing
----------------

The orange curve above uses the nearest *low-ringing* value of ``log_kr``
instead of zero. It removes an artefact at the highest frequency the grid can
represent, and is the better default for most inputs:

.. code-block:: python

   from fftloggin import lowring_log_kr

   log_kr = lowring_log_kr(kernel, dlog=dlog)
   _, k = get_paired_grids(r=r, log_kr=log_kr)
   result = forward(samples, kernel, dlog=dlog, log_kr=log_kr)

Always build the output grid from the same ``log_kr`` that you pass to the
transform. :ref:`low-ringing` explains where the value comes from and why it
should be chosen once, outside any function you differentiate.

Next steps
----------

- :doc:`/user_guide/concepts` explains the algorithm, the bias, and low
  ringing.
- :doc:`/user_guide/tutorial` computes a cosmological correlation function
  from a matter power spectrum.
