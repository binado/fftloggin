Quickstart
==========

This example transforms a real sample array defined on a logarithmic radial
grid, then constructs its paired wavenumber grid.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, forward, get_paired_grids, infer_dlog

   r = jnp.geomspace(1e-2, 1e2, 128)
   a = r * jnp.exp(-r**2 / 2)
   dlog = infer_dlog(r)
   kernel = BesselJKernel(mu=0.0)
   log_kr = 0.0

   r, k = get_paired_grids(r=r, log_kr=log_kr)
   transformed = jax.jit(forward)(a, kernel, dlog=dlog, log_kr=log_kr)

``infer_dlog`` checks the grid eagerly. ``forward`` expects one-dimensional
floating-point samples and scalar ``dlog``, ``bias`` and ``log_kr`` values.
The number of input and output samples is the same; use ``jax.vmap`` to map
over scalar parameters. The :doc:`JAX guide </user_guide/jax>` shows batching
and differentiation.

``log_kr`` is the logarithm of the product of the geometric centers of the
paired grids. ``get_paired_grids`` returns ``(r, k)`` in that order, regardless
of which grid you provide.

To choose the traditional low-ringing center, snap ``log_kr`` explicitly and
pass the returned value to both the grid helper and the transform:

.. code-block:: python

   from fftloggin import lowring_log_kr

   log_kr = lowring_log_kr(kernel, dlog=dlog)
   r, k = get_paired_grids(r=r, log_kr=log_kr)
   transformed = forward(a, kernel, dlog=dlog, log_kr=log_kr)

The snap is piecewise constant in ``log_kr``. For a differentiable fit of
``log_kr``, pass the parameter directly to ``forward`` without snapping.
