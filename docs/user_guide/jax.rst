JAX transformations
===================

:func:`~fftloggin.fftlog.forward` and :func:`~fftloggin.fftlog.inverse` are
pure functions of arrays and scalars, so JAX can compile, batch, and
differentiate them. This opens up uses that a conventional FFTLog code cannot
support: fitting a kernel parameter or bias by gradient descent, computing
many multipoles in one vectorized call, or putting a Hankel transform inside a
differentiable likelihood.

The snippets on this page share this setup, which reuses the Gaussian pair
from the :doc:`/getting_started/quickstart`:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, SphericalBesselJKernel, forward, infer_dlog

   r = jnp.geomspace(1e-4, 1e4, 256)
   dlog = infer_dlog(r)
   samples = r * jnp.exp(-r**2 / 2)

Compile
-------

.. code-block:: python

   compiled = jax.jit(forward)
   result = compiled(samples, BesselJKernel(0.5), dlog=dlog, bias=0.1)

The sample count ``N`` is read from the array shape, so each new length
triggers one recompilation. Kernels, ``dlog``, ``bias`` and ``log_kr`` are
traced values, so changing them does not.

Batch
-----

Each call transforms one one-dimensional array. To transform the same input
for several Bessel orders, map over the kernel parameter. Kernels are pytrees
whose numeric fields may be traced:

.. code-block:: python

   orders = jnp.array([0.0, 1.0, 2.0])
   by_order = jax.jit(jax.vmap(
       lambda mu: forward(samples, BesselJKernel(mu), dlog=dlog)
   ))(orders)                                    # shape (3, 256)

The same pattern computes several spherical-Bessel multipoles at once:

.. code-block:: python

   ells = jnp.array([0.0, 2.0, 4.0])
   multipoles = jax.vmap(
       lambda ell: forward(samples, SphericalBesselJKernel(ell), dlog=dlog)
   )(ells)

To transform a batch of inputs with one kernel, map over the samples:

.. code-block:: python

   batch = jnp.stack([samples, 2 * samples, samples**2])
   results = jax.vmap(
       lambda a: forward(a, BesselJKernel(0.0), dlog=dlog)
   )(batch)

``dlog``, ``bias`` and ``log_kr`` can be mapped the same way. Every value of
``bias`` must lie in the kernel's convergence strip (see :doc:`kernels`).

Differentiate
-------------

Gradients flow through the samples, the kernel parameters, ``bias`` and
``log_kr``. For example, you can recover a Bessel order from a transformed
target:

.. code-block:: python

   target = forward(samples, BesselJKernel(1.0), dlog=dlog)

   def loss(mu):
       prediction = forward(samples, BesselJKernel(mu), dlog=dlog)
       return jnp.sum((prediction - target) ** 2)

   jax.grad(loss)(0.5)   # negative: increasing mu reduces the loss
   jax.grad(loss)(1.0)   # zero at the true order

Differentiating with respect to :math:`\mu` goes through the Gamma functions
in Hamilton's :math:`U_\mu` (172). ``fftloggin`` gives the log-Gamma function
a custom derivative (the digamma function), because JAX's own ``loggamma``
cannot be reverse-differentiated at complex arguments.

Choosing log_kr outside the gradient
------------------------------------

:func:`~fftloggin.fftlog.lowring_log_kr` works under ``jit``, but it rounds
to the nearest low-ringing value, so its derivative is zero almost everywhere.
If ``log_kr`` is a fit parameter, snap it once before the fit, or pass the
raw parameter to the transform and accept a little ringing.

Eager helpers
-------------

Two helpers check concrete values and must run *outside* ``jit``, ``vmap``
and ``grad``:

- :func:`~fftloggin.grids.infer_dlog` checks that a grid is uniformly spaced
  in :math:`\ln r`;
- :func:`~fftloggin.fftlog.validate_parameters` checks that ``dlog``,
  ``bias`` and ``log_kr`` are finite, and warns when the bias lies outside the
  kernel's strip.

Both turn JAX arrays into Python booleans to decide whether to raise, which
is impossible for traced values. Calling them under ``jit`` raises
``TracerBoolConversionError``. The transforms themselves never run these
checks, so call them once on concrete inputs, then trace freely.
