JAX transformations
===================

``forward`` and ``inverse`` are functional JAX operations. Compile them with
``jax.jit``, map across scalar parameters with ``jax.vmap``, and differentiate
through kernel parameters or transform scalars with ``jax.grad``.

Compile a transform
-------------------

.. code-block:: python

   import jax
   from fftloggin import BesselJKernel, forward

   compiled = jax.jit(forward)
   result = compiled(samples, BesselJKernel(0.5), dlog=dlog, bias=0.1)

Batch scalar parameters
-----------------------

The transform accepts one sample array at a time. To transform the same input
for multiple Bessel orders, map over the scalar kernel parameter:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, forward

   orders = jnp.array([0.0, 1.0, 2.0])
   results = jax.jit(jax.vmap(
       lambda mu: forward(samples, BesselJKernel(mu), dlog=dlog)
   ))(orders)

The same pattern applies to ``dlog``, ``bias`` and ``log_kr``. Kernel
instances are pytrees, and their numeric scalar fields can be traced values.

Differentiate a parameter
-------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import BesselJKernel, forward

   def loss(mu):
       prediction = forward(samples, BesselJKernel(mu), dlog=dlog)
       return jnp.sum((prediction - target) ** 2)

   gradient = jax.grad(loss)(0.5)

Eager helpers
-------------

:func:`fftloggin.grids.infer_dlog` checks concrete grid values, and
:func:`fftloggin.fftlog.validate_parameters` checks concrete scalar parameters.
Both use host-side Python decisions and must be called outside ``jit``,
``vmap`` and ``grad``. The transform functions themselves do not perform
those host checks automatically.
