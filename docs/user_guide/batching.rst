Batching operations with vmap
=============================

:func:`~fftloggin.fftlog.forward` transforms **one** one-dimensional array
with **one** kernel. To transform many arrays, many kernels, or both, map the
call with :func:`jax.vmap`. No special batching functions are needed. This page
shows the three common cases:

1. one array, a batch of kernels (for example an array of multipoles);
2. a batch of arrays, one kernel;
3. a batch of arrays and a batch of kernels.

The snippets share this setup:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from fftloggin import SphericalBesselJKernel, forward, infer_dlog, plan

   r = jnp.geomspace(1e-4, 1e4, 256)
   dlog = infer_dlog(r)
   n = r.shape[0]

   samples = r * jnp.exp(-r**2 / 2)                       # one array, (n,)
   batch = jnp.stack([samples, 2 * samples, samples**2])  # B = 3 arrays, (3, n)
   ells = jnp.array([0.0, 2.0, 4.0])                      # L = 3 multipoles

How batching works
------------------

Kernels are JAX pytrees whose numeric fields are scalars.
``SphericalBesselJKernel(ells)`` is therefore a pytree whose ``ell`` leaf has
shape ``(L,)``, but it is *not* a kernel that ``forward`` understands: the
coefficients are evaluated assuming a scalar ``ell``, and calling
``forward(samples, SphericalBesselJKernel(ells), dlog=dlog)`` fails with a
broadcasting error.

``jax.vmap`` removes the batch axis from every leaf before calling your
function, so inside the mapped function each kernel is an ordinary scalar
kernel. Every recipe below is built from this.

One array, batched kernel
-------------------------

Map over the kernel and keep the array fixed:

.. code-block:: python

   kernels = SphericalBesselJKernel(ells)
   out = jax.vmap(lambda k: forward(samples, k, dlog=dlog))(kernels)
   out.shape                                              # (L, n)

``out[i]`` is the transform of ``samples`` with ``ells[i]``.

Batched array, one kernel
-------------------------

Map over the rows of the array and keep the kernel fixed:

.. code-block:: python

   kernel = SphericalBesselJKernel(2.0)
   out = jax.vmap(lambda a: forward(a, kernel, dlog=dlog))(batch)
   out.shape                                              # (B, n)

When the kernel and grid are shared, evaluate the coefficients once with
:func:`~fftloggin.fftlog.plan` and map only over the arrays
(see :doc:`jax`):

.. code-block:: python

   p = plan(kernel, n, dlog=dlog)
   out = jax.vmap(forward, in_axes=(0, None))(batch, p)   # (B, n)

``in_axes=(0, None)`` maps the first argument over its leading axis and passes
the plan unchanged to every call.

Batched array and batched kernel
--------------------------------

Build one plan per kernel by mapping :func:`~fftloggin.fftlog.plan`, then nest
two maps. Every leaf of the resulting plan carries a leading axis of length
``L``:

.. code-block:: python

   plans = jax.vmap(lambda k: plan(k, n, dlog=dlog))(kernels)

   transform = jax.vmap(forward, in_axes=(0, None))   # rows of batch, one plan
   out = jax.vmap(transform, in_axes=(None, 0))(batch, plans)
   out.shape                                              # (L, B, n)

The inner map runs over the ``B`` arrays for a fixed plan. The outer map runs
over the ``L`` plans with the whole batch fixed. ``out[i, j]`` is the transform
of ``batch[j]`` with ``ells[i]``. Swap the two ``in_axes`` to get the layout
``(B, L, n)`` instead.

Building the plans first computes the kernel coefficients once per kernel
instead of once per array and kernel pair.

If the arrays and kernels are *paired*, so that ``batch[i]`` goes with
``ells[i]``, a single map over both is enough:

.. code-block:: python

   out = jax.vmap(forward)(batch, plans)                  # (B, n), needs B == L

Mapping the grid parameters
---------------------------

``dlog``, ``bias`` and ``log_kr`` are scalars too and can be mapped alongside
the kernel. This is useful when the low-ringing ``log_kr`` depends on the
multipole:

.. code-block:: python

   from fftloggin import lowring_log_kr

   def lowring(k):
       return lowring_log_kr(k, dlog=dlog, bias=0.1)

   log_krs = jax.vmap(lowring)(kernels)
   out = jax.vmap(
       lambda k, log_kr: forward(samples, k, dlog=dlog, bias=0.1, log_kr=log_kr)
   )(kernels, log_krs)                                    # (L, n)

:func:`~fftloggin.fftlog.validate_parameters` and
:func:`~fftloggin.grids.infer_dlog` check concrete values and must be called
*outside* ``jax.vmap``, ``jax.jit`` and ``jax.grad``.

Summary
-------

.. list-table::
   :header-rows: 1

   * - Case
     - Call
     - Output
   * - one array, ``L`` kernels
     - ``vmap(lambda k: forward(a, k, ...))(kernels)``
     - ``(L, n)``
   * - ``B`` arrays, one kernel or plan ``p``
     - ``vmap(forward, in_axes=(0, None))(batch, p)``
     - ``(B, n)``
   * - ``B`` arrays, ``L`` plans (all pairs)
     - ``vmap(vmap(forward, in_axes=(0, None)), in_axes=(None, 0))(batch, plans)``
     - ``(L, B, n)``
   * - ``B`` arrays, ``B`` plans (paired)
     - ``vmap(forward)(batch, plans)``
     - ``(B, n)``

All of these compose with ``jax.jit`` and ``jax.grad``; see :doc:`jax`.
