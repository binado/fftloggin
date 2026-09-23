fftloggin
=========

``fftloggin`` implements one-dimensional FFTLog transforms with JAX. Its
functional API works with ``jit``, ``vmap`` and ``grad``; its coordinate and
kernel helpers make logarithmic grids and Mellin transforms explicit.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   concepts
   grids
   kernels
   jax
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

The package does not change process-wide JAX configuration when imported. For
high-accuracy numerical work, enable 64-bit values before importing JAX:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

See the :doc:`installation guide <installation>` for installation options and
the :doc:`API reference <api>` for every public symbol.
