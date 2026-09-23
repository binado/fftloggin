fftloggin
=========

``fftloggin`` implements one-dimensional FFTLog transforms with JAX. Its
functional API works with ``jit``, ``vmap`` and ``grad``; its coordinate and
kernel helpers make logarithmic grids and Mellin transforms explicit.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting_started/index
   user_guide/index
   reference/index

The package does not change process-wide JAX configuration when imported. For
high-accuracy numerical work, enable 64-bit values before importing JAX:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

See the :doc:`installation guide <getting_started/installation>` for
installation options and the :doc:`API reference <reference/api>` for every
public symbol.
