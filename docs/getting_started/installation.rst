Installation
============

Install ``fftloggin`` from PyPI with pip:

.. code-block:: console

   $ python -m pip install fftloggin

or add it to a project managed with ``uv``:

.. code-block:: console

   $ uv add fftloggin

``fftloggin`` requires Python 3.11 or newer and installs JAX as a dependency.
For GPU or TPU support, install the matching JAX build by following the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

Double precision
----------------

JAX computes in 32-bit floats by default, which limits FFTLog to roughly six
significant digits. For numerical work, enable 64-bit mode *before* JAX is
imported:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

On Windows PowerShell, use ``$env:JAX_ENABLE_X64 = "1"``. Alternatively, call
``jax.config.update("jax_enable_x64", True)`` at the top of your script,
before creating any arrays. ``fftloggin`` never changes this setting itself.

Building the documentation
--------------------------

The documentation, including the CAMB tutorial and its figures, is built
with Sphinx from the ``docs`` dependency group:

.. code-block:: console

   $ uv sync --group docs
   $ just docs build
