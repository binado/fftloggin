Installation
============

Install ``fftloggin`` from PyPI with pip:

.. code-block:: console

   $ python -m pip install fftloggin

Or add it to a project managed with ``uv``:

.. code-block:: console

   $ uv add fftloggin

The package requires Python 3.11 or newer and installs JAX as a dependency.
For numerical comparisons that need double precision, set
``JAX_ENABLE_X64=1`` before importing JAX or ``fftloggin``:

.. code-block:: console

   $ export JAX_ENABLE_X64=1

On Windows PowerShell, use ``$env:JAX_ENABLE_X64 = "1"`` in the shell before
starting Python. ``fftloggin`` does not modify JAX's process-wide settings.
