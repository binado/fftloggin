fftloggin
=========

``fftloggin`` computes Hankel transforms of functions sampled on logarithmic
grids, using the FFTLog algorithm, in pure JAX.

Many integrals in physics have the form

.. math::

   \tilde{A}(k) = \int_0^\infty A(r)\, J_\mu(kr)\, k\, dr ,

with the input spread over many decades in :math:`r`. Examples are the
conversion between a power spectrum and a correlation function in cosmology,
and three-dimensional Fourier transforms of isotropic functions more
generally. Direct quadrature is slow and struggles with the oscillating
Bessel function. FFTLog (Talman 1978; Hamilton 2000) instead treats the
integral as a convolution in :math:`\ln r` and evaluates it with two FFTs, in
:math:`\mathcal{O}(N \log N)` operations.

``fftloggin`` implements the algorithm as it is presented in Appendix B of
`Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191>`_, as pure
functions that work with ``jax.jit``, ``jax.vmap`` and ``jax.grad``. You can
differentiate a transform with respect to its input, the Bessel order, the
power-law bias, or the grid offset.

.. code-block:: python

   import jax.numpy as jnp
   from fftloggin import BesselJKernel, forward, infer_dlog

   r = jnp.geomspace(1e-2, 1e2, 128)
   result = forward(r * jnp.exp(-r**2 / 2), BesselJKernel(0.0), dlog=infer_dlog(r))

``fftloggin`` never changes JAX's global configuration. For
double-precision accuracy, set ``JAX_ENABLE_X64=1`` before starting Python.

Where to start
--------------

- :doc:`getting_started/quickstart` runs a first transform and checks it
  against an exact answer.
- :doc:`user_guide/concepts` explains how FFTLog works, following Hamilton's
  equations.
- :doc:`user_guide/tutorial` computes the cosmological correlation function
  and its BAO peak from a CAMB power spectrum.
- :doc:`reference/api` lists every public function and class.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   reference/index
