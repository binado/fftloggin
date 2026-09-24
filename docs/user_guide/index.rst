User guide
==========

The first page explains the FFTLog algorithm, following Hamilton (2000); the
next three show how ``fftloggin`` exposes each part of it. The tutorial puts
everything together on a real cosmology problem.

- :doc:`concepts`: the transform pair, the algorithm, the bias, and low
  ringing.
- :doc:`grids`: how input and output grids are paired, and how to choose
  ``log_kr``.
- :doc:`kernels`: Mellin transforms, convergence strips, and custom kernels.
- :doc:`jax`: compiling, batching, and differentiating transforms.
- :doc:`unequal_time`: double spherical Bessel integrals for angular power
  spectra beyond Limber.
- :doc:`tutorial`: the matter correlation function from a CAMB power
  spectrum.
- :doc:`tutorial_cl`: angular power spectra beyond Limber, checked against
  CAMB.

.. toctree::
   :maxdepth: 1
   :hidden:

   concepts
   grids
   kernels
   jax
   unequal_time
   tutorial
   tutorial_cl
