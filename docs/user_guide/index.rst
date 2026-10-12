User guide
==========

Start with the Mellin-transform interpretation of FFTLog, then follow
Hamilton's presentation of the algorithm. The remaining guides explain how
``fftloggin`` exposes each part of it, and the tutorials put everything
together on real cosmology problems.

- :doc:`mellin`: the Mellin transform, its computation with FFTs, and the
  kernel-integral identity behind FFTLog.
- :doc:`concepts`: the transform pair, the algorithm, the bias, and low
  ringing.
- :doc:`grids`: how input and output grids are paired, and how to choose
  ``log_kr``.
- :doc:`kernels`: Mellin transforms, convergence strips, and custom kernels.
- :doc:`jax`: compiling, batching, and differentiating transforms.
- :doc:`batching`: mapping transforms over arrays, kernels, or both with
  ``jax.vmap``.
- :doc:`unequal_time`: double spherical Bessel integrals for angular power
  spectra beyond Limber.
- :doc:`tutorial`: the matter correlation function from a CAMB power
  spectrum.
- :doc:`tutorial_cl`: angular power spectra beyond Limber, checked against
  CAMB.

.. toctree::
   :maxdepth: 1
   :hidden:

   mellin
   concepts
   grids
   kernels
   jax
   batching
   unequal_time
   tutorial
   tutorial_cl
