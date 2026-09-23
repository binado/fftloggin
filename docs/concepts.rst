Transform conventions
=====================

FFTLog evaluates integral transforms on logarithmically spaced coordinates.
For the Bessel kernel, ``forward`` and ``inverse`` follow the pair

.. math::

   \widetilde{a}(k) = \int_0^\infty J_\mu(kr)\,a(r)\,k\,dr,
   \qquad
   a(r) = \int_0^\infty J_\mu(kr)\,\widetilde{a}(k)\,r\,dk.

The transform is computed from a finite, uniformly logarithmic grid. As with
other FFT methods, finite range and periodic continuation can affect results;
the bias and center offset help control how the sampled function is continued
at the boundaries.

Logarithmic spacing
-------------------

For a positive grid ``x``, ``dlog`` is the constant step in ``log(x)``. Use
``infer_dlog(x)`` to infer and eagerly validate it, or calculate it directly
when grid construction already guarantees uniform logarithmic spacing.
Transforms receive ``dlog`` as a scalar; they do not inspect the coordinate
array.

Grid center and offset
----------------------

``log_kr`` is the logarithm of the product of the geometric centers of the
input and output grids. Paired grids are related by

.. math::

   k_i = \exp(\texttt{log\_kr}) / r_{N-1-i}.

Use :func:`fftloggin.grids.get_paired_grids` to construct either grid from the other.
The paired grids have the same sample count and opposite ordering under the
reciprocal relation.

Bias and ringing
----------------

``bias`` applies a power-law weighting before and after the FFT. It is useful
for functions whose ends do not approach zero at the same rate. The chosen
bias must lie inside the kernel's Mellin strip for the transform integral to
converge; :func:`fftloggin.fftlog.validate_parameters` can check a concrete bias
before tracing.

:func:`fftloggin.fftlog.lowring_log_kr` snaps a requested center to a low-ringing
value. This operation is piecewise constant in ``log_kr``, so use it as a
preprocessing choice rather than inside a differentiable fit.

JAX execution
-------------

The transform functions are pure JAX computations and support ``jit``,
``vmap`` and ``grad``. Their sample length is static from the input array
shape. Parameter validation and grid-spacing inference are eager host-side
helpers; call them outside JAX transformations. See the :doc:`JAX guide <jax>`.
