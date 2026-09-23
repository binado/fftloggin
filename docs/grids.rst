Logarithmic grids
=================

FFTLog input coordinates must be uniformly spaced in their logarithm. Grid
helpers make the spacing and the paired ``r``/``k`` coordinates explicit.

Infer spacing
-------------

.. autofunction:: fftloggin.grids.infer_dlog

This helper performs eager validation and is not intended to run inside
``jax.jit``. If the grid is created from a known ``dlog``, pass that value
directly to the transform.

Construct paired grids
----------------------

.. autofunction:: fftloggin.grids.get_paired_grids

Exactly one of ``r`` or ``k`` is required. The returned pair is always
``(r, k)``. The relation is pure JAX and can be used inside ``jit`` and
``vmap``.

Grid centers and offsets
------------------------

.. autofunction:: fftloggin.grids.get_array_center

.. autofunction:: fftloggin.grids.infer_log_kr

``infer_log_kr`` accepts one anchor value: ``ycenter``, ``ymax`` or ``ymin``.
It returns the logarithm of the center product for the two paired grids.
