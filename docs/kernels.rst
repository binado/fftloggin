Mellin kernels
==============

A kernel supplies the Mellin-space coefficients used by the FFTLog transform.
Built-in kernels are frozen JAX pytrees with scalar parameters as data leaves,
so they can be passed to ``jax.jit`` or constructed from values mapped by
``jax.vmap``.

Built-in kernels
----------------

``BesselJKernel(mu)`` represents the ordinary Bessel function of order
``mu``. ``SphericalBesselJKernel(ell)`` represents the spherical Bessel
function of order ``ell``. Their Mellin strips are open intervals, exposed by
the ``domain`` property.

Construct :class:`fftloggin.ShiftedKernel` with a base kernel and offset to
shift its Mellin argument. Construct :class:`fftloggin.Derivative` with a base
kernel and a positive integer order to represent a derivative. Both wrappers
are frozen JAX pytrees.

The transform bias selects the real part of the Mellin argument at which
kernel coefficients are sampled. Check it against ``kernel.domain``; the
:func:`fftloggin.fftlog.validate_parameters` helper emits a warning when the bias is
outside the domain.

Custom kernels
--------------

Subclass :class:`fftloggin.kernels.Kernel` and implement ``__call__(s)`` to return the
kernel's Mellin-space values. Override ``domain`` when the kernel has a
restricted Mellin strip. Custom kernels used as arguments to ``jax.jit`` must
also be registered as JAX pytrees; keep their numeric parameters in scalar
data leaves.

.. autoclass:: fftloggin.Kernel
   :members: domain, __call__, is_in_domain

.. autoclass:: fftloggin.BesselJKernel

.. autoclass:: fftloggin.SphericalBesselJKernel

.. autoclass:: fftloggin.ShiftedKernel

.. autoclass:: fftloggin.Derivative
