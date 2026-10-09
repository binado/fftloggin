Mellin kernels
==============

:doc:`concepts` showed that FFTLog needs only one thing from the integral
kernel: the closed-form integral of a power law against it, evaluated along a
vertical line in the complex plane. In ``fftloggin`` that function is a
*kernel* object. This page explains the convention it uses, which kernels are
built in, and how to write your own.

From Hamilton's U to a kernel
-----------------------------

For the Bessel function, Hamilton (2000) defines

.. math::

   U_\mu(x) \equiv \int_0^\infty t^{x} J_\mu(t)\, dt
   = 2^{x}\, \frac{\Gamma[(\mu + 1 + x)/2]}{\Gamma[(\mu + 1 - x)/2]}, \tag{172}

and samples it at :math:`x = q + 2\pi i m/L` (174). A kernel in ``fftloggin``
uses the standard **Mellin transform** convention,

.. math::

   \mathcal{M}[K](s) = \int_0^\infty t^{s - 1} K(t)\, dt ,

so the two differ by a unit shift, :math:`s = 1 + x`:

.. math::

   \texttt{BesselJKernel(mu)(s)} = U_\mu(s - 1).

The transforms therefore evaluate ``kernel(1 + bias + 2j*pi*m/L)``. The Mellin
convention lets the same interface describe kernels other than Bessel
functions, and makes the derivative and shift rules below come out in their
usual form.

The convergence strip
---------------------

A Mellin integral converges only for :math:`\operatorname{Re} s` in an open
interval, the kernel's *strip*. For :math:`J_\mu`, small :math:`t` behaves
like :math:`t^{\mu}` and large :math:`t` like :math:`t^{-1/2}\cos(\cdot)`, so
the integral converges for

.. math::

   -\mu < \operatorname{Re} s < \tfrac{3}{2}
   \qquad\Longleftrightarrow\qquad
   -(\mu + 1) < \operatorname{Re} x < \tfrac{1}{2}.

Every kernel reports its strip in the Mellin variable as ``kernel.domain``.
FFTLog samples along :math:`\operatorname{Re} s = 1 + q`, so the bias must
satisfy ``kernel.is_in_domain(1 + bias)``.
:func:`~fftloggin.fftlog.validate_parameters` performs this check eagerly and
emits a :class:`~fftloggin.exceptions.DomainCheckWarning` if it fails.

.. plot::
   :caption: The complex Mellin plane for mu = 1/2 and q = -0.3. FFTLog
             evaluates the kernel at the dots, which lie on the line
             Re s = 1 + q inside the convergence strip (shaded). Crosses mark
             poles of the numerator Gamma function.
   :alt: Complex plane with the shaded convergence strip of the J_1/2 Mellin
         transform, the sampling line at Re s = 1 + q, and poles to its left.

   import matplotlib.pyplot as plt
   import numpy as np

   mu, q = 0.5, -0.3
   n, dlog = 16, 0.5
   period = n * dlog
   m = np.arange(-n // 2, n // 2 + 1)
   omega = 2 * np.pi * m / period
   lower, upper = -mu, 1.5

   fig, ax = plt.subplots(figsize=(7, 4))
   ax.axvspan(lower, upper, color="C0", alpha=0.12, label="convergence strip")
   ax.axvline(1 + q, color="C0", lw=1.5, label=r"$\mathrm{Re}\,s = 1 + q$")
   ax.plot(np.full_like(omega, 1 + q), omega, "o", color="C0", ms=4,
           label=r"samples $s_m = 1 + q + 2\pi i m / L$")
   poles = -mu - 2 * np.arange(3)
   ax.plot(poles, np.zeros_like(poles), "x", color="C3", ms=8, mew=2,
           label=r"poles of $\Gamma[(\mu + s)/2]$")
   ax.axhline(0, color="0.7", lw=0.8)
   ax.set_xlim(-5, 3)
   ax.set_xlabel(r"$\mathrm{Re}\,s$")
   ax.set_ylabel(r"$\mathrm{Im}\,s$")
   ax.legend(loc="lower left", fontsize=8)
   top = ax.secondary_xaxis("top", functions=(lambda s: s - 1, lambda x: x + 1))
   top.set_xlabel(r"Hamilton's $\mathrm{Re}\,x = \mathrm{Re}\,s - 1$")
   fig.tight_layout()

Built-in kernels
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Kernel
     - :math:`K(t)`
     - Strip in :math:`s`
   * - ``BesselJKernel(mu)``
     - :math:`J_\mu(t)`
     - :math:`(-\mu,\ 3/2)`
   * - ``SphericalBesselJKernel(ell)``
     - :math:`j_\ell(t) = (\pi/2t)^{1/2} J_{\ell+1/2}(t)`
     - :math:`(-\ell,\ 2)`

The spherical Bessel kernel follows from the identity Hamilton quotes in
§B.1. Its Mellin transform is
:math:`\sqrt{\pi/2}\;\mathcal{M}[J_{\ell+1/2}](s - 1/2)`. Use it for
three-dimensional Fourier transforms of isotropic functions, such as the
power spectrum multipoles in :doc:`tutorial`.

Symbolic composition
--------------------

Bind a numerical kernel to a :class:`~fftloggin.symbolic.Coordinate` to
compose real-space operations. Expressions are frozen JAX pytree kernels
and work directly with ``plan``, ``forward``, ``inverse``,
``lowring_log_kr``, and ``product_plan``:

.. code-block:: python

   from fftloggin import BesselJKernel, Coordinate, diff, plan

   t = Coordinate("t")
   j = BesselJKernel(mu=1.5)
   expr = t**0.2 * j(1.7 * t)
   derivative = diff(expr, t, order=2)
   combined = 2 * expr - derivative
   p = plan(combined, 256, dlog=0.1, bias=0.0)

Calling ``expr(s)`` with numerical scalar or array arguments evaluates its
Mellin transform. Calling ``expr(b * t)`` rescales the complete real-space
expression. Coordinates with equal names denote the same variable.
Combining different names raises ``TypeError``; explicitly calling
``expr(Coordinate("u"))`` rebinds the expression.

.. list-table::
   :header-rows: 1

   * - Expression
     - Mellin transform
     - Strip
   * - ``c * j(t)``
     - ``c * M(s)``
     - Base strip
   * - ``j(t) + k(t)``
     - ``M_j(s) + M_k(s)``
     - Intersection
   * - ``t**nu * j(t)``
     - ``M(s + nu)``
     - Base strip shifted by ``-nu``
   * - ``j(a * t)``
     - ``exp(-s * log(a)) * M(s)``
     - Base strip
   * - ``diff(j(t), t, order=n)``
     - ``(-1)**n * product(s-r, r=1,...,n) * M(s-n)``
     - Base strip shifted by ``+n``

Derivative identities assume the boundary terms from repeated integration
by parts vanish. The reported derivative strip preserves this convention.
Sum strips are conservative: cancellations do not widen them and empty
intersections remain empty. Trees stay nested without simplification.

Operation order matters:

.. code-block:: python

   diff(t**2 * j(t), t)  # d/dt [t**2 J(t)]
   t**2 * diff(j(t), t)  # t**2 J'(t)

``diff`` accepts a static nonnegative integer order; order zero returns
the expression unchanged. Coefficients, exponents, and scales must be
real scalars. Argument scales must be strictly positive. Use the eager
``validate_parameters`` helper to check finite expression parameters and
positive scales before tracing. Construction and evaluation support
``jit``, ``vmap``, and ``grad`` without host conversions of parameter values.

Translations, nonlinear arguments, arbitrary kernel products, powers of
complete kernels, and scalar addition to kernels are unsupported.
Coordinates and monomial factors alone are not FFTLog kernels.

Migration from transforms
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Transform`` API and ``Kernel.transform`` have been removed.
With ``t = Coordinate("t")``, migrate calls as follows:

.. list-table::
   :header-rows: 1

   * - Previous API
     - Symbolic API
   * - ``j.transform(PowerLaw(nu))``
     - ``t**nu * j(t)``
   * - ``j.transform(Derivative(n))``
     - ``diff(j(t), t, order=n)``
   * - ``j.transform(Scale(a))``
     - ``j(a * t)``
   * - ``j.transform(PowerLaw(nu), Derivative(n))``
     - ``diff(t**nu * j(t), t, order=n)``
   * - Custom kernel ``__call__(s)``
     - Custom kernel ``mellin(s)``

Writing a custom kernel
-----------------------

FFTLog works for any kernel whose Mellin transform you can write down, not
only Bessel functions. As an example, the Laplace kernel
:math:`K(t) = e^{-a t}` has

.. math::

   \mathcal{M}[K](s) = a^{-s}\, \Gamma(s), \qquad 0 < \operatorname{Re} s < \infty,

so ``forward`` computes :math:`\tilde{A}(k) = k \int_0^\infty e^{-a k r}
A(r)\, dr`, a scaled Laplace transform. To implement it, subclass
:class:`~fftloggin.kernels.Kernel`, return the Mellin transform from
``mellin``, report the strip from ``domain``, and register the class as a
JAX pytree with its numeric parameters as data fields:

.. code-block:: python

   from dataclasses import dataclass
   from functools import partial

   import jax
   import jax.numpy as jnp
   from jax.typing import ArrayLike
   from jax.scipy.special import loggamma
   from jax.tree_util import register_dataclass

   from fftloggin import Kernel


   @partial(register_dataclass, data_fields=("rate",), meta_fields=())
   @dataclass(frozen=True)
   class LaplaceKernel(Kernel):
       """Kernel K(t) = exp(-rate * t), with Mellin transform rate**-s Gamma(s)."""

       rate: ArrayLike

       @property
       def domain(self):
           return jnp.asarray(0.0), jnp.asarray(jnp.inf)

       def mellin(self, s: ArrayLike) -> jax.Array:
           s = jnp.asarray(s)
           return jnp.exp(loggamma(s) - s * jnp.log(self.rate))

Evaluating in log space with ``loggamma`` avoids overflow of :math:`\Gamma(s)`
at large :math:`|\operatorname{Im} s|`. Because ``rate`` is a data field, the
kernel can be passed to ``jax.jit`` and differentiated with respect to its
parameter. Checking against the exact pair :math:`A(r) = e^{-r}`,
:math:`\tilde{A}(k) = k/(k + 1)` for ``rate=1``:

.. code-block:: python

   import jax
   from fftloggin import forward, get_paired_grids, lowring_log_kr

   r = jnp.geomspace(1e-4, 1e4, 256)
   dlog = jnp.log(r[1] / r[0])
   kernel = LaplaceKernel(1.0)
   bias = -0.5
   log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias)
   _, k = get_paired_grids(r=r, log_kr=log_kr)

   result = jax.jit(forward)(jnp.exp(-r), kernel, dlog=dlog, bias=bias, log_kr=log_kr)
   exact = k / (k + 1)

Away from the ends of the grid, the relative error is about
:math:`4 \times 10^{-5}`. With ``bias=0`` it is about :math:`10^{-2}`, because
:math:`e^{-r} \to 1` as :math:`r \to 0` and the periodic copies of the input
meet with a jump (see :ref:`the bias <what-the-bias-does>`).

API
---

.. autoclass:: fftloggin.kernels.Kernel
   :no-index:
   :members: domain, mellin, __call__, is_in_domain

The built-in kernels are listed in the :doc:`API reference </reference/api>`.
