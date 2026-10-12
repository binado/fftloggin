The Mellin transform and FFTLog
===============================

FFTLog evaluates an integral by transforming the input in logarithmic
coordinates, multiplying by the Mellin transform of its kernel, and
transforming back. This page develops that connection, including the signs,
normalization, and grid conventions used by ``fftloggin``. The
:doc:`concepts` page continues with Hamilton's presentation of FFTLog,
including bias selection and low ringing; :doc:`kernels` describes the
kernel interface.

The Mellin transform
---------------------

For a function on the positive real axis, the Mellin transform is

.. math::

   \mathcal M[f;s] = \int_0^\infty f(t)\,t^{s-1}\,dt.

The variable :math:`s=c+i\eta` is complex. The integral generally converges
only in a *strip* :math:`a<\operatorname{Re}s<b`, determined by the behavior
of :math:`f` near zero and infinity. The transform consists of both its
expression and its convergence strip: continuing the expression outside
that strip need not preserve its interpretation as the original integral.

Fourier transforms describe oscillations under translation. Mellin
transforms describe power-law modes under scaling. For example, changing
variables gives, for :math:`\alpha>0`,

.. math::

   \mathcal M[f(\alpha\,\cdot);s]
   = \alpha^{-s}\mathcal M[f;s].

Scaling the input multiplies its Mellin transform by a power of the scale.
Along a vertical line, :math:`t^{i\eta}=e^{i\eta\ln t}` is an oscillation
in :math:`\ln t`; :math:`c` sets the accompanying power-law weight.

Two examples are

.. math::

   \mathcal M[e^{-t};s]=\Gamma(s),
   \qquad \operatorname{Re}s>0,

.. math::

   \mathcal M[J_\mu;s]
   =2^{s-1}\frac{\Gamma((\mu+s)/2)}{\Gamma((\mu+2-s)/2)},
   \qquad -\mu<\operatorname{Re}s<\frac32
   \quad (\mu\geq0).

The upper bound in the Bessel example allows conditional convergence from
oscillation; absolute convergence has a smaller strip. We return to this
example under `Hamilton's Bessel kernel`_ below.

Under the conditions for Mellin inversion, the original function is
recovered by

.. math::

   f(t)=\frac{1}{2\pi i}
   \int_{c-i\infty}^{c+i\infty}
   t^{-s}\mathcal M[f;s]\,ds.

The contour is a vertical line inside the convergence strip. When inversion
is valid, moving it within that strip does not change the recovered
function. This continuous statement motivates the choice of bias in FFTLog;
the finite, periodic problem also depends on that choice.

Computing the discretized Mellin transform with FFT
----------------------------------------------------

Set :math:`t=e^x` and :math:`s=c+i\eta`. Since :math:`dt=e^x dx`,

.. math::

   \mathcal M[f;c+i\eta]
   =\int_{-\infty}^{\infty}
   \underbrace{e^{cx}f(e^x)}_{g(x)}\,e^{+i\eta x}\,dx.

Thus Mellin transformation along a fixed vertical line is Fourier
transformation of a weighted function of :math:`\ln t`. This is why
logarithmically spaced input coordinates are useful.

Signs and sampling
~~~~~~~~~~~~~~~~~~

Use the usual FFT conventions,

.. math::

   \operatorname{FFT}(g)_m
   =\sum_{j=0}^{N-1}g_j e^{-2\pi i m j/N},
   \qquad
   \operatorname{IFFT}(G)_j
   =\frac1N\sum_m G_m e^{+2\pi i m j/N}.

The positive exponent in Mellin evaluation therefore matches an inverse
FFT. Sample :math:`x_j=x_{\min}+j\Delta x` and define

.. math::

   L=N\Delta x,\qquad
   \eta_m=\frac{2\pi m}{L},\qquad
   \Delta\eta=\frac{2\pi}{L}.

Here :math:`m` denotes the signed frequency index in FFT order, and
:math:`\Delta x` is ``dlog``. A rectangular quadrature over one period gives

.. math::

   \mathcal M[f;c+i\eta_m]
   \approx N\Delta x\,e^{i\eta_m x_{\min}}
   \operatorname{IFFT}(g)_m,
   \qquad g_j=e^{cx_j}f(e^{x_j}).

The factor :math:`e^{i\eta_m x_{\min}}` accounts for the origin of the
logarithmic grid. The DFT itself is an exact operation on the finite array.
Interpreting it as the continuous integral requires control of truncation
and sampling errors. The associated Fourier interpolant is periodic with
period :math:`L` and has finitely many modes. Its repeated samples extend
one spacing beyond the last sample, so :math:`L=N\Delta x`, rather than
:math:`(N-1)\Delta x`.

The weighted samples determine which function is continued periodically.
If their endpoints join poorly, the periodic continuation has a jump and
produces ringing. Changing the weight changes that continuation, even
though an admissible contour change leaves the continuous answer unchanged.

Reconstruction and normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mellin evaluation and inverse Mellin reconstruction are different
operations. Substituting :math:`ds=i\,d\eta` into the inverse formula gives

.. math::

   f(e^x)=\frac{e^{-cx}}{2\pi}
   \int_{-\infty}^{\infty}
   \mathcal M[f;c+i\eta]\,e^{-i\eta x}\,d\eta.

The :math:`i` cancels from :math:`1/(2\pi i)`, leaving the Fourier inversion
normalization. Reconstruction has a *negative* exponent. It can be
implemented with a forward FFT, or with an inverse FFT and the appropriate
coordinate reversal and phase. FFTLog uses the latter arrangement below.

When the two continuous integrals are replaced by sums, the input spacing
and reconstruction spacing combine to give

.. math::

   \Delta x\,\frac{\Delta\eta}{2\pi}=\frac1N.

Consequently an unnormalized forward FFT followed by a normalized inverse
FFT already supplies this factor. No additional :math:`1/(2\pi i)` needs
to appear in FFTLog's coefficient multiplication.

Hankel transforms and FFTLog
-----------------------------

Consider the kernel integral

.. math::

   I(\lambda)=\int_0^\infty f(t)h(\lambda t)\,dt,
   \qquad \lambda>0.

Bertrand, Bertrand, and Ovarlez, *The Mellin Transform*, Eq. (2.148), express
this as

.. math::
   :label: mellin-kernel-integral

   I(\lambda)=\frac{1}{2\pi i}
   \int_{c-i\infty}^{c+i\infty}
   \lambda^{-s}\mathcal M[h;s]\mathcal M[f;1-s]\,ds.

This identity follows by Mellin inversion of :math:`h` and exchanging the
integrals, when justified. The contour must satisfy both
:math:`c\in\operatorname{strip}(h)` and
:math:`1-c\in\operatorname{strip}(f)`. It reconstructs the output from a
*pointwise product in Mellin space*.

The convolution occurs in logarithmic coordinates. With
:math:`F(x)=e^x f(e^x)` and :math:`H(y)=h(e^y)`,

.. math::

   I(e^y)=\int_{-\infty}^{\infty}F(x)H(y+x)\,dx
         =\int_{-\infty}^{\infty}F(-u)H(y-u)\,du.

This is a convolution with the reflected input :math:`F(-u)`. The reflection
comes from the product :math:`\lambda t` in the original kernel.

From the identity to the algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose :math:`c=1+q`, where :math:`q` is ``bias``. Then

.. math::

   \mathcal M[f;1-s]
   =\int_{-\infty}^{\infty}
   f(e^x)e^{-qx}e^{-i\eta x}\,dx,
   \qquad s=1+q+i\eta.

This factor has the *negative* exponent of a forward FFT. The input FFT
supplies its discrete Fourier coefficients, while the kernel supplies
:math:`\mathcal M[h;s]`. Together with the grid phases, bias weights, and
normalized inverse FFT, the full pipeline realizes the discretized Mellin
identity; the raw input FFT alone does not include the spacing and origin
factors needed to approximate :math:`\mathcal M[f;1-s]`.
``fftloggin`` returns

.. math::

   A(\lambda)=\lambda I(\lambda).

Hamilton (2000), Eq. (158), denotes the input and output by :math:`A(r)` and
:math:`\tilde A(k)`; here they are :math:`f(t)` and :math:`A(\lambda)`.

Multiplying :eq:`mellin-kernel-integral` by :math:`\lambda` leaves the output
weight :math:`\lambda^{-q}` and the negative-exponent modes
:math:`e^{-i\eta\ln\lambda}`. Divide the returned samples by their output
coordinates to obtain :math:`I` itself.

.. list-table:: Ingredients of the forward transform
   :header-rows: 1
   :widths: 30 70

   * - Step
     - Role in Bertrand et al., Eq. (2.148)
   * - Weight the input
     - :math:`f(t)t^{-q}` selects :math:`\operatorname{Re}s=1+q`.
   * - Forward FFT
     - Supplies discrete Fourier coefficients corresponding to
       :math:`\mathcal M[f;1-s_m]` at :math:`s_m=1+q+i\eta_m`, with spacing
       and origin factors accounted for by the full pipeline.
   * - Multiply coefficients
     - Supplies :math:`\mathcal M[h;s_m]` and the paired-grid origin phase.
   * - Inverse FFT and reversal
     - Reconstructs the negative-exponent dependence on :math:`\ln\lambda`.
   * - Weight the output
     - Restores :math:`\lambda^{-q}` for the convention :math:`A=\lambda I`.

For the actual centered grids, let

.. math::

   z_j=\left(j-\frac{N-1}{2}\right)\Delta x,\qquad
   t_j=t_0e^{z_j},\qquad \lambda_j=\lambda_0e^{z_j},\qquad
   \kappa=\ln(t_0\lambda_0)=\texttt{log\_kr}.

The following pseudocode makes the phases and weights explicit. ``a`` holds
the real samples :math:`f(t_j)`, and ``kernel(s)`` evaluates
:math:`\mathcal M[h;s]`:

.. code-block:: python

   n = len(a)
   z = (jnp.arange(n) - (n - 1) / 2) * dlog
   eta = 2 * jnp.pi * jnp.arange(n // 2 + 1) / (n * dlog)
   weight = jnp.exp(-q * z)

   a_tilde = jnp.fft.rfft(a * weight)
   coefficients = kernel(1 + q + 1j * eta) * jnp.exp(-1j * eta * log_kr)
   if n % 2 == 0:
       coefficients = coefficients.at[-1].set(jnp.real(coefficients[-1]))
   reconstructed = jnp.fft.irfft(a_tilde * coefficients, n=n)[::-1]
   result = reconstructed * weight * jnp.exp(-q * log_kr)

Using :math:`(t_j/t_0)^{-q}` in the input weight removes a constant
:math:`t_0^{-q}`. The output factor :math:`e^{-q\kappa}` accounts for it
together with the output-grid center. The coefficient phase
:math:`e^{-i\eta_m\kappa}` accounts for the product of the grid centers.
The reversal uses index :math:`N-1-j`: with this centered convention,
:math:`z_{N-1-j}=-z_j`. The two FFT origin phases combine to give precisely
this reversal, rather than merely reversing the frequency array.

For real input and a real kernel, negative frequencies are conjugates of
positive ones, so ``rfft`` stores only half the spectrum. For even :math:`N`,
the Nyquist frequency represents both signs and its multiplier is replaced
by its real part, as in Hamilton (2000), Eq. (175). Choosing a low-ringing
``log_kr`` makes that multiplier real already; see :ref:`low-ringing`.

FFTLog applies this recipe to a finite Fourier representation of the biased
input. Such a periodically continued power-law series need not have an
ordinary Mellin transform over the whole positive axis. Its coefficients
are instead Fourier-series coefficients, and Hamilton derives their
transforms mode by mode. Eq. (2.148) explains the continuous transform;
the periodic discrete construction supplies the practical algorithm.
Accuracy relative to a nonperiodic integral depends on the range, spacing,
and bias. See :ref:`what-the-bias-does` for examples.

Hamilton's Bessel kernel
~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`h(t)=J_\mu(t)`, Hamilton (2000), Eq. (172), defines

.. math::

   U_\mu(x)=\int_0^\infty t^xJ_\mu(t)\,dt
   =2^x\frac{\Gamma((\mu+1+x)/2)}{\Gamma((\mu+1-x)/2)}
   =\mathcal M[J_\mu;1+x].

The unit shift relates Hamilton's convention to ``BesselJKernel(mu)(s)``.
His Eq. (174) gives the coefficient

.. math::

   u_m=e^{-i\eta_m\kappa}U_\mu(q+i\eta_m)
      =e^{-i\eta_m\kappa}\mathcal M[J_\mu;1+q+i\eta_m].

Thus the pseudocode above is Hamilton's Bessel algorithm with a general
Mellin kernel interface. For :math:`\mu\geq0`, its kernel contour requires
:math:`-(\mu+1)<q<1/2`. The :doc:`kernel guide <kernels>` covers convergence
strips and kernels other than ordinary Bessel functions.

As a concrete example, take :math:`\mu=0` and :math:`f(r)=e^{-r}`. The exact
answer in the library's convention is

.. math::

   A(k)=k\int_0^\infty e^{-r}J_0(kr)\,dr
       =\frac{k}{\sqrt{1+k^2}}.

Enable 64-bit precision before starting Python, for example with
``JAX_ENABLE_X64=1 uv run python``, and use the public API:

.. code-block:: python

   import jax.numpy as jnp
   from fftloggin import (
       BesselJKernel, forward, get_paired_grids, infer_dlog, lowring_log_kr,
   )

   r = jnp.geomspace(1e-4, 1e4, 256)
   dlog = infer_dlog(r)
   kernel = BesselJKernel(mu=0.0)
   bias = -0.5
   log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias)
   _, k = get_paired_grids(r=r, log_kr=log_kr)

   result = forward(jnp.exp(-r), kernel, dlog=dlog, bias=bias, log_kr=log_kr)
   exact = k / jnp.sqrt(1 + k**2)
   interior = (k >= 1e-2) & (k <= 1e2)
   max_relative_error = jnp.max(jnp.abs(result[interior] / exact[interior] - 1))
   print(float(max_relative_error))

With 64-bit values, the maximum relative error is about :math:`3\times10^{-6}`
over :math:`10^{-2}\leq k\leq10^2`. This comparison excludes the ends of the
grid, where the finite range and periodic continuation have the largest
effect. The chosen bias makes the
weighted input :math:`r^{1/2}e^{-r}` decay at both ends of the positive axis.

References
~~~~~~~~~~

- J. Bertrand, P. Bertrand, and J.-P. Ovarlez, *The Mellin Transform*, in
  *The Transforms and Applications Handbook* (1995), Section 2.2.5,
  Eq. (2.148), printed p. 23.
  `HAL manuscript <https://hal.science/hal-03152634v1/document>`_.
- A. J. S. Hamilton, "Uncorrelated modes of the non-linear power spectrum",
  *MNRAS* **312**, 257 (2000), Appendix B, Eqs. (158), (172), (174), (175).
  `arXiv:astro-ph/9905191 <https://arxiv.org/abs/astro-ph/9905191>`_.
