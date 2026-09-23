How FFTLog works
================

FFTLog evaluates Hankel transforms of functions sampled on logarithmic grids
in :math:`\mathcal{O}(N \log N)` operations. The algorithm was proposed by
Talman (1978); ``fftloggin`` follows the presentation in Appendix B of
Hamilton (2000), and this page walks through that appendix. **Equation numbers
in parentheses refer to Hamilton (2000)**, so every formula here can be checked
against the paper. A table at the end maps the paper's symbols to the
arguments of ``fftloggin`` functions.

The transform pair
------------------

``fftloggin`` computes the Hankel (Fourier–Bessel) transform pair

.. math::

   \tilde{A}(k) = \int_0^\infty A(r)\, J_\mu(kr)\, k\, dr, \tag{158}

.. math::

   A(r) = \int_0^\infty \tilde{A}(k)\, J_\mu(kr)\, r\, dk, \tag{159}

with :func:`~fftloggin.fftlog.forward` evaluating (158) and
:func:`~fftloggin.fftlog.inverse` evaluating (159). The order :math:`\mu` can be any
real number. With :math:`\mu = \pm 1/2` the pair reduces to Fourier sine and
cosine transforms, since :math:`J_{1/2}(x) = (2/\pi x)^{1/2}\sin x` and
:math:`J_{-1/2}(x) = (2/\pi x)^{1/2}\cos x` (160–161). Spherical Bessel
functions :math:`j_\lambda(x) = (\pi/2x)^{1/2} J_{\lambda+1/2}(x)` are
covered in the same way.

Hamilton also introduces a *biased* pair. Substituting

.. math::

   a(r) = A(r)\, r^{-q}, \qquad \tilde{a}(k) = \tilde{A}(k)\, k^{q} \tag{157}

turns (158)–(159) into

.. math::

   \tilde{a}(k) = \int_0^\infty a(r)\, (kr)^q J_\mu(kr)\, k\, dr, \tag{155}

.. math::

   a(r) = \int_0^\infty \tilde{a}(k)\, (kr)^{-q} J_\mu(kr)\, r\, dk. \tag{156}

For continuous functions, the two pairs are the same transform written in two
ways. After discretization they are not, and that difference is the reason
the ``bias`` argument exists (see `What the bias does`_ below).

Why logarithmic grids
---------------------

The kernel :math:`J_\mu(kr)` depends only on the product
:math:`kr = e^{\ln k + \ln r}`. As Siegman (1977) noted, in the variables
:math:`\ln r` and :math:`\ln k` the Hankel transform is therefore a
*convolution*. Convolutions become products under a Fourier transform, so a
Hankel transform can be computed as

.. centered:: FFT → multiply by a function → FFT back.

That structure requires samples evenly spaced in :math:`\ln r`, which is why
every FFTLog input lives on a logarithmic grid.

Periodic, band-limited functions
--------------------------------

An ordinary FFT is *exact* for a function that is periodic and contains only
the :math:`N` lowest Fourier modes. FFTLog makes the same assumption in
logarithmic space. Suppose :math:`a(r)` is periodic in :math:`\ln r` with
period :math:`L`,

.. math::

   a(r\, e^{L}) = a(r), \tag{167}

on a fundamental interval :math:`[\ln r_0 - L/2,\ \ln r_0 + L/2]` centred at
:math:`\ln r_0`, and that it contains only the lowest :math:`N` modes:

.. math::

   a(r) = {\sum_m}' c_m\, e^{2\pi i m \ln(r/r_0)/L}. \tag{168}

The primed sum runs over :math:`m = -[N/2], \dots, [N/2]`. For even
:math:`N`, the two outermost (Nyquist) terms get half weight (164). By the
sampling theorem, the coefficients :math:`c_m` are determined exactly by the
samples :math:`a_n = a(r_n)` at :math:`r_n = r_0\, e^{nL/N}`:

.. math::

   c_m = \frac{1}{N} {\sum_n}' a_n\, e^{-2\pi i m n/N}, \tag{169}

which is one FFT. The grid spacing :math:`L/N` is what ``fftloggin`` calls
``dlog``; Hamilton calls one such step a *notch*.

Each mode transforms exactly
----------------------------

Substituting (168) into (155) gives a sum of integrals of power laws against
the Bessel function (171). Each of these has a closed form:

.. math::

   U_\mu(x) \equiv \int_0^\infty t^{x} J_\mu(t)\, dt
   = 2^{x}\, \frac{\Gamma[(\mu + 1 + x)/2]}{\Gamma[(\mu + 1 - x)/2]}. \tag{172}

The transform of the periodic function is therefore itself a finite Fourier
series in :math:`\ln k`,

.. math::

   \tilde{a}(k) = {\sum_m}' c_m\, u_m\, e^{-2\pi i m \ln(k/k_0)/L}, \tag{173}

with coefficients

.. math::

   u_m(\mu, q) = (k_0 r_0)^{-2\pi i m/L}\;
   U_\mu\!\left(q + \frac{2\pi i m}{L}\right). \tag{174}

Evaluated at :math:`k_n = k_0\, e^{nL/N}`, equation (173) is again one FFT.
Nothing has been approximated so far: for a function that really is periodic
and band-limited, FFTLog is exact to machine precision.

Two details matter in practice. First, :math:`u_m^* = u_{-m}`, so a real input
gives a real output. Second, the sampling theorem needs
:math:`u_{-N/2} = u_{N/2}` for even :math:`N`, which (174) does not
guarantee. At the sample points only the real part of the Nyquist term
survives, so FFTLog replaces

.. math::

   u_{\pm N/2} \to \operatorname{Re} u_{N/2}. \tag{175}

This replacement is the source of the ringing discussed under
`Low ringing`_.

The algorithm
-------------

Put together, the biased algorithm of Hamilton §B.4 is what
:func:`~fftloggin.fftlog.forward` executes:

1. **bias** the input: :math:`a_n = A_n\, r_n^{-q}` (157);
2. **FFT** :math:`a_n` to obtain :math:`c_m` (169);
3. **multiply** by :math:`u_m`, from (174) with the replacement (175);
4. **FFT back** to obtain :math:`\tilde{a}_n` (177);
5. **unbias** the output: :math:`\tilde{A}_n = \tilde{a}_n\, k_n^{-q}` (157).

:func:`~fftloggin.fftlog.inverse` runs the same steps but *divides* by
:math:`u_m` in step 3.

.. plot::
   :caption: The five steps of the biased FFTLog algorithm (Hamilton §B.4).
             Equation numbers refer to Hamilton (2000).
   :alt: Flow diagram: A_n, bias, FFT, multiply by u_m, inverse FFT, unbias, tilde A_n.

   import matplotlib.pyplot as plt
   from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

   steps = [
       (r"$A_n$", "input"),
       (r"$\times\, r_n^{-q}$", "bias (157)"),
       ("FFT", r"$c_m$ (169)"),
       (r"$\times\, u_m$", "(174), (175)"),
       (r"FFT$^{-1}$", r"$\tilde a_n$ (177)"),
       (r"$\times\, k_n^{-q}$", "unbias (157)"),
       (r"$\tilde A_n$", "output"),
   ]
   fig, ax = plt.subplots(figsize=(9, 1.6))
   width, gap = 1.0, 0.35
   for i, (label, note) in enumerate(steps):
       x = i * (width + gap)
       edge = "C0" if 0 < i < len(steps) - 1 else "0.4"
       ax.add_patch(FancyBboxPatch(
           (x, 0), width, 0.8, boxstyle="round,pad=0.05",
           facecolor="white", edgecolor=edge, linewidth=1.5,
       ))
       ax.text(x + width / 2, 0.4, label, ha="center", va="center", fontsize=13)
       ax.text(x + width / 2, -0.35, note, ha="center", va="center", fontsize=9,
               color="0.3")
       if i:
           ax.add_patch(FancyArrowPatch(
               (x - gap + 0.05, 0.4), (x - 0.05, 0.4),
               arrowstyle="->", mutation_scale=12, color="0.3",
           ))
   ax.set_xlim(-0.2, len(steps) * (width + gap) - gap + 0.2)
   ax.set_ylim(-0.65, 1.0)
   ax.axis("off")

What the bias does
------------------

In the continuum, the bias :math:`q` changes nothing: (155)–(156) and
(158)–(159) are the same transform. In the discrete case, Hamilton points out
that the bias changes **which sequence is assumed periodic**. Without bias it
is :math:`A_n`; with bias it is :math:`a_n = A_n\, r_n^{-q}`.

FFTLog's errors come from pretending the input repeats. If the two ends of
the input do not match, its periodic copies meet with a jump, and the
transform rings, just as an FFT rings at a discontinuity. A good bias makes
the biased sequence decay at both ends, so the copies join smoothly. The same
reasoning applies to the output side, :math:`\tilde{a}_n = \tilde{A}_n\,
k_n^{q}`.

The figure below uses the exact pair :math:`A(r) = e^{-r}`,
:math:`\tilde{A}(k) = k/\sqrt{1 + k^2}` with :math:`\mu = 0`. Without bias,
:math:`A \to 1` as :math:`r \to 0`, so the periodic copies jump from 1 to 0.
With :math:`q = -1/2`, both :math:`A\, r^{1/2}` and
:math:`\tilde{A}\, k^{-1/2}` vanish at both ends, and the error drops by about
four orders of magnitude.

.. plot::
   :caption: Left: the sequence that FFTLog treats as periodic, repeated over
             three periods, for two biases. Right: relative error against the
             exact transform.
   :alt: Periodic continuation of exp(-r) with and without bias, and the
         corresponding relative errors of the FFTLog transform.

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import numpy as np

   from fftloggin import BesselJKernel, forward, get_paired_grids, lowring_log_kr

   jax.config.update("jax_enable_x64", True)

   n = 256
   r = np.geomspace(1e-4, 1e4, n)
   dlog = np.log(r[1] / r[0])
   period = n * dlog
   kernel = BesselJKernel(0.0)
   signal = np.exp(-r)

   fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
   for q, color in [(0.0, "C3"), (-0.5, "C0")]:
       biased = signal * r**-q
       for shift in (-1, 0, 1):
           left.plot(np.log(r) + shift * period, biased / biased.max(), color=color,
                     label=f"$q={q:g}$" if shift == 0 else None)
       log_kr = lowring_log_kr(kernel, dlog=dlog, bias=q)
       _, k = get_paired_grids(r=r, log_kr=log_kr)
       k = np.asarray(k)
       result = np.asarray(forward(signal, kernel, dlog=dlog, bias=q, log_kr=log_kr))
       exact = k / np.sqrt(1 + k**2)
       right.loglog(k, np.abs(result / exact - 1), color=color, label=f"$q={q:g}$")
   for edge in (-1.5, -0.5, 0.5, 1.5):
       left.axvline(np.log(r).mean() + edge * period, color="0.6", ls=":", lw=1)
   left.set_xlabel(r"$\ln r$")
   left.set_ylabel(r"$A(r)\,r^{-q}$ (normalized)")
   left.legend(loc="center right")
   right.set_xlabel(r"$k$")
   right.set_ylabel("relative error")
   right.legend()
   fig.tight_layout()

The bias is not free. The integral (172) converges only for

.. math::

   -(\mu + 1) < \operatorname{Re} x < \tfrac{1}{2},

and FFTLog samples it at :math:`x = q + 2\pi i m/L`, so :math:`q` must lie in
this strip. At the lower edge, :math:`\mu + 1 + q = 0`, :math:`u_0` is
infinite and the forward transform is singular (and similarly
:math:`\mu + 1 - q = 0` for the inverse). Each kernel reports its strip through
``kernel.domain``, and :func:`~fftloggin.fftlog.validate_parameters` warns if
a concrete bias falls outside it. :doc:`kernels` explains the correspondence.

Low ringing
-----------

The centres :math:`r_0` and :math:`k_0` of the two periodic intervals can be
chosen freely, but they matter through (174): the product :math:`k_0 r_0`
sets the phase of every :math:`u_m`. Replacing the Nyquist coefficient by its
real part (175) discards information unless that coefficient is already real.
Requiring

.. math::

   u_{-N/2} = u_{N/2} \tag{184}

makes the sequence :math:`u_m` "fold smoothly across the period boundary", in
Hamilton's words, and removes the artefact. For real :math:`\mu` and
:math:`q` the condition fixes :math:`k_0 r_0` up to a whole number of notches:

.. math::

   \ln(k_0 r_0) = \frac{L}{N}\left[\frac{1}{\pi}
   \operatorname{Arg} U_\mu\!\left(q + \frac{i\pi N}{L}\right)
   + \text{integer}\right]. \tag{186}

:func:`~fftloggin.fftlog.lowring_log_kr` evaluates (186), choosing the integer that
puts the result nearest to a requested ``log_kr``. A low-ringing value
therefore always exists within half a notch of any value you ask for.

.. plot::
   :caption: Sweeping ln(k0 r0) across three notches for mu = 1/2 and q = 0.
             Top: the imaginary part of the Nyquist coefficient, which
             (175) throws away. Bottom: error of applying the forward
             transform twice to a random sequence, which (187) says should
             return the input. Dashed lines mark the values returned by
             lowring_log_kr.
   :alt: Imaginary part of u at the Nyquist frequency and round-trip error
         as functions of ln(k0 r0), both vanishing at the low-ringing values.

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import numpy as np

   from fftloggin import BesselJKernel, forward, lowring_log_kr

   jax.config.update("jax_enable_x64", True)

   n, dlog = 128, 0.1
   kernel = BesselJKernel(0.5)
   samples = jnp.asarray(np.random.default_rng(0).normal(size=n))
   nyquist = jnp.pi / dlog

   def imag_nyquist(log_kr):
       return jnp.imag(kernel(1 + 1j * nyquist) * jnp.exp(-1j * nyquist * log_kr))

   def round_trip_error(log_kr):
       twice = forward(forward(samples, kernel, dlog=dlog, log_kr=log_kr),
                       kernel, dlog=dlog, log_kr=log_kr)
       return jnp.max(jnp.abs(twice - samples))

   lowring = float(lowring_log_kr(kernel, dlog=dlog))
   notches = lowring / dlog + np.arange(-1, 2)
   sweep = np.sort(np.concatenate([np.linspace(-1.5, 1.5, 400), notches]))

   fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
   top.plot(sweep, jax.vmap(imag_nyquist)(sweep * dlog))
   top.axhline(0, color="0.7", lw=0.8)
   top.set_ylabel(r"Im $u_{N/2}$")
   bottom.semilogy(sweep, jax.vmap(round_trip_error)(sweep * dlog), color="C1")
   bottom.set_ylabel("round-trip error")
   bottom.set_xlabel(r"$\ln(k_0 r_0)$ in notches $L/N$")
   for ax in (top, bottom):
       for notch in notches:
           ax.axvline(notch, color="0.4", ls="--", lw=1)
   fig.tight_layout()

Hamilton notes that moving :math:`\ln(k_0 r_0)` by one whole notch only shifts
the output grid, and the output values with it, by one sample. So snapping to
the nearest low-ringing value costs nothing in coverage. It does make the
result piecewise constant in the requested ``log_kr``, with a zero derivative,
so do the snap once before a fit rather than inside a differentiated
function.

Inverse and unitarity
---------------------

In the continuum, (156) is (155) with :math:`q \to -q`. Discretely, this holds
for even :math:`N` only when the low-ringing condition (184) is met:

.. math::

   v^-_n(\mu, q) = v^+_n(\mu, -q), \tag{185}

where :math:`v^\pm` are the forward and inverse discrete Hankel modes. In
``fftloggin`` terms, ``inverse(x, kernel, bias=q, log_kr=L)`` equals
``forward(x, kernel, bias=-q, log_kr=L)`` to rounding error when ``L`` is a
low-ringing value, and differs otherwise. If the Nyquist coefficient is purely
imaginary, its real part vanishes and the inverse is singular. A low-ringing
centre avoids this too.

With :math:`q = 0` as well, the discrete transform is orthogonal and its own
inverse (187): applying ``forward`` twice returns the input to machine
precision. For numerical work, prefer a low-ringing ``log_kr`` unless you have
a reason not to.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Hamilton (2000)
     - ``fftloggin``
   * - bias :math:`q` (157)
     - ``bias``
   * - log spacing :math:`L/N` (one notch)
     - ``dlog``; see :func:`~fftloggin.grids.infer_dlog`
   * - log period :math:`L`
     - ``N * dlog``, where ``N`` is the number of samples
   * - :math:`\ln(k_0 r_0)` (174), (186)
     - ``log_kr``
   * - grid centres :math:`r_0`, :math:`k_0`
     - :func:`~fftloggin.grids.get_array_center` of each grid
   * - :math:`r_n = r_0 e^{nL/N}`, :math:`k_n = k_0 e^{nL/N}`
     - :func:`~fftloggin.grids.get_paired_grids`
   * - :math:`U_\mu(x)` (172)
     - ``BesselJKernel(mu)(1 + x)``; see :doc:`kernels`
   * - unbiased pair :math:`A`, :math:`\tilde A` (158)–(159)
     - inputs and outputs of :func:`~fftloggin.fftlog.forward` and
       :func:`~fftloggin.fftlog.inverse`
   * - low-ringing :math:`\ln(k_0 r_0)` (186)
     - :func:`~fftloggin.fftlog.lowring_log_kr`

``fftloggin`` indexes samples from ``0`` to ``N - 1`` rather than from
:math:`-[N/2]` to :math:`[N/2]`. The grid centre is the geometric mean of the
endpoints in both conventions.

The transforms are pure JAX functions that work with ``jit``, ``vmap`` and
``grad``. The :doc:`JAX guide <jax>` covers the details.

References
----------

- A. J. S. Hamilton, "Uncorrelated modes of the non-linear power spectrum",
  *MNRAS* **312**, 257 (2000), Appendix B.
  `arXiv:astro-ph/9905191 <https://arxiv.org/abs/astro-ph/9905191>`_.
- J. D. Talman, "Numerical Fourier and Bessel transforms in logarithmic
  variables", *J. Comput. Phys.* **29**, 35 (1978).
- A. E. Siegman, "Quasi fast Hankel transform", *Opt. Lett.* **1**, 13 (1977).
