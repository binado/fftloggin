from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.fft import irfft, rfft

from .kernels import Kernel

LN_2 = np.log(2)


def _forward_hankel_transform(
    a: npt.ArrayLike,
    u: npt.ArrayLike,
    logc: npt.ArrayLike,
    dlog: float,
    bias: float,
    **kwargs,
):
    a = np.asarray(a)
    u = np.asarray(u)
    logc = np.asarray(logc)
    na = a.shape[-1]
    # Step 1: bias a by (r_n / r_0)^{-q}
    i = np.arange(na).astype(a.dtype)
    ic = (na - 1) / 2
    bias_power_law = np.exp(-bias * (i - ic) * dlog)
    a_biased = a * bias_power_law

    # Step 2: FFT
    a_biased_fftd = rfft(a_biased, **kwargs)

    # Step 3: multiply by coefficients
    # coeffs may be batched, while a is not
    ak_biased = irfft(a_biased_fftd * u, na, **kwargs)
    ak_biased = np.flip(ak_biased)

    # Step 4: unbias ak by (k_0 r_0)^{-q} (k_n / k_0)^{-q}
    ak = ak_biased * bias_power_law * np.exp(-bias * logc)
    return ak


def _inverse_hankel_transform(
    ak: npt.ArrayLike,
    u: npt.ArrayLike,
    logc: npt.ArrayLike,
    dlog: float,
    bias: float,
    **kwargs,
):
    ak = np.asarray(ak)
    u = np.asarray(u)
    logc = np.asarray(logc)
    na = ak.shape[-1]
    # Step 1: bias a by (k_0 r_0)^{q} (k_n / k_0)^{q}
    i = np.arange(na).astype(ak.dtype)
    ic = (na - 1) / 2
    bias_power_law = np.exp(bias * (i - ic) * dlog)
    ak_biased = ak * bias_power_law * np.exp(bias * logc)

    # Step 2: FFT
    ak_biased_fftd = rfft(ak_biased, **kwargs)

    # Step 3: divide by coefficients
    # coeffs may be batched, while a is not
    a_biased = irfft(ak_biased_fftd / np.conjugate(u), na, **kwargs)
    a_biased = np.flip(a_biased)

    # Step 4: unbias ak by (r_n / r_0)^{q}
    a = a_biased * bias_power_law
    return a


def optimal_logcenter(kernel: Kernel, dlog: float, bias: float) -> npt.NDArray:
    """
    Implements Eq.(30) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf
    """
    s = bias + 1j * np.pi / dlog + 1
    arg = np.imag(np.log(kernel.forward(s)))
    return dlog * arg / np.pi


def compute_kernel_coefficients(
    kernel: Kernel, n: int, logc: npt.ArrayLike, dlog: float, bias: float
):
    """
    Implements Eq.(18) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf

    Parameters
    ----------
    logc : array_like
        The value of log(k0r0). If an iterable, its shape must match the
    batch dimension of the kernel.

    """
    # Length of real Fourier transform
    ns = n // 2 + 1
    m = np.arange(0, ns)
    angle = 2 * np.pi * m * 1j / (n * dlog)
    s = bias + angle + 1
    coeffs = kernel.forward(s)
    # If we are batching, logc should have shape (*batch_shape,)
    # Need to broadcast correctly to (*batch_shape, 1)
    logc = np.asarray(logc)
    if logc.ndim > 0 and logc.shape[-1] > 1:
        logc = np.expand_dims(logc, axis=-1)
    coeffs = coeffs * np.exp(-angle * logc)
    # Handle Nyquist frequency for even n
    if n % 2 == 0:
        coeffs[-1] = np.real(coeffs[-1])

    return coeffs


class FFTLog:
    """
    Pure FFTLog transform algorithm for fast Hankel transforms.

    FFTLog implements the fast Hankel transform algorithm described in Hamilton (2000).
    This class focuses purely on the transform computation - for coordinate management
    and data storage, use the Grid class from fftloggin.grids.

    Parameters
    ----------
    kernel : Kernel
        Mellin transform kernel instance (e.g., BesselJKernel). The kernel
        should be configured with the desired bias parameter.
    n : int
        Number of sampling points.
    dlog : float
        Uniform logarithmic spacing.
    minimize_ringing : bool, optional
        Whether to snap logc to low-ringing condition (default: True).
    logc : float, optional
        Log-center parameter log(k_c * r_c) (default: 1.0).

    Attributes
    ----------
    kernel : Kernel
        The Mellin transform kernel (includes bias parameter).
    n : int
        Number of sampling points.
    dlog : float
        Uniform logarithmic spacing.
    minimize_ringing : bool
        Whether logc is snapped to minimize ringing.
    logc : float
        Log-center parameter (cached property).
    kernel_coefficients : ndarray
        Precomputed FFT coefficients (cached property).

    Examples
    --------
    Basic usage with Grid (recommended):

    >>> import numpy as np
    >>> from fftloggin import Grid
    >>> from fftloggin.kernels import BesselJKernel
    >>>
    >>> # Create grid from r array
    >>> r = np.logspace(-2, 2, 128)
    >>> grid = Grid.from_r(r, kernel=BesselJKernel(0))
    >>>
    >>> # Perform transform
    >>> a = np.exp(-(grid.r/1.0)**2)
    >>> A = grid.forward(a)
    >>>
    >>> # Access coordinates
    >>> print(grid.k)  # Output wavenumbers

    Advanced usage (direct FFTLog):

    >>> from fftloggin import FFTLog
    >>> from fftloggin.kernels import BesselJKernel
    >>>
    >>> # Create transform with bias in kernel
    >>> kernel = BesselJKernel(mu=0, bias=0.0)
    >>> fftlog = FFTLog(kernel=kernel, n=128, dlog=0.05)
    >>>
    >>> # Transform data (you manage coordinates separately)
    >>> a = np.random.randn(128)
    >>> A = fftlog.forward(a)

    See Also
    --------
    Grid : Workspace class that manages coordinates and data
    BesselJKernel : Standard Hankel transform kernel
    Kernel : Base class for custom kernels

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
    """

    def __init__(
        self,
        kernel: Kernel,
        n: int,
        dlog: float,
        minimize_ringing: bool = True,
        logc: float = 1,
    ) -> None:
        self._kernel = kernel
        self._n = n
        self._dlog = dlog
        self._minimize_ringing = minimize_ringing
        self._logc = logc

    def _cleanup(self) -> None:
        del self.logc
        del self.kernel_coefficients

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @kernel.setter
    def kernel(self, other: Kernel):
        self._kernel = other
        self._cleanup()

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, other: int):
        self._n = other
        self._cleanup()

    @property
    def dlog(self) -> float:
        return self._dlog

    @dlog.setter
    def dlog(self, other: float):
        self._dlog = other
        self._cleanup()

    @property
    def bias(self) -> float:
        return self.kernel.bias

    @bias.setter
    def bias(self, other: float):
        self.kernel.bias = other
        self._cleanup()

    @property
    def minimize_ringing(self) -> bool:
        return self._minimize_ringing

    @minimize_ringing.setter
    def minimize_ringing(self, other: bool):
        self._minimize_ringing = other
        self._cleanup()

    @cached_property
    def logc(self) -> npt.ArrayLike:
        if self.minimize_ringing:
            return self.shift_logcenter(self._logc)
        else:
            return self._logc

    @cached_property
    def kernel_coefficients(self) -> npt.NDArray:
        return self.compute_kernel_coefficients()

    def compute_kernel_coefficients(self) -> npt.NDArray:
        """
        Implements Eq.(18) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf

        Parameters
        ----------
        logc : array_like
            The value of log(k0r0). If an iterable, its shape must match the
        batch dimension of the kernel.

        """
        return compute_kernel_coefficients(
            self.kernel, self.n, self.logc, self.dlog, self.bias
        )

    @classmethod
    def from_array(cls, a: npt.ArrayLike, kernel: Kernel, axis: int = -1) -> "FFTLog":
        log_a = np.log(a)
        n = log_a.shape[axis]
        if n < 2:
            raise ValueError(
                f"Expected array to have at least two points in axis {axis}"
            )
        dlog = log_a[1] - log_a[0]
        dlog_arr = np.diff(log_a)
        if not np.allclose(dlog_arr, dlog):
            raise ValueError("Expected array with even log-spacing")

        return cls(kernel, n, dlog, bias=kernel.bias)

    def optimal_logcenter(self) -> npt.NDArray:
        """
        Implements Eq.(30) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf
        """
        return optimal_logcenter(self.kernel, self.dlog, self.bias)

    def shift_logcenter(self, logc: npt.ArrayLike) -> npt.NDArray:
        logc = np.asarray(logc)
        optimal = self.optimal_logcenter()
        # Snap to nearest integer multiple of dlog from optimal
        # This matches Fortran's krgood: krgood = kr * exp((arg - round(arg)) * dlnr)
        shift = (logc - optimal) / self.dlog
        return optimal + np.round(shift) * self.dlog

    def forward(
        self,
        a: npt.ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform forward Hankel transform: a(r) -> A(k).

        Computes the discrete Hankel transform using the FFTLog algorithm.
        This is a pure computation method - coordinate management should be
        handled separately (typically via the Grid class).

        Parameters
        ----------
        a : array_like
            Real input array to be transformed. Must be sampled on a
            logarithmically-spaced grid with spacing dlog.
        **kwargs
            Additional keyword arguments passed to scipy.fft.rfft.

        Returns
        -------
        A : ndarray
            The transformed output array, representing the function on
            a logarithmically-spaced wavenumber grid.

        Notes
        -----
        The array size is automatically adjusted if input size doesn't match
        self.n. The transform assumes input is sampled on a log-spaced grid.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin import FFTLog
        >>> from fftloggin.kernels import BesselJKernel
        >>>
        >>> # Direct usage (you manage coordinates)
        >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)
        >>> a = np.random.randn(128)
        >>> A = fftlog.forward(a)
        >>> print(A.shape)
        (128,)

        See Also
        --------
        inverse : Inverse Hankel transform
        Grid.forward : Recommended high-level interface with coordinate management
        """
        a = np.asarray(a)
        na = a.shape[-1]
        if na != self.n:
            self.n = na

        return _forward_hankel_transform(
            a, self.kernel_coefficients, self.logc, self.dlog, self.bias, **kwargs
        )

    def inverse(
        self,
        ak: npt.ArrayLike,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform inverse Hankel transform: A(k) -> a(r).

        Computes the inverse discrete Hankel transform using the FFTLog algorithm.
        This is a pure computation method - coordinate management should be
        handled separately (typically via the Grid class).

        Parameters
        ----------
        ak : array_like
            Real input array to be inverse transformed. Must be sampled on a
            logarithmically-spaced grid with spacing dlog.
        **kwargs
            Additional keyword arguments passed to scipy.fft.rfft.

        Returns
        -------
        a : ndarray
            The inverse transformed output array, representing the function on
            a logarithmically-spaced radial grid.

        Notes
        -----
        The array size is automatically adjusted if input size doesn't match
        self.n. The transform assumes input is sampled on a log-spaced grid.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin import FFTLog
        >>> from fftloggin.kernels import BesselJKernel
        >>>
        >>> # Direct usage (you manage coordinates)
        >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)
        >>> A = np.random.randn(128)
        >>> a = fftlog.inverse(A)
        >>> print(a.shape)
        (128,)

        See Also
        --------
        forward : Forward Hankel transform
        Grid.inverse : Recommended high-level interface with coordinate management
        """
        ak = np.asarray(ak)
        na = ak.shape[-1]
        if na != self.n:
            self.n = na

        return _inverse_hankel_transform(
            ak, self.kernel_coefficients, self.logc, self.dlog, self.bias, **kwargs
        )
