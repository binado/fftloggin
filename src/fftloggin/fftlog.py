import warnings
from enum import Enum
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.fft import irfft, rfft

from .grids import Grid
from .exceptions import ArgumentOutOfDomainError, DomainCheckWarning
from .kernels import Kernel

LN_2 = np.log(2)


class DomainCheckMode(Enum):
    """
    Mode for domain validation in FFTLog.

    Controls how FFTLog responds when the bias parameter places the
    evaluation point outside the kernel's domain of convergence.

    Attributes
    ----------
    RAISE : str
        Raise a ValueError if domain validation fails.
    WARN : str
        Issue a DomainCheckWarning if domain validation fails.
    SILENT : str
        Skip domain validation entirely.

    Examples
    --------
    >>> from fftloggin import FFTLog, DomainCheckMode
    >>> from fftloggin.kernels import BesselJKernel
    >>> fftlog = FFTLog(
    ...     kernel=BesselJKernel(0),
    ...     n=64,
    ...     dlog=0.1,
    ...     bias=0.0,
    ...     check_domain=DomainCheckMode.RAISE
    ... )
    """

    RAISE = "raise"
    WARN = "warn"
    SILENT = "silent"


def _forward_hankel_transform(
    a: npt.ArrayLike,
    u: npt.ArrayLike,
    logc: npt.ArrayLike,
    dlog: npt.ArrayLike,
    bias: npt.ArrayLike,
    **kwargs,
):
    """
    Low-level forward Hankel transform implementation.

    Parameters
    ----------
    a : array_like
        Input array with shape (n,).
    u : array_like
        FFT coefficients with shape (*batch_shape, ns) where ns = n//2 + 1.
    logc : array_like
        Log-center parameter. Scalar or shape (*batch_shape, 1).
    dlog : array_like
        Logarithmic spacing. Scalar or shape (*batch_shape, 1).
    bias : array_like
        Power-law bias. Scalar or shape (*batch_shape, 1).
    **kwargs
        Additional arguments for scipy.fft.rfft.

    Returns
    -------
    ak : ndarray
        Transformed array with shape (n,) for scalar params or (*batch_shape, n)
        for batched params.
    """
    a = np.asarray(a)
    u = np.asarray(u)
    logc = np.asarray(logc)
    bias = np.asarray(bias)
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
    ak_biased = np.flip(ak_biased, axis=-1)  # type: ignore

    # Step 4: unbias ak by (k_0 r_0)^{-q} (k_n / k_0)^{-q}
    ak = ak_biased * bias_power_law * np.exp(-bias * logc)
    return ak


def _inverse_hankel_transform(
    ak: npt.ArrayLike,
    u: npt.ArrayLike,
    logc: npt.ArrayLike,
    dlog: npt.ArrayLike,
    bias: npt.ArrayLike,
    **kwargs,
):
    """
    Low-level inverse Hankel transform implementation.

    Parameters
    ----------
    ak : array_like
        Input array with shape (n,).
    u : array_like
        FFT coefficients with shape (*batch_shape, ns) where ns = n//2 + 1.
    logc : array_like
        Log-center parameter. Scalar or shape (*batch_shape, 1).
    dlog : array_like
        Logarithmic spacing. Scalar or shape (*batch_shape, 1).
    bias : array_like
        Power-law bias. Scalar or shape (*batch_shape, 1).
    **kwargs
        Additional arguments for scipy.fft.rfft.

    Returns
    -------
    a : ndarray
        Inverse transformed array with shape (n,) for scalar params or
        (*batch_shape, n) for batched params.
    """
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
    a_biased = np.flip(a_biased, axis=-1)  # type: ignore

    # Step 4: unbias ak by (r_n / r_0)^{q}
    a = a_biased * bias_power_law
    return a


def optimal_logcenter(
    kernel: Kernel, dlog: npt.ArrayLike, bias: npt.ArrayLike = 0.0
) -> npt.NDArray:
    """
    Compute optimal log-center parameter to minimize ringing.

    Implements Eq.(30) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf

    Parameters
    ----------
    kernel : Kernel
        Mellin transform kernel.
    dlog : array_like
        Logarithmic spacing. Can be scalar or array with shape (*batch_shape, 1).
    bias : array_like, optional
        Power-law bias exponent (default: 0.0). Can be scalar or array with
        shape (*batch_shape, 1).

    Returns
    -------
    logc : ndarray
        Optimal log-center parameter. Scalar or shape (*batch_shape, 1).
    """
    dlog = np.asarray(dlog)
    bias = np.asarray(bias)
    s = 1j * np.pi / dlog + 1 + bias
    arg = np.angle(kernel.forward(s))
    return dlog * arg / np.pi


def compute_kernel_coefficients(
    kernel: Kernel,
    n: int,
    kr: npt.ArrayLike,
    dlog: npt.ArrayLike,
    bias: npt.ArrayLike = 0.0,
) -> npt.NDArray:
    """
    Compute FFT coefficients for FFTLog transform.

    Implements Eq.(18) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf

    Parameters
    ----------
    kernel : Kernel
        Mellin transform kernel.
    n : int
        Number of sampling points.
    kr : array_like
        The product k*r at the geometric center of the grid. Can be scalar
        or array with shape (*batch_shape, 1).
    dlog : array_like
        Logarithmic spacing. Can be scalar or array with shape (*batch_shape, 1).
    bias : array_like, optional
        Power-law bias exponent (default: 0.0). Can be scalar or array with
        shape (*batch_shape, 1).

    Returns
    -------
    coeffs : ndarray
        FFT coefficients with shape (ns,) for scalar inputs or (*batch_shape, ns)
        for batched inputs, where ns = n//2 + 1.
    """
    dlog = np.asarray(dlog)
    bias = np.asarray(bias)
    # Length of real Fourier transform
    ns = n // 2 + 1
    m = np.arange(0, ns)
    angle = 2 * np.pi * m * 1j / (n * dlog)
    s = angle + 1 + bias
    coeffs = kernel.forward(s)
    kr = np.asarray(kr)
    coeffs = coeffs / kr**angle
    # Handle Nyquist frequency for even n
    if n % 2 == 0:
        coeffs[..., -1] = np.real(coeffs[..., -1])

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
        Mellin transform kernel instance (e.g., BesselJKernel).
    n : int
        Number of sampling points.
    dlog : array_like
        Uniform logarithmic spacing. Can be scalar or array with shape
        (*batch_shape, 1) for batch transforms.
    bias : array_like, optional
        Exponent of power law bias (default: 0.0). Can be scalar or array
        with shape (*batch_shape, 1) for batch transforms.
    lowring : bool, optional
        Whether to snap kr to low-ringing condition (default: True).
    kr : array_like, optional
        The product k*r at the geometric center of the grid (default: 1.0).
        Can be scalar or array with shape (*batch_shape, 1) for batch transforms.
    check_domain : DomainCheckMode or str, optional
        How to handle domain validation at construction time (default: WARN).
        Can be a DomainCheckMode enum value or string ('raise', 'warn', 'silent').
        - RAISE: Raise ValueError if bias is outside kernel's domain of convergence.
        - WARN: Issue DomainCheckWarning if bias is outside kernel's domain of
          convergence.
        - SILENT: Skip domain validation entirely.

    Attributes
    ----------
    kernel : Kernel
        The Mellin transform kernel.
    n : int
        Number of sampling points.
    dlog : array_like
        Uniform logarithmic spacing. Scalar or shape (*batch_shape, 1).
    bias : array_like
        Exponent of power law bias. Scalar or shape (*batch_shape, 1).
    lowring : bool
        Whether kr is snapped to minimize ringing.
    kr : array_like
        The product k*r at the geometric center of the grid (cached property).
        Scalar or shape (*batch_shape, 1).
    logc : array_like
        Natural logarithm of kr (cached property). Scalar or shape (*batch_shape, 1).
    kernel_coefficients : ndarray
        Precomputed FFT coefficients (cached property). Shape (*batch_shape, ns)
        where ns = n//2 + 1.

    Examples
    --------
    Direct FFTLog usage (you manage coordinates separately):

    >>> import numpy as np
    >>> from fftloggin import FFTLog
    >>> from fftloggin.kernels import BesselJKernel
    >>>
    >>> # Create transform
    >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)
    >>>
    >>> # Transform data (you manage coordinates separately)
    >>> a = np.random.randn(128)
    >>> A = fftlog.forward(a)

    For managing coordinates, use FFTLog.create_grid():

    >>> import numpy as np
    >>> from fftloggin import FFTLog
    >>> from fftloggin.kernels import BesselJKernel
    >>>
    >>> # Create FFTLog from r array
    >>> r = np.logspace(-2, 2, 128)
    >>> fftlog = FFTLog.from_array(r, BesselJKernel(0), kr=1.0)
    >>>
    >>> # Create grid to manage coordinates
    >>> grid = fftlog.create_grid(r=r)
    >>>
    >>> # Access coordinates
    >>> print(grid.k.shape)  # Output wavenumbers
    (128,)

    Batching with Array Parameters
    -------------------------------
    Parameters dlog, bias, and kr can be arrays for batch transforms.
    Arrays must have shape (*batch_shape, 1) for proper broadcasting
    with the sample dimension (n,), producing results with shape (*batch_shape, n).

    Using the helper function (recommended):

    >>> from fftloggin import prepare_batch_params
    >>> dlog, bias, kr = prepare_batch_params(0.05, 0.0, [0.5, 1.0, 2.0])
    >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=dlog, bias=bias, kr=kr)
    >>> a = np.random.randn(128)
    >>> result = fftlog.forward(a)
    >>> result.shape
    (3, 128)

    Manual reshaping for batched parameters:

    >>> kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)  # shape (3, 1)
    >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05, kr=kr)
    >>> result = fftlog.forward(a)  # shape (3, 128)

    Multiple batch dimensions via outer product broadcasting:

    >>> dlog, bias, kr = prepare_batch_params(
    ...     [0.04, 0.05],  # 2 values
    ...     0.0,           # scalar
    ...     [0.5, 1.0]     # 2 values
    ... )
    >>> # After prepare_batch_params: dlog=(2,1), bias=(), kr=(2,1)
    >>> # They broadcast together to shape (2,1)
    >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=dlog, bias=bias, kr=kr)
    >>> result = fftlog.forward(a)  # shape (2, 128)

    Important: Batched kernels can be combined with batched parameters.
    For batched kernels with different `mu` values:

    >>> mu_batch = np.array([0, 1, 2]).reshape(3, 1)  # shape (3, 1)
    >>> kernel_batch = BesselJKernel(mu_batch)  # Batched kernel
    >>> fftlog = FFTLog(kernel=kernel_batch, n=128, dlog=0.05, kr=0.5)
    >>> result = fftlog.forward(a)  # Result shape (3, 128)

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
        dlog: npt.ArrayLike,
        bias: npt.ArrayLike = 0.0,
        lowring: bool = True,
        kr: npt.ArrayLike = 1,
        check_domain: DomainCheckMode | str = DomainCheckMode.WARN,
    ) -> None:
        self._kernel = kernel
        self._n = n
        self._dlog = dlog
        self._bias = bias
        self._lowring = lowring
        self._kr = kr
        self._check_domain = (
            DomainCheckMode(check_domain)
            if isinstance(check_domain, str)
            else check_domain
        )

        # Validate domain at construction time
        self._validate_domain()

    def _cleanup(self) -> None:
        try:
            del self.logc
        except AttributeError:
            pass
        try:
            del self.kernel_coefficients
        except AttributeError:
            pass
        try:
            del self.optimal_logcenter
        except AttributeError:
            pass

    def _validate_domain(self) -> None:
        """Validate that bias is within the kernel's domain of convergence."""
        if self._check_domain == DomainCheckMode.SILENT:
            return

        if self._check_domain == DomainCheckMode.RAISE:
            self.check_domain(raise_exception=True)
            return

        if self._check_domain == DomainCheckMode.WARN:
            try:
                self.check_domain(raise_exception=True)
            except ArgumentOutOfDomainError as e:
                warnings.warn(str(e), DomainCheckWarning, stacklevel=3)

    def check_domain(self, raise_exception: bool = False) -> bool:
        """
        Check if the current configuration is within the kernel's domain.

        This is a lightweight check that uses kernel.is_in_domain() directly
        rather than forcing recomputation of cached properties.

        Parameters
        ----------
        raise_exception : bool, optional
            If True, raise ArgumentOutOfDomainError when invalid. Default is False.

        Returns
        -------
        bool
            True if configuration is valid, False otherwise.

        Raises
        ------
        ArgumentOutOfDomainError
            If raise_exception is True and configuration is invalid.

        Examples
        --------
        >>> from fftloggin import FFTLog, DomainCheckMode
        >>> from fftloggin.kernels import BesselJKernel
        >>> fftlog = FFTLog(
        ...     kernel=BesselJKernel(0),
        ...     n=64,
        ...     dlog=0.1,
        ...     bias=0.0,
        ...     check_domain=DomainCheckMode.SILENT
        ... )
        >>> fftlog.check_domain()
        True
        >>> fftlog.bias = 1.0  # Outside domain for mu=0
        >>> fftlog.check_domain()
        False
        """
        s = 1 + np.asarray(self._bias)
        is_valid = self._kernel.is_in_domain(s)

        if not is_valid and raise_exception:
            inf, sup = self._kernel.domain
            # Format inf/sup for broadcasting with s
            from .utils import safe_broadcast

            inf, _ = safe_broadcast(inf, s)
            sup, _ = safe_broadcast(sup, s)

            # Create context message with helpful bias range
            context_msg = (
                f"FFTLog bias parameter: 1 + bias = {np.real(s)}. "
                f"Valid bias range: ({inf - 1}, {sup - 1})"
            )

            raise ArgumentOutOfDomainError(
                s_values=s, kernel=self._kernel, context=context_msg
            )

        return is_valid

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @kernel.setter
    def kernel(self, other: Kernel):
        self._kernel = other
        self._cleanup()
        self._validate_domain()

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, other: int):
        self._n = other
        self._cleanup()

    @property
    def dlog(self) -> npt.ArrayLike:
        return self._dlog

    @dlog.setter
    def dlog(self, other: npt.ArrayLike):
        self._dlog = other
        self._cleanup()

    @property
    def bias(self) -> npt.ArrayLike:
        return self._bias

    @bias.setter
    def bias(self, other: npt.ArrayLike):
        self._bias = other
        self._cleanup()
        self._validate_domain()

    @property
    def lowring(self) -> bool:
        return self._lowring

    @lowring.setter
    def lowring(self, other: bool):
        self._lowring = other
        self._cleanup()

    @cached_property
    def kr(self) -> npt.ArrayLike:
        if self.lowring:
            logc_snapped = self.shift_logcenter(np.log(self._kr))
            return np.exp(logc_snapped)
        else:
            return self._kr

    @cached_property
    def logc(self) -> npt.ArrayLike:
        return np.log(self.kr)

    @cached_property
    def kernel_coefficients(self) -> npt.NDArray:
        return self._compute_kernel_coefficients()

    def _compute_kernel_coefficients(self) -> npt.NDArray:
        """Compute FFT coefficients using instance parameters."""
        return compute_kernel_coefficients(
            self.kernel, self.n, self.kr, self.dlog, self.bias
        )

    @classmethod
    def from_array(
        cls,
        x: npt.ArrayLike,
        kernel: Kernel,
        bias: npt.ArrayLike = 0.0,
        kr: npt.ArrayLike = 1.0,
        lowring: bool = True,
        check_domain: DomainCheckMode | str = DomainCheckMode.WARN,
    ) -> "FFTLog":
        """
        Create FFTLog instance from a log-spaced coordinate array.

        Parameters
        ----------
        x : array_like
            Log-spaced coordinate array (1D).
        kernel : Kernel
            Mellin transform kernel.
        bias : array_like, optional
            Power-law bias exponent. Default is 0.0.
        kr : array_like, optional
            The product k*r at the geometric center of the grid. Default is 1.0.
        lowring : bool, optional
            Whether to snap kr to minimize ringing. Default is True.
        check_domain : DomainCheckMode or str, optional
            How to handle domain validation (default: WARN).

        Returns
        -------
        FFTLog
            Configured FFTLog instance.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.fftlog import FFTLog
        >>> from fftloggin.kernels import BesselJKernel
        >>> r = np.logspace(-2, 2, 128)
        >>> fftlog = FFTLog.from_array(r, BesselJKernel(0), bias=0.0, kr=1.0)
        """
        from .grids import infer_dlog

        x = np.asarray(x)
        n = x.shape[-1]
        dlog = infer_dlog(x)

        return cls(
            kernel,
            n,
            dlog,
            bias=bias,
            lowring=lowring,
            kr=kr,
            check_domain=check_domain,
        )

    def create_grid(
        self,
        r: npt.ArrayLike | None = None,
        k: npt.ArrayLike | None = None,
    ) -> Grid:
        """
        Create a Grid from one coordinate array using the FFTLog kr parameter.

        Exactly one of r or k must be provided. The other coordinate array
        is computed using get_other_array() with the FFTLog instance's kr.

        Parameters
        ----------
        r : array_like, optional
            Input radial coordinates. If provided, k is computed.
        k : array_like, optional
            Output wavenumber coordinates. If provided, r is computed.

        Returns
        -------
        Grid
            Configured Grid with both r and k arrays.

        Raises
        ------
        ValueError
            If neither r nor k is provided, or if both are provided.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.fftlog import FFTLog
        >>> from fftloggin.kernels import BesselJKernel
        >>> r = np.logspace(-2, 2, 128)
        >>> fftlog = FFTLog.from_array(r, BesselJKernel(0), kr=1.0)
        >>> grid = fftlog.create_grid(r=r)
        >>> print(f"{grid.k[0]:.6f}")  # First k value
        0.010129
        """
        from .grids import Grid, get_other_array

        if (r is None and k is None) or (r is not None and k is not None):
            raise ValueError(
                "Exactly one of 'r' or 'k' must be provided. "
                f"Got r={r is not None}, k={k is not None}"
            )

        if r is not None:
            r_arr = np.asarray(r)
            k_arr = get_other_array(r_arr, self.logc)
            return Grid(r_arr, k_arr)
        else:
            k_arr = np.asarray(k)
            r_arr = get_other_array(k_arr, self.logc)
            return Grid(r_arr, k_arr)

    @cached_property
    def optimal_logcenter(self) -> npt.NDArray:
        """
        Compute optimal log-center parameter to minimize ringing.

        Implements Eq.(30) of https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf

        Returns
        -------
        ndarray
            Optimal log-center parameter. Scalar or shape (*batch_shape, 1).
        """
        return optimal_logcenter(self.kernel, self.dlog, self.bias)

    def shift_logcenter(self, logc: npt.ArrayLike) -> npt.NDArray:
        logc = np.asarray(logc)
        dlog = np.asarray(self.dlog)
        optimal = self.optimal_logcenter
        # Snap to nearest integer multiple of dlog from optimal
        # This matches Fortran's krgood: krgood = kr * exp((arg - round(arg)) * dlnr)
        shift = (logc - optimal) / dlog
        return optimal + np.round(shift) * dlog

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
