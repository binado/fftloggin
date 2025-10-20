import warnings

import numpy as np
import numpy.typing as npt
from scipy import special

from ._backend import _fht, _ifht
from ._backend_numexpr import NUMEXPR_AVAILABLE, _fht_numexpr, _ifht_numexpr

LN_2 = np.log(2)


class Grid:
    """
    Grid for FFTLog transforms with logarithmically-spaced sampling points.

    This class represents both input and output grids for forward and inverse
    Hankel transforms. The grids are always logarithmically spaced and related
    by the FFTLog algorithm.

    Attributes
    ----------
    r : ndarray
        Radial coordinate grid, logarithmically spaced.
    k : ndarray
        Wavenumber coordinate grid, logarithmically spaced.
    n : int
        Number of sampling points.
    dln : float
        Uniform logarithmic spacing.
    offset : float
        Offset of the output array's logarithmic spacing.
    central : float
        Central value controlling the absolute scale of the grids.
    mode : str
        Transform mode: 'fht' for forward or 'ifht' for inverse.

    Examples
    --------
    >>> from fftloggin import Grid
    >>> # Create grid for forward transform
    >>> grid = Grid.fht(128, dln=0.05, offset=0.0, central=1.0)
    >>> r, k = grid.r, grid.k
    >>> # Create grid from existing array
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 64)
    >>> grid = Grid.from_r(r, offset=0.0)
    """

    def __init__(
        self,
        n: int,
        dln: float,
        offset: float,
        central: float,
        mode: str,
    ):
        """
        Initialize Grid (use class methods fht/ifht/from_r/from_k instead).

        Parameters
        ----------
        n : int
            Number of sampling points.
        dln : float
            Uniform logarithmic spacing.
        offset : float
            Offset of the output array's logarithmic spacing.
        central : float
            Central value controlling the absolute scale.
        mode : str
            Transform mode: 'fht' or 'ifht'.
        """
        if mode not in ("fht", "ifht"):
            raise ValueError(f"mode must be 'fht' or 'ifht', got '{mode}'")

        self._n = n
        self._dln = dln
        self._offset = offset
        self._central = central
        self._mode = mode

        # Generate grids based on mode
        ic = (n - 1) // 2
        i = np.arange(n)

        if mode == "fht":
            # Forward: r is input, k is output
            self._r = central * np.exp((i - ic) * dln)
            self._k = (1.0 / central) * np.exp((i - ic) * dln + offset)
        else:  # mode == 'ifht'
            # Inverse: k is input, r is output
            self._k = central * np.exp((i - ic) * dln)
            self._r = (1.0 / central) * np.exp((i - ic) * dln + offset)

    @classmethod
    def fht(
        cls,
        n: int,
        dln: float,
        offset: float = 0.0,
        central: float = 1.0,
    ) -> "Grid":
        """
        Create grid for forward Hankel transform.

        Parameters
        ----------
        n : int
            Number of sampling points.
        dln : float
            Uniform logarithmic spacing: dln = ln(r[i+1]/r[i]).
        offset : float, optional
            Offset of the output k array's logarithmic spacing. Default is 0.0.
        central : float, optional
            Central value: r is multiplied by central, k is divided by central.
            Default is 1.0.

        Returns
        -------
        Grid
            Grid instance with .r (input) and .k (output) attributes.

        Examples
        --------
        >>> grid = Grid.fht(128, dln=0.05)
        >>> r, k = grid.r, grid.k
        """
        return cls(n, dln, offset, central, mode="fht")

    @classmethod
    def ifht(
        cls,
        n: int,
        dln: float,
        offset: float = 0.0,
        central: float = 1.0,
    ) -> "Grid":
        """
        Create grid for inverse Hankel transform.

        Parameters
        ----------
        n : int
            Number of sampling points.
        dln : float
            Uniform logarithmic spacing: dln = ln(k[i+1]/k[i]).
        offset : float, optional
            Offset of the output r array's logarithmic spacing. Default is 0.0.
        central : float, optional
            Central value: k is multiplied by central, r is divided by central.
            Default is 1.0.

        Returns
        -------
        Grid
            Grid instance with .k (input) and .r (output) attributes.

        Examples
        --------
        >>> grid = Grid.ifht(128, dln=0.05)
        >>> k, r = grid.k, grid.r
        """
        return cls(n, dln, offset, central, mode="ifht")

    @classmethod
    def from_r(
        cls,
        r: npt.ArrayLike,
        offset: float = 0.0,
    ) -> "Grid":
        """
        Create forward transform grid from existing r array.

        Infers grid parameters (n, dln, central) from the provided r array
        and generates the corresponding k output grid.

        Parameters
        ----------
        r : array_like
            Input radial grid (1D array) with logarithmically-spaced values.
            Must have at least 2 points.
        offset : float, optional
            Offset of the output k array's logarithmic spacing. Default is 0.0.

        Returns
        -------
        Grid
            Grid instance with inferred parameters.

        Examples
        --------
        >>> import numpy as np
        >>> r = np.logspace(-2, 2, 64)
        >>> grid = Grid.from_r(r)
        >>> k = grid.k
        """
        r = np.asarray(r)
        if r.ndim != 1:
            raise ValueError("r must be a 1D array")
        if len(r) < 2:
            raise ValueError("r must have at least 2 points")

        n = len(r)
        ic = (n - 1) // 2

        # Infer dln from logarithmic spacing
        log_r = np.log(r)
        dln_array = np.diff(log_r)
        dln = np.mean(dln_array)

        # Check that spacing is approximately uniform
        if not np.allclose(dln_array, dln, rtol=1e-6):
            raise ValueError(
                f"Input array r is not uniformly spaced in log. "
                f"Spacing varies by up to {np.max(np.abs(dln_array - dln)) / dln * 100:.2f}%"
            )

        # Infer central from r[ic]
        central = r[ic]

        return cls(n, dln, offset, central, mode="fht")

    @classmethod
    def from_k(
        cls,
        k: npt.ArrayLike,
        offset: float = 0.0,
    ) -> "Grid":
        """
        Create inverse transform grid from existing k array.

        Infers grid parameters (n, dln, central) from the provided k array
        and generates the corresponding r output grid.

        Parameters
        ----------
        k : array_like
            Input wavenumber grid (1D array) with logarithmically-spaced values.
            Must have at least 2 points.
        offset : float, optional
            Offset of the output r array's logarithmic spacing. Default is 0.0.

        Returns
        -------
        Grid
            Grid instance with inferred parameters.

        Examples
        --------
        >>> import numpy as np
        >>> k = np.logspace(-2, 2, 64)
        >>> grid = Grid.from_k(k)
        >>> r = grid.r
        """
        k = np.asarray(k)
        if k.ndim != 1:
            raise ValueError("k must be a 1D array")
        if len(k) < 2:
            raise ValueError("k must have at least 2 points")

        n = len(k)
        ic = (n - 1) // 2

        # Infer dln from logarithmic spacing
        log_k = np.log(k)
        dln_array = np.diff(log_k)
        dln = np.mean(dln_array)

        # Check that spacing is approximately uniform
        if not np.allclose(dln_array, dln, rtol=1e-6):
            raise ValueError(
                f"Input array k is not uniformly spaced in log. "
                f"Spacing varies by up to {np.max(np.abs(dln_array - dln)) / dln * 100:.2f}%"
            )

        # Infer central from k[ic]
        central = k[ic]

        return cls(n, dln, offset, central, mode="ifht")

    @property
    def r(self) -> np.ndarray:
        """Radial coordinate grid."""
        return self._r

    @property
    def k(self) -> np.ndarray:
        """Wavenumber coordinate grid."""
        return self._k

    @property
    def n(self) -> int:
        """Number of sampling points."""
        return self._n

    @property
    def dln(self) -> float:
        """Uniform logarithmic spacing."""
        return self._dln

    @property
    def offset(self) -> float:
        """Offset of the output array's logarithmic spacing."""
        return self._offset

    @property
    def central(self) -> float:
        """Central value controlling the absolute scale."""
        return self._central

    @property
    def mode(self) -> str:
        """Transform mode ('fht' or 'ifht')."""
        return self._mode

    def __repr__(self) -> str:
        """String representation of the Grid."""
        return (
            f"Grid(n={self.n}, dln={self.dln:.6f}, offset={self.offset:.6f}, "
            f"central={self.central:.6f}, mode='{self.mode}')"
        )


def fhtoffset(
    dln: float,
    mu: npt.ArrayLike,
    initial: npt.ArrayLike = 0.0,
    bias: float = 0.0,
) -> np.ndarray:
    """
    Return optimal offset for the fast Hankel transform.

    Vectorized version of scipy.fft.fhtoffset that supports broadcasting
    of mu, offset, and bias parameters.

    Parameters
    ----------
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order of the Hankel transform, any positive or negative real number.
    initial : array_like, optional
        Original offset of the uniform logarithmic spacing.
        Default is 0.0.
    bias : array_like, optional
        Exponent of power law bias. Default is 0.0.

    Returns
    -------
    offset_opt : ndarray
        Optimal offset of the uniform logarithmic spacing of the output array.

    Notes
    -----
    This function computes an optimal value for the offset parameter that
    minimizes ringing in the transform by making the transform kernel periodic.
    This implements the "low-ringing" condition from equations (28)-(30) of [1]_.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    Examples
    --------
    >>> import numpy as np
    >>> from fftlog_vectorized import fhtoffset, fht
    >>> dln = 0.1
    >>> mu = 0.5
    >>> offset_opt = fhtoffset(dln, mu)
    >>> # Use optimal offset for transform
    >>> a = np.random.randn(128)
    >>> A = fht(a, dln, mu, offset=offset_opt)
    """
    mu = np.asarray(mu)
    lnkr = np.asarray(initial)
    q = bias
    xp = (mu + 1 + q) / 2
    xm = (mu + 1 - q) / 2
    y = np.pi / (2 * dln)
    zp = special.loggamma(xp + 1j * y)
    zm = special.loggamma(xm + 1j * y)
    arg = (LN_2 - lnkr) / dln + (zp.imag + zm.imag) / np.pi
    return lnkr + (arg - np.round(arg)) * dln


def fht(
    a: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    use_numexpr: bool = False,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    """
    Compute the fast Hankel transform or general Mellin transform.

    Vectorized transform that supports broadcasting of mu, offset, and bias
    parameters with optional numexpr optimization and custom kernel functions.

    Parameters
    ----------
    a : array_like
        Real input array to be transformed.
        Shape: (..., n) where n is the length along the transform axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order parameter for the kernel (interpretation depends on kernel).
        For Bessel kernel: order of the Hankel transform.
        Can be scalar or array that broadcasts with offset and bias.
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Can be scalar or array that broadcasts with mu and bias.
        Default is 0.0, which is equivalent to offset = log(kr) for kr = 1.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.
        Can be scalar or array that broadcasts with mu and offset.
        Default is 0.0, which gives an unbiased transform.
    axis : int, optional
        Axis along which to perform the transform. Default is -1.
    use_numexpr : bool, optional
        Whether to use numexpr optimization if available.
        - True: Use numexpr if available
        - False: Use standard numpy implementation
        Default is False.
    kernel : callable, optional
        Mellin transform kernel function with signature kernel(mu, y, q) -> complex.
        Returns kernel coefficients at frequency y.
        Either kernel or log_kernel must be provided (not both).
    log_kernel : callable, optional
        Log-Mellin transform kernel with signature log_kernel(mu, y, q) -> complex.
        Returns LOG of kernel coefficients (preferred for numerical stability).
        Either kernel or log_kernel must be provided (not both).

    Returns
    -------
    A : ndarray
        The transformed output array. Shape depends on broadcasting of
        mu, offset, and bias with the input array.

    Raises
    ------
    ValueError
        If neither kernel nor log_kernel is provided, or if both are provided.

    Notes
    -----
    This function computes a discrete version of the Mellin transform.
    For the standard Hankel transform with Bessel kernel J_μ:

        A(k) = \\int_{0}^{\\infty} a(r) J_{\\mu}(kr) (kr)^{q} k dr

    where J_{\\mu} is the Bessel function of order \\mu and q = bias.

    The algorithm is based on the FFTLog method presented in [1]_.

    Numexpr optimization provides significant speedup for large arrays
    (typically 2-5x for n > 10000) but may have slight overhead for
    small arrays (n < 1000).

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    Examples
    --------
    >>> import numpy as np
    >>> from fftloggin import fht
    >>> from fftloggin.kernels import bessel_mellin_log_kernel
    >>> # Logarithmically spaced input
    >>> r = np.logspace(-3, 3, 128)
    >>> dln = np.log(r[1]/r[0])
    >>> a = np.exp(-(r/1.0)**2)  # Gaussian
    >>> # Standard Hankel transform
    >>> A = fht(a, dln, mu=0, log_kernel=bessel_mellin_log_kernel)
    >>> # Multiple transforms with different orders
    >>> mu_vals = np.array([0, 0.5, 1.0])[:, np.newaxis]
    >>> A_multi = fht(a, dln, mu=mu_vals, log_kernel=bessel_mellin_log_kernel)
    >>> # With numexpr optimization
    >>> A_fast = fht(a, dln, mu=0, log_kernel=bessel_mellin_log_kernel, use_numexpr=True)
    """
    # Use appropriate implementation
    if use_numexpr and NUMEXPR_AVAILABLE:
        return _fht_numexpr(a, dln, mu, offset, bias, axis, kernel, log_kernel)
    elif use_numexpr and not NUMEXPR_AVAILABLE:
        warnings.warn(
            "numexpr requested but not available, using standard implementation",
            RuntimeWarning,
            stacklevel=2,
        )
        return _fht(a, dln, mu, offset, bias, axis, kernel, log_kernel)
    else:
        return _fht(a, dln, mu, offset, bias, axis, kernel, log_kernel)


def ifht(
    A: npt.ArrayLike,
    dln: float,
    mu: npt.ArrayLike,
    offset: npt.ArrayLike = 0.0,
    bias: float = 0.0,
    axis: int = -1,
    use_numexpr: bool = False,
    kernel: callable = None,
    log_kernel: callable = None,
) -> np.ndarray:
    """
    Compute the inverse fast Hankel transform or general Mellin transform.

    Vectorized transform that supports broadcasting of mu, offset, and bias
    parameters with optional numexpr optimization and custom kernel functions.

    Parameters
    ----------
    A : array_like
        Real input array to be transformed back.
        Shape: (..., n) where n is the length along the transform axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : array_like
        Order parameter for the kernel (interpretation depends on kernel).
        For Bessel kernel: order of the Hankel transform.
        Can be scalar or array that broadcasts with offset and bias.
    offset : array_like, optional
        Offset of the uniform logarithmic spacing of the output array.
        Can be scalar or array that broadcasts with mu and bias.
        Default is 0.0.
    bias : array_like, optional
        Exponent of power law bias, any positive or negative real number.
        Can be scalar or array that broadcasts with mu and offset.
        Default is 0.0.
    axis : int, optional
        Axis along which to perform the transform. Default is -1.
    use_numexpr : bool, optional
        Whether to use numexpr optimization if available.
        - True: Use numexpr if available
        - False: Use standard numpy implementation
        Default is False.
    kernel : callable, optional
        Mellin transform kernel function with signature kernel(mu, y, q) -> complex.
        Returns kernel coefficients at frequency y.
        Either kernel or log_kernel must be provided (not both).
    log_kernel : callable, optional
        Log-Mellin transform kernel with signature log_kernel(mu, y, q) -> complex.
        Returns LOG of kernel coefficients (preferred for numerical stability).
        Either kernel or log_kernel must be provided (not both).

    Returns
    -------
    a : ndarray
        The inverse transformed output array.

    Raises
    ------
    ValueError
        If neither kernel nor log_kernel is provided, or if both are provided.

    Notes
    -----
    This function computes the inverse of the transformation computed by `fht`.

    Examples
    --------
    >>> import numpy as np
    >>> from fftloggin import fht, ifht
    >>> from fftloggin.kernels import bessel_mellin_log_kernel
    >>> r = np.logspace(-3, 3, 128)
    >>> dln = np.log(r[1]/r[0])
    >>> a = np.exp(-(r/1.0)**2)
    >>> A = fht(a, dln, mu=0, log_kernel=bessel_mellin_log_kernel)
    >>> a_reconstructed = ifht(A, dln, mu=0, log_kernel=bessel_mellin_log_kernel)
    >>> np.allclose(a, a_reconstructed)
    True
    """
    # Use appropriate implementation
    if use_numexpr and NUMEXPR_AVAILABLE:
        return _ifht_numexpr(A, dln, mu, offset, bias, axis, kernel, log_kernel)
    elif use_numexpr and not NUMEXPR_AVAILABLE:
        warnings.warn(
            "numexpr requested but not available, using standard implementation",
            RuntimeWarning,
            stacklevel=2,
        )
        return _ifht(A, dln, mu, offset, bias, axis, kernel, log_kernel)
    else:
        return _ifht(A, dln, mu, offset, bias, axis, kernel, log_kernel)
