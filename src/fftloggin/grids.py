"""
Grid utilities for FFTLog transforms.

This module provides the Grid class for managing log-spaced coordinate arrays,
along with helper functions for coordinate transformations.
"""

import numpy as np
import numpy.typing as npt


def infer_dlog(x: npt.ArrayLike, rtol: float = 1e-5) -> npt.NDArray:
    """
    Infer logarithmic spacing from array.

    Parameters
    ----------
    x : array_like
        Logarithmically-spaced array with shape (..., n).
    rtol : float, optional
        Relative tolerance for checking uniform spacing. Default is 1e-5.

    Returns
    -------
    dlog : ndarray
        Uniform logarithmic spacing: dlog = log(x[1]/x[0]).
        Shape is () for 1D input or (*batch_shape,) for batched input.

    Raises
    ------
    ValueError
        If array is not uniformly log-spaced or has fewer than 2 elements.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 128)
    >>> dlog = infer_dlog(r)
    >>> print(f"{dlog:.6f}")
    0.072522

    Notes
    -----
    For use with FFTLog batching, reshape the output to add a trailing singleton:
    `dlog.reshape(-1, 1)` or use `prepare_batch_params()`.
    """
    x = np.asarray(x)
    logx = np.log(x)
    dlog = (logx[..., -1] - logx[..., 0]) / (x.shape[-1] - 1)
    dlog_arr = np.diff(logx)
    batch_shape = dlog_arr.shape[:-1]
    dlog_broadcast = dlog.reshape(*batch_shape, 1)

    if not np.allclose(dlog_arr, dlog_broadcast, rtol=rtol):
        raise ValueError(
            f"Array is not uniformly log-spaced. "
            f"Expected spacing: {np.array2string(np.asarray(dlog), precision=3)}, "
            f"got range: [{np.array2string(dlog_arr.min(), precision=3)}, "
            f"{np.array2string(dlog_arr.max(), precision=3)}]"
        )

    return dlog


def get_array_center(x: npt.ArrayLike) -> npt.NDArray:
    """
    Compute the geometric center of a log-spaced array.

    Parameters
    ----------
    x : array_like
        Log-spaced array.

    Returns
    -------
    center : ndarray
        Geometric center: sqrt(x_min * x_max)
    """
    x = np.asarray(x)
    return np.sqrt(x[..., 0] * x[..., -1])


def get_other_array(
    x: npt.ArrayLike,
    logc: npt.ArrayLike,
) -> npt.NDArray:
    """
    Compute the corresponding coordinate array given one coordinate array and logc.

    This function is symmetric with respect to the roles of x and y, so it computes:
    y = exp(logc) / x[::-1]

    This can be used to compute:
    - k from r: get_other_array(r, logc) → k
    - r from k: get_other_array(k, logc) → r

    Follows scipy's convention for the transformation.

    Parameters
    ----------
    x : array_like
        Input log-spaced coordinate array with shape (..., n).
    logc : array_like
        Log-center parameter: log(y_c * x_c). In scipy, this was called 'offset'.
        Can be scalar or array with shape (*batch_shape,) or (*batch_shape, 1)
        for batch operations.

    Returns
    -------
    y : ndarray
        Output coordinate array with shape (..., n) matching x, or
        (*batch_shape, n) if logc has batch dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 128)
    >>> logc = 0.0
    >>> k = get_other_array(r, logc)  # Compute k from r
    >>> r_reconstructed = get_other_array(k, logc)  # Reconstruct r from k
    >>> np.allclose(r_reconstructed, r)
    True

    With batched logc:

    >>> logc_batch = np.array([0.0, 1.0]).reshape(-1, 1)
    >>> k_batch = get_other_array(r, logc_batch)
    >>> k_batch.shape
    (2, 128)
    """
    x = np.asarray(x)
    # Symmetric formula: y = exp(logc) / x[::-1]
    return np.exp(logc) / x[..., ::-1]


def infer_logc(
    x: npt.ArrayLike,
    logc: npt.ArrayLike | None = None,
    ycenter: npt.ArrayLike | None = None,
    ymax: npt.ArrayLike | None = None,
    ymin: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Infer log-center parameter from coordinate array and one of several convenience arguments.

    Given a log-spaced coordinate array x and one convenience argument (logc, ycenter, ymax, or ymin),
    compute the log-center parameter logc that determines the corresponding y array via:
    y = exp(logc) / x[::-1]

    This function is symmetric with respect to the roles of x and y, so it can be used for both:
    - from_r with x=r, y=k (computing the k array corresponding to r)
    - from_k with x=k, y=r (computing the r array corresponding to k)

    Parameters
    ----------
    x : array_like
        Log-spaced coordinate array.
    logc : array_like, optional
        Log-center parameter: log(y_c * x_c). Use directly if provided.
        Can be a scalar or array for batch operations.
    ycenter : array_like, optional
        Central y value. Converts to logc using x_center = sqrt(x_min * x_max).
        Can be a scalar or array for batch operations.
    ymax : array_like, optional
        Maximum y value. Converts to logc using y_max * x_min = exp(logc).
        Can be a scalar or array for batch operations.
    ymin : array_like, optional
        Minimum y value. Converts to logc using y_min * x_max = exp(logc).
        Can be a scalar or array for batch operations.

    Returns
    -------
    logc : ndarray
        The log-center parameter. Shape () for scalar inputs or (*batch_shape,)
        for batched inputs.

    Raises
    ------
    ValueError
        If all optional arguments are None, or if x array is invalid.

    Notes
    -----
    Arguments are checked in order: logc → ycenter → ymax → ymin.
    The first non-None value is used to compute logc.

    For use with FFTLog batching, reshape the output to add a trailing singleton:
    `logc.reshape(-1, 1)` or use `prepare_batch_params()`.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 128)
    >>> # Use logc directly
    >>> logc1 = infer_logc(r, logc=0.0)
    >>> # Use ycenter (k_center when x is r)
    >>> logc2 = infer_logc(r, ycenter=1.0)
    >>> # Use ymax (k_max when x is r)
    >>> logc3 = infer_logc(r, ymax=100.0)

    Batched inputs:

    >>> ycenter_batch = np.array([0.5, 1.0, 2.0])
    >>> logc_batch = infer_logc(r, ycenter=ycenter_batch)
    >>> logc_batch.shape
    (3,)
    """
    x = np.asarray(x)
    xmin = x[..., 0]
    xmax = x[..., -1]
    xcenter = np.sqrt(xmin * xmax)

    if logc is not None:
        return np.asarray(logc)
    elif ycenter is not None:
        return np.asarray(np.log(ycenter * xcenter))
    elif ymax is not None:
        return np.asarray(np.log(ymax * xmin))
    elif ymin is not None:
        return np.asarray(np.log(ymin * xmax))
    else:
        raise ValueError(
            "One of 'logc', 'ycenter', 'ymax', or 'ymin' must be provided. "
            "All arguments are None."
        )


class Grid:
    """
    Container for log-spaced coordinate grids.

    The Grid class holds paired coordinate arrays (r, k) related via the
    FFTLog transformation: k = exp(logc) / r[::-1]

    This provides a simple, stateless container for coordinate arrays,
    keeping them synchronized and providing utility properties.

    Attributes
    ----------
    r : ndarray
        Input radial coordinates (log-spaced).
    k : ndarray
        Output wavenumber coordinates (log-spaced).
    n : int
        Number of points.
    dlog : ndarray
        Logarithmic spacing.
    kr : ndarray
        Product k*r at the geometric center of the grid: k_c * r_c.
    logc : ndarray
        Log-center parameter: log(k_c * r_c).
    rcenter : ndarray
        Central r value: sqrt(r_min * r_max).
    kcenter : ndarray
        Central k value: sqrt(k_min * k_max).

    Examples
    --------
    >>> import numpy as np
    >>> from fftloggin.grids import Grid, infer_logc, get_other_array
    >>> from fftloggin.fftlog import FFTLog
    >>> from fftloggin.kernels import BesselJKernel

    >>> # Create a grid from r array
    >>> r = np.logspace(-2, 2, 128)
    >>> logc = infer_logc(r, logc=0.0)
    >>> k = get_other_array(r, logc)
    >>> grid = Grid(r, k)

    >>> # Or use FFTLog.create_grid() for convenience
    >>> fftlog = FFTLog.from_array(r, BesselJKernel(0), kr=1.0)
    >>> grid = fftlog.create_grid(r=r)

    >>> # Access grid properties
    >>> grid.n
    128
    >>> f"{grid.dlog:.6f}"
    '0.072522'
    >>> f"{grid.rcenter:.1f}"
    '1.0'
    >>> isinstance(grid.kcenter, (float, np.floating))
    True
    """

    def __init__(
        self,
        r: npt.ArrayLike,
        k: npt.ArrayLike,
    ) -> None:
        """
        Create a Grid from coordinate arrays.

        Parameters
        ----------
        r : array_like
            Input radial coordinates (must be log-spaced).
        k : array_like
            Output wavenumber coordinates (must be log-spaced).
        """
        self._setup(r, k)

    def _setup(self, r: npt.ArrayLike, k: npt.ArrayLike) -> None:
        self._r = np.asarray(r)
        self._k = np.asarray(k)

        # Validate that arrays have the same length
        nr = self._r.shape[-1]
        nk = self._k.shape[-1]
        if nr != nk:
            raise ValueError(
                f"r and k arrays must have the same length. Got r={nr}, k={nk}"
            )
        if nr < 2:
            raise ValueError("r and k arrays must have at least 2 elements")
        self._n = nr

        # Validate that arrays are log-spaced
        dlog_r = infer_dlog(self._r)
        dlog_k = infer_dlog(self._k)
        if not np.allclose(dlog_r, dlog_k):
            raise ValueError("r and k arrays must have the same log-spacing")

        self._dlog = dlog_r
        self._rcenter = np.sqrt(self._r[..., 0] * self._r[..., -1])
        self._kcenter = np.sqrt(self._k[..., 0] * self._k[..., -1])
        self._logc = np.log(self._rcenter * self._kcenter)

    @property
    def r(self) -> npt.NDArray:
        return self._r

    @r.setter
    def r(self, other: npt.ArrayLike) -> None:
        k = get_other_array(other, self.logc)
        self._setup(other, k)

    @property
    def k(self) -> npt.NDArray:
        return self._k

    @k.setter
    def k(self, other: npt.ArrayLike) -> None:
        r = get_other_array(other, self.logc)
        self._setup(r, other)

    @property
    def n(self) -> int:
        """Number of sampling points."""
        return self._n

    @property
    def dlog(self) -> npt.NDArray:
        """Logarithmic spacing."""
        return self._dlog

    @property
    def logc(self) -> npt.NDArray:
        """Log-center parameter: log(k_c * r_c)."""
        return self._logc

    @property
    def kr(self) -> npt.NDArray:
        """Product k*r at the geometric center of the grid: k_c * r_c."""
        return np.exp(self._logc)

    @property
    def rcenter(self) -> npt.NDArray:
        """Central r value: sqrt(r_min * r_max)."""
        return self._rcenter

    @property
    def kcenter(self) -> npt.NDArray:
        """Central k value: sqrt(k_min * k_max)."""
        return self._kcenter

    def __repr__(self) -> str:
        return (
            f"Grid(n={self.n}, dlog={self.dlog:.6f}, "
            f"r=[{self.r.min():.3e}, {self.r.max():.3e}], "
            f"k=[{self.k.min():.3e}, {self.k.max():.3e}])"
        )
