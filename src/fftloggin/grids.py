"""
Grid utilities for FFTLog transforms.

This module provides the Grid class for managing log-spaced coordinate arrays,
along with helper functions for coordinate transformations.
"""

from typing import Self

import numpy as np
import numpy.typing as npt


def infer_dlog(x: npt.ArrayLike) -> float:
    """
    Infer logarithmic spacing from array.

    Parameters
    ----------
    x : array_like
        Logarithmically-spaced array.

    Returns
    -------
    dlog : float
        Uniform logarithmic spacing: dlog = log(x[1]/x[0])

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
    0.031762
    """
    x = np.asarray(x)
    if x.shape[-1] < 2:
        raise ValueError("Array must have at least 2 elements")

    log_x = np.log(x)
    dlog = log_x[1] - log_x[0]
    dlog_arr = np.diff(log_x)

    rtol = 1e-5
    if not np.allclose(dlog_arr, dlog, rtol=rtol):
        raise ValueError(
            f"Array is not uniformly log-spaced. "
            f"Expected spacing: {dlog:.6f}, "
            f"got range: [{dlog_arr.min():.6f}, {dlog_arr.max():.6f}]"
        )

    return float(dlog)


def get_other_array(
    x: npt.ArrayLike,
    logc: float,
) -> np.ndarray:
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
        Input log-spaced coordinate array.
    logc : float
        Log-center parameter: log(y_c * x_c). In scipy, this was called 'offset'.

    Returns
    -------
    y : ndarray
        Output coordinate array.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 128)
    >>> logc = 0.0
    >>> k = get_other_array(r, logc)  # Compute k from r
    >>> r_reconstructed = get_other_array(k, logc)  # Reconstruct r from k
    >>> np.allclose(r_reconstructed, r)
    True
    """
    x = np.asarray(x)
    # Symmetric formula: y = exp(logc) / x[::-1]
    return np.exp(logc) / x[::-1]


def infer_logc(
    x: npt.ArrayLike,
    logc: float | None = None,
    ycenter: float | None = None,
    ymax: float | None = None,
    ymin: float | None = None,
) -> float:
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
    logc : float, optional
        Log-center parameter: log(y_c * x_c). Use directly if provided.
    ycenter : float, optional
        Central y value. Converts to logc using x_center = sqrt(x_min * x_max).
    ymax : float, optional
        Maximum y value. Converts to logc using y_max * x_min = exp(logc).
    ymin : float, optional
        Minimum y value. Converts to logc using y_min * x_max = exp(logc).

    Returns
    -------
    logc : float
        The log-center parameter.

    Raises
    ------
    ValueError
        If all optional arguments are None, or if x array is invalid.

    Notes
    -----
    Arguments are checked in order: logc → ycenter → ymax → ymin.
    The first non-None value is used to compute logc.

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
    """
    x = np.asarray(x)
    x_min = x.min()
    x_max = x.max()
    x_center = np.sqrt(x_min * x_max)

    if logc is not None:
        return float(logc)
    elif ycenter is not None:
        return float(np.log(ycenter * x_center))
    elif ymax is not None:
        return float(np.log(ymax * x_min))
    elif ymin is not None:
        return float(np.log(ymin * x_max))
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
    dlog : float
        Logarithmic spacing.
    logc : float
        Log-center parameter: log(k_c * r_c).
    rcenter : float
        Central r value: sqrt(r_min * r_max).
    kcenter : float
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
    >>> fftlog = FFTLog.from_array(r, BesselJKernel(0), logc=0.0)
    >>> grid = fftlog.create_grid(r=r)

    >>> # Access grid properties
    >>> print(grid.n)       # Number of points
    >>> print(grid.dlog)    # Log spacing
    >>> print(grid.logc)    # Log-center parameter
    >>> print(grid.rcenter) # Central r value
    >>> print(grid.kcenter) # Central k value
    """

    def __init__(
        self,
        r: npt.ArrayLike,
        k: npt.ArrayLike,
    ):
        """
        Create a Grid from coordinate arrays.

        Parameters
        ----------
        r : array_like
            Input radial coordinates (must be log-spaced).
        k : array_like
            Output wavenumber coordinates (must be log-spaced).
        """
        self.r = np.asarray(r)
        self.k = np.asarray(k)

        # Validate that arrays have the same length
        nr = self.r.shape[-1]
        nk = self.r.shape[-1]
        if nr != nk:
            raise ValueError(
                f"r and k arrays must have the same length. Got r={nr}, k={nk}"
            )

    @property
    def n(self) -> int:
        """Number of sampling points."""
        return self.r.shape[-1]

    @property
    def dlog(self) -> float:
        """Logarithmic spacing."""
        return infer_dlog(self.r)

    @property
    def logc(self) -> float:
        """Log-center parameter: log(k_c * r_c)."""
        r_center = self.rcenter
        k_center = self.kcenter
        return float(np.log(k_center * r_center))

    @property
    def rcenter(self) -> float:
        """Central r value: sqrt(r_min * r_max)."""
        return float(np.sqrt(self.r.min() * self.r.max()))

    @property
    def kcenter(self) -> float:
        """Central k value: sqrt(k_min * k_max)."""
        return float(np.sqrt(self.k.min() * self.k.max()))

    def copy(self) -> Self:
        """
        Create a copy of this Grid.

        Returns
        -------
        Grid
            New Grid instance with copied coordinate arrays.
        """
        return Grid(self.r.copy(), self.k.copy())

    def __repr__(self) -> str:
        return (
            f"Grid(n={self.n}, dlog={self.dlog:.6f}, "
            f"r=[{self.r.min():.3e}, {self.r.max():.3e}], "
            f"k=[{self.k.min():.3e}, {self.k.max():.3e}])"
        )
