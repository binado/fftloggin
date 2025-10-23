"""
Grid utilities for FFTLog transforms.

This module provides the Grid class, which combines coordinate arrays,
transform algorithm, and data storage in a single workspace object.
"""

from typing import Self

import numpy as np
import numpy.typing as npt

from .fftlog import FFTLog
from .kernels import Kernel


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
    if len(x) < 2:
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


def corresponding_k(
    r: npt.ArrayLike,
    logc: npt.ArrayLike,
) -> np.ndarray:
    """
    Compute k array corresponding to r array.

    Follows scipy's convention: k = exp(logc) / r[::-1]
    where logc is the log-center parameter (equivalent to offset in scipy).

    Parameters
    ----------
    r : array_like
        Input radial coordinates (must be log-spaced).
    logc : array_like
        Log-center parameter: log(k_c * r_c). In scipy, this was called 'offset'.
    dlog : float
        Logarithmic spacing (unused but kept for API consistency).

    Returns
    -------
    k : ndarray
        Output wavenumber coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 128)
    >>> dlog = infer_dlog(r)
    >>> logc = 0.0
    >>> k = corresponding_k(r, logc, dlog)
    """
    r = np.asarray(r)
    # Scipy formula: k = exp(offset) / r[::-1]
    # In our API: offset → logc
    return np.exp(logc) / r[::-1]


def corresponding_r(
    k: npt.ArrayLike,
    logc: npt.ArrayLike,
) -> np.ndarray:
    """
    Compute r array corresponding to k array.

    Inverse of corresponding_k() - given output coordinates k,
    compute input coordinates r.

    Follows scipy's convention: r = exp(logc) / k[::-1]

    Parameters
    ----------
    k : array_like
        Output wavenumber coordinates (must be log-spaced).
    logc : array_like
        Log-center parameter: log(k_c * r_c). In scipy, this was called 'offset'.
    dlog : float
        Logarithmic spacing (unused but kept for API consistency).

    Returns
    -------
    r : ndarray
        Input radial coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.logspace(-2, 2, 128)
    >>> dlog = infer_dlog(k)
    >>> logc = 0.0
    >>> r = corresponding_r(k, logc, dlog)
    """
    k = np.asarray(k)
    # Inverse of corresponding_k: r = exp(logc) / k[::-1]
    return np.exp(logc) / k[::-1]


class Grid:
    """
    Workspace for FFTLog transforms with coordinate and data management.

    The Grid class bundles together:
    - Coordinate arrays (r, k)
    - Transform algorithm (FFTLog instance)
    - Data storage (ar, ak properties)

    This provides a stateful, workspace-style API where users work with
    a single object that manages both coordinates and transformed data.

    Attributes
    ----------
    r : ndarray
        Input radial coordinates (log-spaced).
    k : ndarray
        Output wavenumber coordinates (log-spaced).
    ar : ndarray
        Input data on r grid (set after .forward() or via property setter).
    ak : ndarray
        Transformed data on k grid (set after .forward() or .inverse()).
    fftlog : FFTLog
        The underlying transform algorithm.
    n : int
        Number of points.
    dlog : float
        Logarithmic spacing.

    Examples
    --------
    >>> import numpy as np
    >>> from fftloggin.grids import Grid
    >>> from fftloggin.kernels import BesselJKernel

    >>> # Create from existing r array
    >>> r = np.logspace(-2, 2, 128)
    >>> grid = Grid.from_r(r, kernel=BesselJKernel(0))

    >>> # Transform and store
    >>> a = np.exp(-(grid.r/1.0)**2)
    >>> A = grid.forward(a)

    >>> # Access results
    >>> print(grid.ar)  # Input data
    >>> print(grid.ak)  # Transformed data
    >>> print(grid.k)   # Output coordinates

    >>> # Inverse transform
    >>> grid2 = Grid.from_k(grid.k, kernel=BesselJKernel(0))
    >>> a_reconstructed = grid2.inverse(grid.ak)
    >>> np.allclose(a_reconstructed, a)  # Round-trip
    True
    """

    def __init__(
        self,
        r: npt.ArrayLike,
        k: npt.ArrayLike,
        fftlog: FFTLog,
    ):
        """
        Direct constructor (typically use from_r() or from_k() instead).

        Parameters
        ----------
        r : array_like
            Input radial coordinates.
        k : array_like
            Output wavenumber coordinates.
        fftlog : FFTLog
            Configured transform instance.
        """
        self.r = np.asarray(r)
        self.k = np.asarray(k)
        self.fftlog = fftlog

        # Data storage (initially empty)
        self._ar = None
        self._ak = None

    @classmethod
    def from_r(
        cls,
        r: npt.ArrayLike,
        kernel: Kernel,
        bias: float = 0.0,
        minimize_ringing: bool = True,
        logc: float | None = None,
    ) -> Self:
        """
        Create Grid from input radial coordinates.

        Parameters
        ----------
        r : array_like
            Input radial coordinates (must be log-spaced).
        kernel : Kernel
            Mellin transform kernel.
        bias : float, optional
            Power-law bias exponent.
        minimize_ringing : bool, optional
            Whether to snap logc to low-ringing condition.
        logc : float, optional
            Log-center parameter. If None, uses optimal value.

        Returns
        -------
        Grid
            Configured grid with r and k arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.grids import Grid
        >>> from fftloggin.kernels import BesselJKernel
        >>> r = np.logspace(-2, 2, 128)
        >>> grid = Grid.from_r(r, kernel=BesselJKernel(0))
        """
        r = np.asarray(r)
        n = len(r)
        dlog = infer_dlog(r)

        # Create FFTLog instance
        fftlog = FFTLog(
            kernel=kernel,
            n=n,
            dlog=dlog,
            bias=bias,
            minimize_ringing=minimize_ringing,
            logc=logc if logc is not None else 0.0,
        )

        # Generate corresponding k array
        k = corresponding_k(r, fftlog.logc)

        return cls(r, k, fftlog)

    @classmethod
    def from_k(
        cls,
        k: npt.ArrayLike,
        kernel: Kernel,
        bias: float = 0.0,
        minimize_ringing: bool = True,
        logc: float | None = None,
    ) -> Self:
        """
        Create Grid from output wavenumber coordinates.

        Symmetric to from_r() - use this for inverse transforms where
        you already have the k array.

        Parameters
        ----------
        k : array_like
            Output wavenumber coordinates (must be log-spaced).
        kernel : Kernel
            Mellin transform kernel.
        bias : float, optional
            Power-law bias exponent.
        minimize_ringing : bool, optional
            Whether to snap logc to low-ringing condition.
        logc : float, optional
            Log-center parameter. If None, uses optimal value.

        Returns
        -------
        Grid
            Configured grid with r and k arrays.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.grids import Grid
        >>> from fftloggin.kernels import BesselJKernel
        >>> k = np.logspace(-2, 2, 128)
        >>> grid = Grid.from_k(k, kernel=BesselJKernel(0))
        """
        from .fftlog import FFTLog

        k = np.asarray(k)
        n = len(k)
        dlog = infer_dlog(k)

        fftlog = FFTLog(
            kernel=kernel,
            n=n,
            dlog=dlog,
            bias=bias,
            minimize_ringing=minimize_ringing,
            logc=logc if logc is not None else 0.0,
        )

        # Generate corresponding r array
        r = corresponding_r(k, fftlog.logc)

        return cls(r, k, fftlog)

    @classmethod
    def from_fftlog(
        cls,
        fftlog: FFTLog,
        r_center: float = 1.0,
    ) -> Self:
        """
        Create Grid from existing FFTLog instance.

        Reads n, dlog from the FFTLog and generates r, k arrays.

        Parameters
        ----------
        fftlog : FFTLog
            Configured FFTLog instance.
        r_center : float, optional
            Central value of r grid (default: 1.0).

        Returns
        -------
        Grid
            Configured grid with generated r and k arrays.

        Examples
        --------
        >>> from fftloggin import FFTLog
        >>> from fftloggin.grids import Grid
        >>> from fftloggin.kernels import BesselJKernel
        >>> fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)
        >>> grid = Grid.from_fftlog(fftlog, r_center=1.0)
        >>> ic = (128 - 1) // 2
        >>> np.isclose(grid.r[ic], 1.0)  # Central value
        True
        """
        n = fftlog.n
        dlog = fftlog.dlog

        # Create r grid centered at r_center
        i = np.arange(n)
        ic = (n - 1) / 2
        r = r_center * np.exp((i - ic) * dlog)

        # Generate corresponding k array using scipy formula
        k = corresponding_k(r, fftlog.logc)

        return cls(r, k, fftlog)

    @property
    def n(self) -> int:
        """Number of sampling points (delegated to fftlog.n)."""
        return self.fftlog.n

    @property
    def dlog(self) -> float:
        """Logarithmic spacing (delegated to fftlog.dlog)."""
        return self.fftlog.dlog

    @property
    def ar(self) -> np.ndarray:
        """
        Input data on r grid.

        Raises
        ------
        ValueError
            If no input data has been set.
        """
        if self._ar is None:
            raise ValueError(
                "No input data available. Use .forward(a) or set .ar = a first."
            )
        return self._ar

    @ar.setter
    def ar(self, value: npt.ArrayLike):
        """
        Set input data on r grid.

        Parameters
        ----------
        value : array_like
            Input data. Must have shape (..., n) where n matches grid size.

        Raises
        ------
        ValueError
            If array shape doesn't match grid size.
        """
        self._ar = np.asarray(value)
        if self._ar.shape[-1] != self.n:
            raise ValueError(
                f"Input array has wrong shape. Expected (..., {self.n}), "
                f"got (..., {self._ar.shape[-1]})"
            )

    @property
    def ak(self) -> np.ndarray:
        """
        Transformed data on k grid.

        Raises
        ------
        ValueError
            If no transformed data has been computed.
        """
        if self._ak is None:
            raise ValueError(
                "No transformed data available. Use .forward(a) or .inverse(ak) first."
            )
        return self._ak

    @ak.setter
    def ak(self, value: npt.ArrayLike):
        """
        Set transformed data on k grid.

        Parameters
        ----------
        value : array_like
            Output data. Must have shape (..., n) where n matches grid size.

        Raises
        ------
        ValueError
            If array shape doesn't match grid size.
        """
        self._ak = np.asarray(value)
        if self._ak.shape[-1] != self.n:
            raise ValueError(
                f"Output array has wrong shape. Expected (..., {self.n}), "
                f"got (..., {self._ak.shape[-1]})"
            )

    def forward(self, ar: npt.ArrayLike | None = None, **kwargs) -> np.ndarray:
        """
        Perform forward transform: a(r) -> A(k).

        Stores both input and output in .ar and .ak attributes, then returns
        the output array.

        Parameters
        ----------
        ar : array_like, optional
            Input data on r grid. If None, uses previously set .ar.
        **kwargs
            Additional arguments passed to FFTLog.forward().

        Returns
        -------
        ak : ndarray
            Transformed data on k grid.

        Raises
        ------
        ValueError
            If no input data is provided and .ar is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.grids import Grid
        >>> from fftloggin.kernels import BesselJKernel
        >>> r = np.logspace(-2, 2, 128)
        >>> grid = Grid.from_r(r, kernel=BesselJKernel(0))
        >>> a = np.exp(-(grid.r/1.0)**2)
        >>> A = grid.forward(a)
        >>> print(A.shape)
        (128,)

        >>> # Or set data first, then transform
        >>> grid.ar = a
        >>> A = grid.forward()
        """
        if ar is not None:
            self.ar = ar  # Use setter for validation

        if self._ar is None:
            raise ValueError("No input data. Provide ar argument or set .ar first.")

        self._ak = self.fftlog.forward(self._ar, **kwargs)
        return self._ak

    def inverse(self, ak: npt.ArrayLike | None = None, **kwargs) -> np.ndarray:
        """
        Perform inverse transform: A(k) -> a(r).

        Stores both input and output in .ak and .ar attributes, then returns
        the output array.

        Parameters
        ----------
        ak : array_like, optional
            Input data on k grid. If None, uses previously set .ak.
        **kwargs
            Additional arguments passed to FFTLog.inverse().

        Returns
        -------
        ar : ndarray
            Inverse transformed data on r grid.

        Raises
        ------
        ValueError
            If no input data is provided and .ak is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from fftloggin.grids import Grid
        >>> from fftloggin.kernels import BesselJKernel
        >>> k = np.logspace(-2, 2, 128)
        >>> grid = Grid.from_k(k, kernel=BesselJKernel(0))
        >>> A = np.exp(-(grid.k/1.0)**2)
        >>> a = grid.inverse(A)
        """
        if ak is not None:
            self.ak = ak  # Use setter for validation

        if self._ak is None:
            raise ValueError("No output data. Provide ak argument or set .ak first.")

        self._ar = self.fftlog.inverse(self._ak, **kwargs)
        return self._ar

    def clear(self) -> Self:
        """
        Clear stored data (but keep coordinates and transform).

        Returns
        -------
        self
            For method chaining.
        """
        self._ar = None
        self._ak = None
        return self

    def copy(self) -> Self:
        """
        Create a copy of this Grid.

        Returns
        -------
        Grid
            New Grid instance with copied data.
        """
        new_grid = Grid(self.r.copy(), self.k.copy(), self.fftlog)
        if self._ar is not None:
            new_grid._ar = self._ar.copy()
        if self._ak is not None:
            new_grid._ak = self._ak.copy()
        return new_grid

    def __repr__(self) -> str:
        has_ar = self._ar is not None
        has_ak = self._ak is not None
        return (
            f"Grid(n={self.n}, dlog={self.dlog:.6f}, "
            f"r=[{self.r.min():.3e}, {self.r.max():.3e}], "
            f"k=[{self.k.min():.3e}, {self.k.max():.3e}], "
            f"ar={'set' if has_ar else 'unset'}, "
            f"ak={'set' if has_ak else 'unset'})"
        )
