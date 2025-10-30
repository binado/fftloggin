"""
Cosmology utilities for FFTLog transforms.

This module provides tools for computing line-of-sight integrals and other
cosmological transforms using the FFTLog algorithm.
"""

import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from .fftlog import FFTLog
from .grids import Grid, infer_dlog
from .kernels import SphericalBesselJKernel

__all__ = ("RadialIntegrator",)


class RadialIntegrator:
    r"""
    Compute radial integrals for angular power spectrum calculations.

    This class performs line-of-sight integrals of the form:

    .. math::

        \Delta_\ell(k) = \int d\chi\, W(\chi) T(\chi,k) j_\ell(k\chi)

    using the FFTLog algorithm with spherical Bessel function kernels.
    The integration is performed efficiently via fast Hankel transforms.

    Parameters
    ----------
    chi : array_like
        Comoving distance array (will be resampled onto FFTLog grid).
    s : array_like
        Source/window function values evaluated at chi points.
    ells : array_like
        Multipole moments (can be scalar or array for batch computation).
    fftlog_bias : float, optional
        FFTLog bias parameter for numerical stability. Default is 0.0.
    n : int, optional
        Number of FFTLog sampling points. If None, inferred from chi. Default is None.
    dlog : float, optional
        Logarithmic spacing for FFTLog grid. If None, inferred from chi. Default is None.
    recenter : bool or float, optional
        If True, center at chi value corresponding to median of source function.
        If False, center at geometric mean of chi endpoints.
        If float, use as explicit r0 center value. Default is True.
    lowring : bool, optional
        If True, snap kr to minimize ringing artifacts. Default is False.
    compute : bool, optional
        If True, automatically compute the integral in constructor. Default is True.
    **interp_kwargs
        Additional keyword arguments passed to build_interpolator, such as bc_type.

    Attributes
    ----------
    fftlog : FFTLog
        The FFTLog instance used for the transform computation.
    grid : Grid
        The Grid instance with chi (r) and k coordinate arrays.
    source_interpolator : callable
        The interpolator used to resample the source function. Can be set directly
        for custom interpolation schemes.
    s_resampled : ndarray
        Source function resampled onto the FFTLog chi grid (available after compute()).
    result : ndarray
        Computed radial integral (available after compute()). Shape is (n_ells, n_k)
        if ells is an array, or (n_k,) if ells is scalar.
    chi_mask : ndarray
        Boolean mask indicating valid chi values within the input range.
        True where chi is within the original input bounds.
    ells : ndarray
        Multipole moments used in the computation.
    chi : ndarray
        Comoving distance array (same as grid.r).
    k : ndarray
        Wavenumber array (same as grid.k).

    Examples
    --------
    >>> import numpy as np
    >>> from fftloggin.cosmology import RadialIntegrator
    >>>
    >>> # Basic usage with auto-compute
    >>> chi = np.logspace(0, 3, 128)
    >>> s = np.exp(-((chi - 100) / 50)**2)  # Gaussian window
    >>> ells = np.array([0, 10, 20, 30])
    >>> integrator = RadialIntegrator(chi, s, ells, recenter=True)
    >>> print(integrator.result.shape)  # (4, 128)
    >>>
    >>> # Deferred computation
    >>> integrator = RadialIntegrator(chi, s, ells, compute=False)
    >>> result = integrator.compute()  # Compute when ready
    >>>
    >>> # Direct r0 specification
    >>> integrator = RadialIntegrator(chi, s, ells, recenter=150.0)
    >>>
    >>> # Custom interpolation settings
    >>> integrator = RadialIntegrator(chi, s, ells, bc_type='clamped')
    >>>
    >>> # Custom interpolator
    >>> from scipy.interpolate import interp1d
    >>> integrator = RadialIntegrator(chi, s, ells, compute=False)
    >>> integrator.source_interpolator = interp1d(chi, s, kind='linear', fill_value=0.0, bounds_error=False)
    >>> result = integrator.compute()

    Notes
    -----
    - The kr parameter is automatically set to :math:`\ell + 1`, following the
      pyccl convention.
    - When recenter=True, the geometric center of the chi array is moved to the
      chi value corresponding to the median of the source function s.
    - When recenter=False, the geometric center is set to the geometric mean of
      the chi array endpoints.
    - When recenter is a float, it is used directly as the r0 center value.
    - The chi_mask property helps identify which parts of the resampled grid
      correspond to the original input range vs extrapolated regions.
    - The input arrays are always interpolated onto the FFTLog-generated chi grid
      using cubic spline interpolation by default.
    - The source_interpolator can be set directly for custom interpolation schemes.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
    .. [2] Assassi et al., 2017, arXiv:1705.05022
    .. [3] Fang et al., 2020, arXiv:1911.11947

    See Also
    --------
    FFTLog : Core FFTLog transform algorithm
    SphericalBesselJKernel : Spherical Bessel function kernel
    """

    def __init__(
        self,
        chi: npt.ArrayLike,
        s: npt.ArrayLike,
        ells: npt.ArrayLike,
        fftlog_bias: float = 0.0,
        n: int | None = None,
        dlog: npt.ArrayLike | None = None,
        recenter: bool | float = True,
        lowring: bool = False,
        compute: bool = True,
        **interp_kwargs,
    ) -> None:
        # Store and validate inputs
        self._chi_input = np.asarray(chi)
        self._s_input = np.asarray(s)
        self.ells = np.asarray(ells)
        self._interp_kwargs = interp_kwargs

        nchi = self._chi_input.shape[-1]
        ns = self._s_input.shape[-1]

        if nchi != ns:
            raise ValueError(
                f"chi and s must have the same length. Got chi: {nchi}, s: {ns}"
            )

        if nchi < 2:
            raise ValueError("chi array must have at least 2 elements")

        # Auto-infer n and dlog if not provided
        if n is None:
            n = nchi

        if dlog is None:
            # Infer from input chi
            try:
                dlog = infer_dlog(self._chi_input)
            except ValueError:
                # If input is not log-spaced, use a reasonable default
                dlog = np.log(self._chi_input[..., -1] / self._chi_input[..., 0]) / (
                    nchi - 1
                )

        # Determine chi center based on recenter parameter
        chi_center = self._compute_r0(recenter=recenter)

        # Create kernel with vectorized ells
        kernel = SphericalBesselJKernel(self.ells)

        # Set kr = ells + 1 (following pyccl convention)
        kr = self.ells + 1.0

        # For batch transforms, reshape kr to have trailing singleton dimension
        # This enables proper broadcasting in FFTLog without modifying core code
        kr = np.asarray(kr)
        if kr.ndim > 0:
            kr = kr.reshape(-1, 1)

        # Create FFTLog instance with specified center
        self._fftlog = FFTLog(
            kernel=kernel, n=n, dlog=dlog, bias=fftlog_bias, lowring=lowring, kr=kr
        )

        # Create Grid - FFTLog will generate its own chi array
        # Use create_grid with a temporary array to get the proper chi values
        log_chi_center = np.log(chi_center)
        log_chi_min = log_chi_center - (n - 1) / 2.0 * dlog
        chi_fftlog = np.exp(log_chi_min + np.arange(n) * dlog)

        self._grid = self._fftlog.create_grid(r=chi_fftlog)

        # Create mask for valid chi values (within input range)
        self._chi_mask = (self._grid.r >= self._chi_input[..., 0]) & (
            self._grid.r <= self._chi_input[..., -1]
        )

        # Build interpolator
        self._source_interpolator = self.build_interpolator(
            self._chi_input, self._s_input, **self._interp_kwargs
        )

        # Initialize result placeholders
        self._result = None
        self._s_resampled = None

        # Optionally auto-compute
        if compute:
            self.compute()

    def _compute_r0(self, recenter: bool | float) -> float:
        """
        Compute the r0 center value based on recenter parameter.

        Parameters
        ----------
        recenter : bool or float
            If True, find chi value at median of source function.
            If False, use geometric mean of chi endpoints.
            If float, use directly as r0.

        Returns
        -------
        float
            The r0 center value for the FFTLog grid.
        """
        if isinstance(recenter, bool):
            if recenter:
                # Find chi value corresponding to median of source function
                # Use weighted median: where cumulative sum reaches half total
                cdf = cumulative_trapezoid(self._s_input, self._chi_input, initial=0)
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                    median_idx = np.searchsorted(cdf, 0.5)
                    median_idx = min(median_idx, self._chi_input.shape[-1] - 1)
                    return float(self._chi_input[..., median_idx])
                else:
                    # Fallback: use geometric mean of endpoints
                    return float(
                        np.sqrt(self._chi_input[..., 0] * self._chi_input[..., -1])
                    )
            else:
                # Use geometric mean of chi endpoints
                return float(
                    np.sqrt(self._chi_input[..., 0] * self._chi_input[..., -1])
                )
        else:
            # Direct float value
            return float(recenter)

    def build_interpolator(self, chi: npt.ArrayLike, s: npt.ArrayLike, **kwargs):
        """
        Build cubic spline interpolator for resampling source function.

        This method can be overridden for custom interpolation schemes.
        By default, uses scipy's CubicSpline with natural boundary conditions.

        Parameters
        ----------
        chi : array_like
            Input chi coordinates.
        s : array_like
            Source function values at chi coordinates.
        **kwargs
            Additional keyword arguments passed to CubicSpline.
            Common options include bc_type ('natural', 'clamped', 'not-a-knot').

        Returns
        -------
        callable
            Interpolator function that takes chi values and returns interpolated s values.

        Examples
        --------
        >>> integrator = RadialIntegrator(chi, s, ells, bc_type='clamped')
        """
        bc_type = kwargs.pop("bc_type", "natural")
        extrapolate = kwargs.pop("extrapolate", False)
        return CubicSpline(chi, s, bc_type=bc_type, extrapolate=extrapolate, **kwargs)

    def compute(self, **fft_kwargs) -> npt.NDArray:
        """
        Compute the radial integral using the current source interpolator.

        This method resamples the source function onto the FFTLog grid using
        the current interpolator, then performs the FFTLog forward transform.
        Can be called multiple times with different FFT parameters.

        Parameters
        ----------
        **fft_kwargs
            Additional keyword arguments passed to scipy.fft.rfft via
            fftlog.forward(). Common options include 'workers' for parallel FFT.

        Returns
        -------
        ndarray
            The computed radial integral. Also stored in self.result property.
            Shape is (n_ells, n_k) if ells is an array, or (n_k,) if ells is scalar.

        Examples
        --------
        >>> integrator = RadialIntegrator(chi, s, ells, compute=False)
        >>> result1 = integrator.compute()  # First computation
        >>> result2 = integrator.compute(workers=-1)  # Recompute with parallel FFT
        """
        self._s_resampled = self._source_interpolator(self.grid.r)
        self._s_resampled = np.nan_to_num(self._s_resampled)
        result = self.fftlog.forward(self.s_resampled, **fft_kwargs)
        # Divide forward transform by k because of our convention
        # for the fht integration measure kdr
        self._result = result / self.grid.k

        return self._result

    @property
    def fftlog(self) -> FFTLog:
        """FFTLog instance used for the transform."""
        return self._fftlog

    @property
    def grid(self) -> Grid:
        """Grid instance with chi (r) and k coordinate arrays."""
        return self._grid

    @property
    def source_interpolator(self):
        """
        Source function interpolator used for resampling.

        Returns
        -------
        callable
            Interpolator function that takes chi values and returns s values.

        Notes
        -----
        This property can be set directly for custom interpolation schemes.
        Setting a new interpolator invalidates cached results.
        """
        return self._source_interpolator

    @source_interpolator.setter
    def source_interpolator(self, interpolator):
        """
        Set a custom source function interpolator.

        Parameters
        ----------
        interpolator : callable
            Must be callable with signature: interpolator(chi) -> s_values.

        Raises
        ------
        TypeError
            If interpolator is not callable.
        """
        if not callable(interpolator):
            raise TypeError("Interpolator must be callable")
        self._source_interpolator = interpolator
        # Invalidate cached results
        self._result = None
        self._s_resampled = None

    @property
    def s_resampled(self) -> npt.NDArray:
        """
        Source function resampled onto the FFTLog chi grid.

        Returns
        -------
        ndarray
            Resampled source function values.

        Raises
        ------
        ValueError
            If compute() has not been called yet.
        """
        if self._s_resampled is None:
            raise ValueError(
                "Source function not yet resampled. Call compute() first or set "
                "compute=True in constructor."
            )
        return self._s_resampled

    @property
    def result(self) -> npt.NDArray:
        r"""
        Computed radial integral :math:`\Delta_\ell(k)`.

        Returns
        -------
        ndarray
            Shape is (n_ells, n_k) if ells is an array, or (n_k,) if ells is scalar.

        Raises
        ------
        ValueError
            If compute() has not been called yet.
        """
        if self._result is None:
            raise ValueError(
                "Result not yet computed. Call compute() first or set "
                "compute=True in constructor."
            )
        return self._result

    @property
    def chi_mask(self) -> npt.NDArray:
        """
        Boolean mask for valid chi values within the input range.

        Returns
        -------
        ndarray
            Boolean array where True indicates chi values within the original
            input range.
        """
        return self._chi_mask

    @property
    def chi(self) -> npt.NDArray:
        """Comoving distance array (same as grid.r)."""
        return self._grid.r

    @property
    def k(self) -> npt.NDArray:
        """Wavenumber array (same as grid.k)."""
        return self._grid.k
