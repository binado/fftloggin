"""FFT backend protocol, shared workspace, and dtype helpers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


def _copy_to_out(out: npt.NDArray, result: npt.NDArray) -> npt.NDArray:
    """Copy *result* into the pre-allocated *out* array and return it.

    Raises ``ValueError`` if shapes differ; uses ``"same_kind"`` casting.
    """
    if out.shape != result.shape:
        raise ValueError(f"out has shape {out.shape}, expected {result.shape}.")
    np.copyto(out, result, casting="same_kind")
    return out


def _complex_dtype(real_dtype: npt.DTypeLike) -> np.dtype:
    """Return the complex dtype that corresponds to *real_dtype*.

    ``float32`` → ``complex64``; ``float64`` → ``complex128``.
    """
    return np.dtype(np.result_type(real_dtype, np.complex64))


def _real_dtype_from_complex(complex_dtype: npt.DTypeLike) -> np.dtype:
    """Return the real dtype that is the component type of *complex_dtype*.

    ``complex64`` → ``float32``; ``complex128`` → ``float64``.
    """
    return np.empty((), dtype=complex_dtype).real.dtype


class FFTWorkspace:
    """Cache for FFT plans and aligned buffers.

    Passed to :class:`FFTBackend` methods so that backends that support
    plan caching (e.g. PyFFTW) can reuse previously computed plans and
    aligned memory across repeated calls with the same transform size.

    Attributes
    ----------
    buffers : dict
        Maps a hashable key (typically ``(n, dtype)``) to a ``(input, output)``
        pair of aligned NumPy arrays.
    plans : dict
        Maps the same key to a backend-specific plan object (e.g. a
        ``pyfftw.FFTW`` instance).

    Methods
    -------
    clear()
        Discard all cached buffers and plans.
    """

    def __init__(self) -> None:
        self.buffers: dict[tuple, tuple[npt.NDArray, npt.NDArray]] = {}
        self.plans: dict[tuple, Any] = {}

    def clear(self) -> None:
        self.buffers.clear()
        self.plans.clear()


class FFTBackend(Protocol):
    """Protocol for FFT backends used by FFTLog."""

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        """Compute the real-input FFT along the last axis.

        Parameters
        ----------
        x : array_like
            Real-valued input array.
        n : int, optional
            Transform size. Defaults to ``x.shape[-1]``.
        out : ndarray, optional
            Pre-allocated output array for the complex spectrum.
        workspace : FFTWorkspace, optional
            Cache for plans and aligned buffers (used by PyFFTW backend).
        overwrite_x : bool, optional
            If ``True``, the input may be destroyed (used by SciPy backend).
        **kwargs
            Additional keyword arguments forwarded to the underlying FFT library.

        Returns
        -------
        ndarray
            Complex array of shape ``(..., n // 2 + 1)``.
        """
        ...

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        """Compute the inverse real-input FFT along the last axis.

        Parameters
        ----------
        x : array_like
            Complex-valued input array (one-sided spectrum).
        n : int, optional
            Length of the output. Defaults to ``2 * (x.shape[-1] - 1)``.
        out : ndarray, optional
            Pre-allocated real output array.
        workspace : FFTWorkspace, optional
            Cache for plans and aligned buffers (used by PyFFTW backend).
        overwrite_x : bool, optional
            If ``True``, the input may be destroyed (used by SciPy backend).
        **kwargs
            Additional keyword arguments forwarded to the underlying FFT library.

        Returns
        -------
        ndarray
            Real array of shape ``(..., n)``.
        """
        ...
