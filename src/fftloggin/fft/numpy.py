"""NumPy FFT backend."""

from __future__ import annotations

import numpy.typing as npt
from numpy.fft import irfft, rfft

from .protocol import FFTWorkspace


class NumPyFFTBackend:
    """FFT backend that delegates to numpy.fft."""

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        """Delegates to ``numpy.fft.rfft``. ``workspace`` and ``overwrite_x`` are ignored."""
        _ = workspace, overwrite_x
        return rfft(x, n=n, out=out, **kwargs)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        """Delegates to ``numpy.fft.irfft``. ``workspace`` and ``overwrite_x`` are ignored."""
        _ = workspace, overwrite_x
        return irfft(x, n=n, out=out, **kwargs)
