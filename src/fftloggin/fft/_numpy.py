"""NumPy FFT backend."""

from __future__ import annotations

import numpy.typing as npt
from numpy.fft import irfft as numpy_irfft
from numpy.fft import rfft as numpy_rfft

from ._protocol import FFTWorkspace


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
        _ = workspace, overwrite_x
        return numpy_rfft(x, n=n, out=out, **kwargs)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace, overwrite_x
        return numpy_irfft(x, n=n, out=out, **kwargs)
