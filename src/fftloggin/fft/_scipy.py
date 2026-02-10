"""SciPy FFT backend."""

from __future__ import annotations

import numpy.typing as npt
from scipy.fft import irfft as scipy_irfft
from scipy.fft import rfft as scipy_rfft

from ._protocol import FFTWorkspace, _copy_to_out


class SciPyFFTBackend:
    """FFT backend that delegates to scipy.fft."""

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = scipy_rfft(x, n=n, overwrite_x=overwrite_x, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = scipy_irfft(x, n=n, overwrite_x=overwrite_x, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)
