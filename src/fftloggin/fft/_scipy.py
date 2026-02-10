"""SciPy FFT backend."""

from __future__ import annotations

import numpy.typing as npt
from scipy.fft import irfft, rfft

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
        result = rfft(x, n=n, overwrite_x=overwrite_x, **kwargs)
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
        result = irfft(x, n=n, overwrite_x=overwrite_x, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)
