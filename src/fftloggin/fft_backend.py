"""FFT backend protocol and implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
from numpy.fft import irfft as numpy_irfft
from numpy.fft import rfft as numpy_rfft
from scipy.fft import irfft as scipy_irfft
from scipy.fft import rfft as scipy_rfft


class FFTBackend(Protocol):
    """Protocol for FFT backends used by FFTLog."""

    def rfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray: ...

    def irfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray: ...


class SciPyFFTBackend:
    """FFT backend that delegates to scipy.fft."""

    def rfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray:
        return scipy_rfft(x, n=n, **kwargs)

    def irfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray:
        return scipy_irfft(x, n=n, **kwargs)


class NumPyFFTBackend:
    """FFT backend that delegates to numpy.fft."""

    def rfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray:
        return numpy_rfft(x, n=n, **kwargs)

    def irfft(self, x: npt.ArrayLike, n: int | None = None, **kwargs) -> np.ndarray:
        return numpy_irfft(x, n=n, **kwargs)


DEFAULT_FFT_BACKEND = SciPyFFTBackend()
