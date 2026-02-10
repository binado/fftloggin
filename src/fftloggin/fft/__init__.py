"""FFT backend protocol and implementations."""

from __future__ import annotations

from ._numpy import NumPyFFTBackend
from ._protocol import (
    FFTBackend,
    FFTWorkspace,
    _complex_dtype,
    _copy_to_out,
    _real_dtype_from_complex,
)
from ._scipy import SciPyFFTBackend

try:
    from ._pyfftw import PyFFTWBackend

    _HAVE_PYFFTW = True
except ImportError:  # pragma: no cover - depends on optional pyfftw install
    PyFFTWBackend = None  # type: ignore[assignment,misc]
    _HAVE_PYFFTW = False

DEFAULT_FFT_BACKEND_FACTORY: type[FFTBackend]
if _HAVE_PYFFTW:
    DEFAULT_FFT_BACKEND_FACTORY = PyFFTWBackend
else:
    DEFAULT_FFT_BACKEND_FACTORY = SciPyFFTBackend

__all__ = [
    "FFTBackend",
    "FFTWorkspace",
    "NumPyFFTBackend",
    "PyFFTWBackend",
    "SciPyFFTBackend",
    "DEFAULT_FFT_BACKEND_FACTORY",
    "_complex_dtype",
    "_copy_to_out",
    "_real_dtype_from_complex",
]
