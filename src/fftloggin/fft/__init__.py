"""FFT backend protocol and implementations."""

from __future__ import annotations

from .numpy import NumPyFFTBackend
from .protocol import (
    FFTBackend,
    FFTWorkspace,
    _complex_dtype,
    _copy_to_out,
    _real_dtype_from_complex,
)
from .scipy import SciPyFFTBackend

DEFAULT_FFT_BACKEND_FACTORY: type[FFTBackend]
# Use SciPy as the default backend for stable cross-platform behavior.
# PyFFTW remains available as an explicit opt-in backend.
DEFAULT_FFT_BACKEND_FACTORY = SciPyFFTBackend


def __getattr__(name: str):
    if name == "PyFFTWBackend":
        try:
            from .pyfftw import PyFFTWBackend
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AttributeError(
                "PyFFTWBackend requires optional dependency 'pyfftw'. "
                'Install with `pip install "fftloggin[fftw]"` or `uv add "fftloggin[fftw]"`.'
            ) from exc
        return PyFFTWBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FFTBackend",
    "FFTWorkspace",
    "NumPyFFTBackend",
    "SciPyFFTBackend",
    "DEFAULT_FFT_BACKEND_FACTORY",
    "_complex_dtype",
    "_copy_to_out",
    "_real_dtype_from_complex",
]
