from . import kernels
from .exceptions import ArgumentOutOfDomainError, DomainCheckWarning
from .fft import (
    FFTBackend,
    FFTWorkspace,
    NumPyFFTBackend,
    SciPyFFTBackend,
)
from .fftlog import FFTLog
from .grids import Grid
from .kernels import BesselJKernel, Derivative, Kernel
from .utils import prepare_batch_params


def __getattr__(name: str) -> type[FFTBackend]:
    if name == "PyFFTWBackend":
        try:
            from .fft import PyFFTWBackend
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AttributeError(
                "PyFFTWBackend requires optional dependency 'pyfftw'. "
                'Install with `pip install "fftloggin[fftw]"` or `uv add "fftloggin[fftw]"`.'
            ) from exc
        return PyFFTWBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "ArgumentOutOfDomainError",
    "DomainCheckWarning",
    "FFTBackend",
    "FFTWorkspace",
    "FFTLog",
    "Grid",
    "Kernel",
    "BesselJKernel",
    "Derivative",
    "NumPyFFTBackend",
    "prepare_batch_params",
    "SciPyFFTBackend",
    "kernels",
)
