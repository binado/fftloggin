"""FFT backend protocol, shared workspace, and dtype helpers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


def _copy_to_out(out: npt.NDArray, result: npt.NDArray) -> npt.NDArray:
    if out.shape != result.shape:
        raise ValueError(f"out has shape {out.shape}, expected {result.shape}.")
    np.copyto(out, result, casting="same_kind")
    return out


def _complex_dtype(real_dtype: npt.DTypeLike) -> np.dtype:
    return np.dtype(np.result_type(real_dtype, np.complex64))


def _real_dtype_from_complex(complex_dtype: npt.DTypeLike) -> np.dtype:
    return np.empty((), dtype=complex_dtype).real.dtype


class FFTWorkspace:
    """Cache for FFT plans and aligned buffers."""

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
    ) -> npt.NDArray: ...

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        overwrite_x: bool = False,
        **kwargs,
    ) -> npt.NDArray: ...
