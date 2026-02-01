"""FFT backend protocol and implementations."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from numpy.fft import irfft as numpy_irfft
from numpy.fft import rfft as numpy_rfft
from scipy.fft import irfft as scipy_irfft
from scipy.fft import rfft as scipy_rfft

try:
    import pyfftw

    _HAVE_PYFFTW = True
except Exception:  # pragma: no cover - depends on optional pyfftw install
    pyfftw = None  # type: ignore[assignment]
    _HAVE_PYFFTW = False


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
        self.plans: dict[tuple, object] = {}

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
        **kwargs,
    ) -> npt.NDArray: ...

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray: ...


class SciPyFFTBackend:
    """FFT backend that delegates to scipy.fft."""

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = scipy_rfft(x, n=n, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = scipy_irfft(x, n=n, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)


class NumPyFFTBackend:
    """FFT backend that delegates to numpy.fft."""

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = numpy_rfft(x, n=n, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        _ = workspace
        result = numpy_irfft(x, n=n, **kwargs)
        if out is None:
            return result
        return _copy_to_out(out, result)


class PyFFTWBackend:
    """FFT backend that delegates to pyFFTW with plan and buffer reuse."""

    def __init__(self) -> None:
        if not _HAVE_PYFFTW:  # pragma: no cover - depends on optional pyfftw install
            raise ImportError(
                "pyfftw is required for PyFFTWBackend. Install with `pip install pyfftw`."
            )
        self._workspace = FFTWorkspace()
        self._fftw_kwargs: dict[str, object] = {"normalise_idft": True}

    def _get_workspace(self, workspace: FFTWorkspace | None) -> FFTWorkspace:
        return workspace or self._workspace

    def _get_threads(self, kwargs: dict) -> int:
        threads = kwargs.pop("threads", None)
        if threads is None and "workers" in kwargs:
            threads = kwargs.pop("workers")
        if threads is None and pyfftw is not None:
            threads = getattr(pyfftw.config, "NUM_THREADS", 1)
        return int(threads or 1)

    def _get_flags(self, kwargs: dict) -> tuple:
        planner_effort = kwargs.pop("planner_effort", None)
        flags = kwargs.pop("flags", None)
        if flags is not None:
            if isinstance(flags, tuple):
                return flags
            return (flags,)
        if planner_effort is None:
            planner_effort = "FFTW_MEASURE"
        if isinstance(planner_effort, tuple):
            return planner_effort
        return (planner_effort,)

    def _validate_kwargs(self, kwargs: dict) -> None:
        axes = kwargs.pop("axes", None)
        if axes not in (None, -1, (-1,)):
            raise ValueError(
                "PyFFTWBackend only supports transforms along the last axis."
            )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected keyword arguments for PyFFTWBackend: {unknown}"
            )

    def _get_buffers(
        self,
        workspace: FFTWorkspace,
        direction: Literal["forward", "backward"],
        real_shape: tuple[int, ...],
        complex_shape: tuple[int, ...],
        real_dtype: np.dtype,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        key = (direction, real_shape, complex_shape, real_dtype)
        cached = workspace.buffers.get(key)
        if cached is not None:
            return cached
        if pyfftw is None:  # pragma: no cover - safety check
            raise RuntimeError("pyfftw is not available.")
        if direction == "forward":
            inbuf = pyfftw.empty_aligned(real_shape, dtype=real_dtype)
            outbuf = pyfftw.empty_aligned(
                complex_shape, dtype=_complex_dtype(real_dtype)
            )
        else:
            inbuf = pyfftw.empty_aligned(
                complex_shape, dtype=_complex_dtype(real_dtype)
            )
            outbuf = pyfftw.empty_aligned(real_shape, dtype=real_dtype)
        workspace.buffers[key] = (inbuf, outbuf)
        return inbuf, outbuf

    def _get_plan(
        self,
        workspace: FFTWorkspace,
        direction: str,
        inbuf: npt.NDArray,
        outbuf: npt.NDArray,
        threads: int,
        flags: tuple,
    ) -> object:
        plan_key = (direction, inbuf.shape, outbuf.shape, inbuf.dtype, threads, flags)
        cached = workspace.plans.get(plan_key)
        if cached is not None:
            return cached
        if pyfftw is None:  # pragma: no cover - safety check
            raise RuntimeError("pyfftw is not available.")
        fftw_direction = "FFTW_FORWARD" if direction == "rfft" else "FFTW_BACKWARD"
        plan = pyfftw.FFTW(
            inbuf,
            outbuf,
            axes=(-1,),
            direction=fftw_direction,
            flags=flags,
            threads=threads,
            **self._fftw_kwargs,
        )
        workspace.plans[plan_key] = plan
        return plan

    def rfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        if pyfftw is None:  # pragma: no cover - depends on optional pyfftw install
            raise RuntimeError("pyfftw is not available.")
        threads = self._get_threads(kwargs)
        flags = self._get_flags(kwargs)
        self._validate_kwargs(kwargs)

        x_arr = np.asarray(x)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        if x_arr.dtype.kind != "f" or x_arr.dtype not in (np.float32, np.float64):
            x_arr = x_arr.astype(np.float64, copy=False)
        real_dtype = np.dtype(x_arr.dtype)
        n_in = x_arr.shape[-1]
        n_out = n if n is not None else n_in
        if n_out <= 0:
            raise ValueError("n must be a positive integer.")
        batch_shape = x_arr.shape[:-1]
        in_shape = batch_shape + (n_out,)
        out_shape = batch_shape + (n_out // 2 + 1,)

        ws = self._get_workspace(workspace)
        inbuf, outbuf = self._get_buffers(
            ws,
            "forward",
            real_shape=in_shape,
            complex_shape=out_shape,
            real_dtype=real_dtype,
        )
        inbuf[...] = 0
        if n_in >= n_out:
            inbuf[...] = x_arr[..., :n_out]
        else:
            inbuf[..., :n_in] = x_arr
        plan = self._get_plan(ws, "rfft", inbuf, outbuf, threads, flags)
        plan()
        if out is None:
            return np.array(outbuf, copy=True)
        return _copy_to_out(out, outbuf)

    def irfft(
        self,
        x: npt.ArrayLike,
        n: int | None = None,
        out: npt.NDArray | None = None,
        workspace: FFTWorkspace | None = None,
        **kwargs,
    ) -> npt.NDArray:
        if pyfftw is None:  # pragma: no cover - depends on optional pyfftw install
            raise RuntimeError("pyfftw is not available.")
        threads = self._get_threads(kwargs)
        flags = self._get_flags(kwargs)
        self._validate_kwargs(kwargs)

        x_arr = np.asarray(x)
        if x_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
        if x_arr.dtype.kind != "c" or x_arr.dtype not in (np.complex64, np.complex128):
            x_arr = x_arr.astype(np.complex128, copy=False)
        complex_dtype = np.dtype(x_arr.dtype)
        real_dtype = _real_dtype_from_complex(complex_dtype)
        n_in = x_arr.shape[-1]
        n_out = n if n is not None else 2 * (n_in - 1)
        if n_out <= 0:
            raise ValueError("n must be a positive integer.")
        expected_ns = n_out // 2 + 1
        batch_shape = x_arr.shape[:-1]
        in_shape = batch_shape + (expected_ns,)
        out_shape = batch_shape + (n_out,)

        ws = self._get_workspace(workspace)
        inbuf, outbuf = self._get_buffers(
            ws,
            "backward",
            real_shape=out_shape,
            complex_shape=in_shape,
            real_dtype=real_dtype,
        )
        inbuf[...] = 0
        if n_in >= expected_ns:
            inbuf[...] = x_arr[..., :expected_ns]
        else:
            inbuf[..., :n_in] = x_arr
        plan = self._get_plan(ws, "irfft", inbuf, outbuf, threads, flags)
        plan()
        if out is None:
            return np.array(outbuf, copy=True)
        return _copy_to_out(out, outbuf)


if _HAVE_PYFFTW:
    DEFAULT_FFT_BACKEND = PyFFTWBackend()
else:
    DEFAULT_FFT_BACKEND = SciPyFFTBackend()
