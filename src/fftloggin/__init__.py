"""Differentiable FFTLog transforms built on JAX."""

import os as _os
from contextlib import nullcontext as _nullcontext

_typecheck_import_context = _nullcontext()
if _os.environ.get("FFTLOGGIN_RUNTIME_TYPECHECK") == "1":
    try:
        import beartype as _beartype  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "FFTLOGGIN_RUNTIME_TYPECHECK=1 requires beartype; install the "
            "fftloggin development dependencies with `uv sync --group dev`."
        ) from error

    from jaxtyping import install_import_hook as _install_import_hook

    _typecheck_import_context = _install_import_hook(
        ["fftloggin.fftlog", "fftloggin.grids", "fftloggin.kernels"],
        "beartype.beartype",
    )

with _typecheck_import_context:
    from . import kernels
    from .exceptions import ArgumentOutOfDomainError, DomainCheckWarning
    from .fftlog import forward, inverse, lowring_log_kr, validate_parameters
    from .grids import get_array_center, get_paired_grids, infer_dlog, infer_log_kr
    from .kernels import (
        BesselJKernel,
        Derivative,
        Kernel,
        ShiftedKernel,
        SphericalBesselJKernel,
    )

__all__ = (
    "ArgumentOutOfDomainError",
    "BesselJKernel",
    "Derivative",
    "DomainCheckWarning",
    "Kernel",
    "ShiftedKernel",
    "SphericalBesselJKernel",
    "forward",
    "get_array_center",
    "get_paired_grids",
    "infer_dlog",
    "infer_log_kr",
    "inverse",
    "kernels",
    "lowring_log_kr",
    "validate_parameters",
)
