"""Utility functions for fftloggin."""

from typing import Literal

import numpy as np
import numpy.typing as npt


def allocate_broadcasted_array(*arrs: npt.NDArray, dtype: npt.DTypeLike | None = None):
    if dtype is None:
        dtype = np.result_type(*arrs)
    else:
        dtype = np.dtype(dtype)
    out_shape = np.broadcast_shapes(*(arr.shape for arr in arrs))
    return np.empty(out_shape, dtype=dtype)


def in_place_compatible(
    target: npt.NDArray, *others: npt.NDArray, casting: str = "same_kind"
) -> bool:
    if not others:
        return True
    shape = np.broadcast_shapes(target.shape, *(arr.shape for arr in others))
    if shape != target.shape:
        return False
    result_dtype = np.result_type(target.dtype, *(arr.dtype for arr in others))
    return np.can_cast(result_dtype, target.dtype, casting=casting)


def append_dims(
    a: npt.ArrayLike, ndim: int, where: Literal["left", "right"] = "right"
) -> npt.NDArray:
    """
    Append singleton dimensions to an array.

    Parameters
    ----------
    a : array_like
        Input array.
    ndim : int
        Number of singleton dimensions to append.
    where : {"left", "right"}, default "right"
        Whether to append dimensions on the left or right.

    Returns
    -------
    ndarray
        Array with shape (*a.shape, 1, ..., 1) if where="right",
        or (1, ..., 1, *a.shape) if where="left".

    Examples
    --------
    >>> a = np.array([1, 2, 3])  # shape (3,)
    >>> append_dims(a, 2, where="right").shape
    (3, 1, 1)
    >>> append_dims(a, 2, where="left").shape
    (1, 1, 3)
    """
    a = np.asarray(a)
    if where == "right":
        return a.reshape(a.shape + (1,) * ndim)
    elif where == "left":
        return a.reshape((1,) * ndim + a.shape)
    else:
        raise ValueError(f"where must be 'left' or 'right', got {where}")


def count_trailing_ones(shape: tuple) -> int:
    """Count consecutive 1s at the beginning of a shape tuple."""
    count = 0
    for dim in shape:
        if dim == 1:
            count += 1
        else:
            break
    return count


def outer_broadcast(
    left: npt.ArrayLike, right: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Reshape left and right for outer-product style broadcasting.

    This function reshapes the left operand to have trailing singleton dimensions
    as needed to align with the right operand. NumPy's broadcasting rules will
    automatically pad the right operand on the left during operations.

    Parameters
    ----------
    left : array_like
        Left operand.
    right : array_like
        Right operand.

    Returns
    -------
    left_reshaped : ndarray
        Left with trailing singleton dimensions appended as needed.
    right_reshaped : ndarray
        Right operand (returned unchanged if compatible).

    Examples
    --------
    >>> left = np.array([1, 2, 3])  # shape (3,)
    >>> right = np.array([4, 5])    # shape (2,)
    >>> left_r, right_r = outer_broadcast(left, right)
    >>> left_r.shape
    (3, 1)
    >>> right_r.shape
    (2,)
    >>> (left_r + right_r).shape
    (3, 2)
    """
    left = np.asarray(left)
    right = np.asarray(right)

    # Count existing trailing ones in left
    n_trailing_ones_left = count_trailing_ones(left.shape[::-1])

    # Only append dimensions we actually need
    n_dims_to_add_left = max(0, right.ndim - n_trailing_ones_left)

    # Reshape left to add trailing singleton dimensions
    left = append_dims(left, n_dims_to_add_left, where="right")

    return left, right


def safe_broadcast(
    left: npt.ArrayLike, right: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Safely broadcast two arrays, using outer_broadcast if needed.

    This function checks if both operands are multidimensional and applies
    outer_broadcast if necessary. Otherwise, returns arrays as-is for
    standard NumPy broadcasting.

    Parameters
    ----------
    left : array_like
        Left operand (typically batch parameter).
    right : array_like
        Right operand (typically along-axis data).

    Returns
    -------
    left_broadcast : ndarray
        Left operand, possibly reshaped for broadcasting.
    right_broadcast : ndarray
        Right operand as ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> # Scalar case - no reshaping needed
    >>> left, right = safe_broadcast(1.0, np.array([1, 2, 3]))
    >>> left.shape, right.shape
    ((), (3,))
    >>>
    >>> # Batch case - applies outer_broadcast
    >>> left, right = safe_broadcast(np.array([1, 2]), np.array([3, 4, 5]))
    >>> left.shape, right.shape
    ((2, 1), (3,))
    >>> (left + right).shape
    (2, 3)
    """
    left = np.asarray(left)
    right = np.asarray(right)

    # Only use outer_broadcast for non-scalar batched operations
    if left.ndim > 0 and right.ndim > 0:
        return outer_broadcast(left, right)
    else:
        # Scalar case or one operand is scalar - standard broadcasting
        return left, right


def prepare_batch_params(
    *params: npt.ArrayLike,
) -> tuple[npt.NDArray, ...]:
    """
    Prepare parameters for batched FFTLog transforms.

    Ensures parameters have compatible shapes with a trailing singleton dimension
    for proper broadcasting in FFTLog operations. Converts arrays to shape
    (*batch_shape, 1) and validates shape compatibility. Scalars remain scalars.

    Parameters
    ----------
    *params : array_like
        Variable number of parameters. Can be scalars or arrays.

    Returns
    -------
    tuple of ndarray
        Prepared parameters with shape () for scalars or (*batch_shape, 1) for arrays.

    Raises
    ------
    ValueError
        If array parameters have incompatible broadcast shapes.

    Examples
    --------
    Convert 1D arrays to batched shape:

    >>> dlog, bias, kr = prepare_batch_params(0.05, 0.0, [0.5, 1.0, 2.0])
    >>> kr.shape
    (3, 1)
    >>> dlog.shape
    ()

    Incompatible shapes raise error:

    >>> prepare_batch_params([0.04, 0.05], 0.0, [0.5, 1.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: Batch parameters have incompatible shapes...

    Compatible batch shapes:

    >>> dlog, bias, kr = prepare_batch_params([0.04, 0.05], [0.0, 0.1], [1.0, 2.0])
    >>> dlog.shape, bias.shape, kr.shape
    ((2, 1), (2, 1), (2, 1))

    Works with any number of parameters:

    >>> a, b = prepare_batch_params([1, 2, 3], [4, 5, 6])
    >>> a.shape, b.shape
    ((3, 1), (3, 1))

    See Also
    --------
    FFTLog : Main FFTLog class that accepts batched parameters

    Notes
    -----
    This function follows the design principle that users should supply
    pre-shaped arrays for batching. FFTLog operations expect parameters
    with shape (*batch_shape, 1) to broadcast naturally with data arrays
    of shape (n,), producing results with shape (*batch_shape, n).

    Scalars are never broadcasted - they remain scalars and will broadcast
    naturally during FFTLog operations.
    """
    arrays = []

    for param in params:
        arr = np.asarray(param)

        # Add trailing singleton dimension if not present (but keep scalars as scalars)
        if arr.ndim > 0 and arr.shape[-1] != 1:
            arr = arr.reshape(arr.shape + (1,))

        arrays.append(arr)

    # Validate broadcast compatibility (only for non-scalar arrays)
    non_scalar_arrays = [arr for arr in arrays if arr.ndim > 0]

    if len(non_scalar_arrays) > 1:
        try:
            # Just validate batch shapes are compatible, don't actually broadcast
            # Remove trailing singleton for validation
            np.broadcast_shapes(*[arr.shape[:-1] for arr in non_scalar_arrays])
        except ValueError as e:
            shapes_str = ", ".join(f"{arr.shape}" for arr in arrays)
            raise ValueError(
                f"Batch parameters have incompatible shapes: {shapes_str}. "
                f"Parameters must have compatible shapes for broadcasting. "
                f"Original error: {e}"
            ) from e

    return tuple(arrays)
