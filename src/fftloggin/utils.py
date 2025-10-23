"""Utility functions for fftloggin."""

from typing import Literal

import numpy as np
import numpy.typing as npt


def append_dims(
    a: npt.ArrayLike, ndim: int, where: Literal["left", "right"] = "right"
) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray]:
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
