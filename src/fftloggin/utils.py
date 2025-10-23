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


def outer_broadcast(
    left: npt.ArrayLike, right: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape left and right for outer-product style broadcasting.

    This function reshapes the inputs so that left has trailing singleton
    dimensions and right has leading singleton dimensions, enabling NumPy's
    broadcasting rules to produce an outer product-like result.

    Parameters
    ----------
    left : array_like
        Left operand.
    right : array_like
        Right operand.

    Returns
    -------
    left_reshaped : ndarray
        Left with shape (*left.shape, 1, ..., 1) where the number of trailing
        1s equals right.ndim.
    right_reshaped : ndarray
        Right with shape (1, ..., 1, *right.shape) where the number of leading
        1s equals left.ndim.

    Examples
    --------
    >>> left = np.array([1, 2, 3])  # shape (3,)
    >>> right = np.array([4, 5])    # shape (2,)
    >>> left_r, right_r = outer_broadcast(left, right)
    >>> left_r.shape
    (3, 1)
    >>> right_r.shape
    (1, 2)
    >>> (left_r + right_r).shape
    (3, 2)
    """
    left = np.asarray(left)
    right = np.asarray(right)

    # Save original ndim values before reshaping
    left_ndim = left.ndim
    right_ndim = right.ndim

    # Reshape left to add trailing singleton dimensions
    left = append_dims(left, right_ndim, where="right")

    # Reshape right to add leading singleton dimensions
    right = append_dims(right, left_ndim, where="left")

    return left, right
