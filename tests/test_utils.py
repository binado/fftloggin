"""Tests for utility functions in fftloggin.utils."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fftloggin.utils import append_dims, outer_broadcast


class TestAppendDims:
    """Tests for append_dims function."""

    def test_append_dims_right_scalar(self):
        """Test append_dims with scalar and where='right'."""
        a = np.array(5.0)  # shape ()
        result = append_dims(a, 2, where="right")
        assert result.shape == (1, 1)

    def test_append_dims_right_1d(self):
        """Test append_dims with 1D array and where='right'."""
        a = np.array([1, 2, 3])  # shape (3,)
        result = append_dims(a, 2, where="right")
        assert result.shape == (3, 1, 1)
        assert_array_equal(result[:, 0, 0], a)

    def test_append_dims_right_2d(self):
        """Test append_dims with 2D array and where='right'."""
        a = np.arange(6).reshape(2, 3)  # shape (2, 3)
        result = append_dims(a, 3, where="right")
        assert result.shape == (2, 3, 1, 1, 1)
        assert_array_equal(result[:, :, 0, 0, 0], a)

    def test_append_dims_left_scalar(self):
        """Test append_dims with scalar and where='left'."""
        a = np.array(5.0)  # shape ()
        result = append_dims(a, 2, where="left")
        assert result.shape == (1, 1)

    def test_append_dims_left_1d(self):
        """Test append_dims with 1D array and where='left'."""
        a = np.array([1, 2, 3])  # shape (3,)
        result = append_dims(a, 2, where="left")
        assert result.shape == (1, 1, 3)
        assert_array_equal(result[0, 0, :], a)

    def test_append_dims_left_2d(self):
        """Test append_dims with 2D array and where='left'."""
        a = np.arange(6).reshape(2, 3)  # shape (2, 3)
        result = append_dims(a, 3, where="left")
        assert result.shape == (1, 1, 1, 2, 3)
        assert_array_equal(result[0, 0, 0, :, :], a)

    def test_append_dims_zero_ndim(self):
        """Test append_dims with zero dimensions to append."""
        a = np.array([1, 2, 3])
        result_left = append_dims(a, 0, where="left")
        result_right = append_dims(a, 0, where="right")
        assert result_left.shape == (3,)
        assert result_right.shape == (3,)
        assert_array_equal(result_left, a)
        assert_array_equal(result_right, a)

    def test_append_dims_invalid_where(self):
        """Test append_dims with invalid where parameter."""
        a = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="where must be 'left' or 'right'"):
            append_dims(a, 2, where="invalid")

    def test_append_dims_array_like_input(self):
        """Test append_dims with array-like inputs (list, tuple)."""
        # Test with list
        result_list = append_dims([1, 2, 3], 2, where="right")
        assert result_list.shape == (3, 1, 1)

        # Test with tuple
        result_tuple = append_dims((1, 2, 3), 2, where="right")
        assert result_tuple.shape == (3, 1, 1)

    def test_append_dims_preserves_values(self):
        """Test that append_dims preserves values."""
        a = np.array([[1, 2], [3, 4]])
        result = append_dims(a, 2, where="right")
        # Verify values are preserved (just reshaped)
        assert_array_equal(result[:, :, 0, 0], a)


class TestOuterBroadcast:
    """Tests for outer_broadcast function."""

    def test_outer_broadcast_1d_1d(self):
        """Test outer_broadcast with two 1D arrays."""
        left = np.array([1, 2, 3])
        right = np.array([4, 5])
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (3, 1)
        assert right_r.shape == (2,)
        # Broadcasting should work (NumPy pads right on the left)
        result = left_r + right_r
        assert result.shape == (3, 2)

    def test_outer_broadcast_scalar_1d(self):
        """Test outer_broadcast with scalar and 1D array.

        Scalar (ndim=0) gets 1 trailing dim -> (1,)
        1D array (ndim=1) stays unchanged -> (3,)
        """
        left = 5.0
        right = np.array([1, 2, 3])
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (1,)
        assert right_r.shape == (3,)
        result = left_r + right_r
        assert result.shape == (3,)

    def test_outer_broadcast_1d_scalar(self):
        """Test outer_broadcast with 1D array and scalar.

        1D array (ndim=1) stays unchanged -> (3,)
        Scalar (ndim=0) stays unchanged -> ()
        """
        left = np.array([1, 2, 3])
        right = 5.0
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (3,)
        assert right_r.shape == ()
        result = left_r + right_r
        assert result.shape == (3,)

    def test_outer_broadcast_scalar_scalar(self):
        """Test outer_broadcast with two scalars."""
        left = 3.0
        right = 5.0
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == ()
        assert right_r.shape == ()
        result = left_r + right_r
        assert result.shape == ()

    def test_outer_broadcast_2d_1d(self):
        """Test outer_broadcast with 2D and 1D arrays.

        2D array gets 1 trailing dim -> (2, 3, 1)
        1D array stays unchanged -> (2,)
        """
        left = np.arange(6).reshape(2, 3)
        right = np.array([1, 2])
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (2, 3, 1)
        assert right_r.shape == (2,)
        result = left_r + right_r
        assert result.shape == (2, 3, 2)

    def test_outer_broadcast_values_preserved(self):
        """Test that outer_broadcast preserves values."""
        left = np.array([1, 2, 3])
        right = np.array([10, 20])
        left_r, right_r = outer_broadcast(left, right)

        # Check values are preserved (just reshaped)
        assert_array_equal(left_r[:, 0], left)
        assert_array_equal(right_r, right)

    def test_outer_broadcast_broadcast_result(self):
        """Test that outer_broadcast result broadcasts correctly."""
        left = np.array([1, 2, 3])
        right = np.array([10, 20])
        left_r, right_r = outer_broadcast(left, right)

        result = left_r + right_r
        expected = np.array(
            [
                [11, 21],
                [12, 22],
                [13, 23],
            ]
        )
        assert_array_equal(result, expected)

    def test_outer_broadcast_multiplication(self):
        """Test outer_broadcast with multiplication."""
        left = np.array([1, 2, 3])
        right = np.array([2, 3])
        left_r, right_r = outer_broadcast(left, right)

        result = left_r * right_r
        expected = np.array(
            [
                [2, 3],
                [4, 6],
                [6, 9],
            ]
        )
        assert_array_equal(result, expected)

    def test_outer_broadcast_with_floats(self):
        """Test outer_broadcast with floating point arrays."""
        left = np.array([0.1, 0.2, 0.3])
        right = np.array([1.5, 2.5])
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (3, 1)
        assert right_r.shape == (2,)
        result = left_r + right_r
        expected = np.array(
            [
                [1.6, 2.6],
                [1.7, 2.7],
                [1.8, 2.8],
            ]
        )
        assert_allclose(result, expected)

    def test_outer_broadcast_array_like_inputs(self):
        """Test outer_broadcast with array-like inputs."""
        left = [1, 2, 3]
        right = [4, 5]
        left_r, right_r = outer_broadcast(left, right)

        assert left_r.shape == (3, 1)
        assert right_r.shape == (2,)
        result = left_r + right_r
        assert result.shape == (3, 2)
