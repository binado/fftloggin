"""Tests for Array API compatibility using array-api-strict."""

import numpy as np
import pytest

from fftloggin import FFTLog
from fftloggin.kernels import BesselJKernel

array_api_strict = pytest.importorskip("array_api_strict")


@pytest.fixture
def fftlog():
    return FFTLog(kernel=BesselJKernel(0), n=128, dlog=0.05)


@pytest.fixture
def np_input():
    return np.ones(128)


def to_strict(a):
    """Convert numpy array to array-api-strict array."""
    return array_api_strict.asarray(a)


class TestForwardArrayAPI:
    def test_returns_strict_array(self, fftlog, np_input):
        a = to_strict(np_input)
        result = fftlog.forward(a)
        assert not isinstance(result, np.ndarray)

    def test_matches_numpy_result(self, fftlog, np_input):
        np_result = fftlog.forward(np_input)
        strict_result = fftlog.forward(to_strict(np_input))
        np.testing.assert_allclose(np.asarray(strict_result), np_result, rtol=1e-10)

    def test_output_shape(self, fftlog, np_input):
        result = fftlog.forward(to_strict(np_input))
        assert result.shape == (128,)

    def test_size_mismatch_raises(self, fftlog):
        a = to_strict(np.ones(64))
        with pytest.raises(ValueError, match="does not match"):
            fftlog.forward(a)


class TestInverseArrayAPI:
    def test_returns_strict_array(self, fftlog, np_input):
        ak = to_strict(np_input)
        result = fftlog.inverse(ak)
        assert not isinstance(result, np.ndarray)

    def test_matches_numpy_result(self, fftlog, np_input):
        np_result = fftlog.inverse(np_input)
        strict_result = fftlog.inverse(to_strict(np_input))
        np.testing.assert_allclose(np.asarray(strict_result), np_result, rtol=1e-10)


class TestRoundtripArrayAPI:
    def test_forward_inverse_roundtrip(self, fftlog):
        a_np = np.exp(-(np.linspace(-3, 3, 128) ** 2))
        a = to_strict(a_np)
        ak = fftlog.forward(a)
        a_recovered = fftlog.inverse(ak)
        np.testing.assert_allclose(np.asarray(a_recovered), a_np, rtol=1e-5)


class TestBatchedArrayAPI:
    def test_batched_forward(self):
        from fftloggin import prepare_batch_params

        dlog, bias, kr = prepare_batch_params(0.05, 0.0, [0.5, 1.0, 2.0])
        fftlog = FFTLog(kernel=BesselJKernel(0), n=128, dlog=dlog, bias=bias, kr=kr)

        a_np = np.ones(128)
        np_result = fftlog.forward(a_np)

        a_strict = to_strict(a_np)
        strict_result = fftlog.forward(a_strict)
        assert strict_result.shape == (3, 128)
        np.testing.assert_allclose(np.asarray(strict_result), np_result, rtol=1e-10)
