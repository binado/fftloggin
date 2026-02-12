"""Tests for JAX JIT compatibility of FFTLog transforms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin.fftlog import FFTLog
from fftloggin.kernels import BesselJKernel

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


@pytest.fixture
def fftlog_and_input():
    """Create a simple FFTLog instance and test input."""
    r = np.logspace(-2, 2, 64)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0), kr=1.0)
    a = jnp.asarray(np.exp(-(r**2) / 2))
    return fftlog, a


def test_forward_is_jittable(fftlog_and_input):
    fftlog, a = fftlog_and_input
    eager = fftlog.forward(a)
    jitted = jax.jit(fftlog.forward)(a)
    assert_allclose(jitted, eager, rtol=1e-12)


def test_inverse_is_jittable(fftlog_and_input):
    fftlog, a = fftlog_and_input
    ak = fftlog.forward(a)
    eager = fftlog.inverse(ak)
    jitted = jax.jit(fftlog.inverse)(ak)
    assert_allclose(jitted, eager, atol=1e-15)


def test_roundtrip_under_jit(fftlog_and_input):
    fftlog, a = fftlog_and_input
    roundtrip = jax.jit(fftlog.inverse)(jax.jit(fftlog.forward)(a))
    assert_allclose(roundtrip, a, atol=1e-15)


def test_batched_forward_is_jittable():
    r = np.logspace(-2, 2, 64)
    kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(0), kr=kr)
    a = jnp.asarray(np.exp(-(r**2) / 2))

    eager = fftlog.forward(a)
    jitted = jax.jit(fftlog.forward)(a)
    assert_allclose(jitted, eager, rtol=1e-12)
    assert jitted.shape == (3, 64)
