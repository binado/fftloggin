"""Tests for the public Mellin kernel API."""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import digamma

from fftloggin import BesselJKernel


@pytest.mark.parametrize("mu, s", [(0.0, 0.2), (1.0, 0.75), (10.0, 1.25)])
def test_bessel_kernel_mu_gradient_matches_jax_digamma(mu, s):
    mu = jnp.asarray(mu)
    s = jnp.asarray(s)
    first = digamma((mu + s) / 2)
    second = digamma((mu + 2 - s) / 2)
    actual = jax.jit(jax.grad(lambda value: BesselJKernel(value)(s)))(mu)
    expected = BesselJKernel(mu)(s) * (first - second) / 2

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mu, s", [(0.0, 0.2), (1.0, 0.75), (10.0, 1.25)])
def test_bessel_kernel_s_gradient_matches_jax_digamma(mu, s):
    mu = jnp.asarray(mu)
    s = jnp.asarray(s)
    first = digamma((mu + s) / 2)
    second = digamma((mu + 2 - s) / 2)
    actual = jax.jit(jax.grad(lambda value: BesselJKernel(mu)(value)))(s)
    expected = BesselJKernel(mu)(s) * (jnp.log(2) + (first + second) / 2)

    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6)
