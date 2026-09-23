"""Tests for the public Mellin kernel API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import digamma
from jaxtyping import TypeCheckError
from numpy.testing import assert_allclose
from scipy.special import digamma as scipy_digamma
from scipy.special import loggamma

from fftloggin import (
    BesselJKernel,
    Derivative,
    ShiftedKernel,
    SphericalBesselJKernel,
)

VALUE_RTOL = 1e-11
VALUE_ATOL = 1e-12
GRAD_RTOL = 1e-6
GRAD_ATOL = 1e-8
GRAD_STEP = 1e-5


def _bessel_reference(mu, s):
    s = np.asarray(s)
    return np.exp(
        (s - 1) * np.log(2) + loggamma((mu + s) / 2) - loggamma((mu + 2 - s) / 2)
    )


def _spherical_reference(ell, s):
    return np.sqrt(np.pi / 2) * _bessel_reference(ell + 0.5, np.asarray(s) - 0.5)


def _derivative_reference(mu, s, order):
    s = np.asarray(s)
    factor = np.prod([s - n for n in range(1, order + 1)], axis=0)
    return (-1) ** order * factor * _bessel_reference(mu, s - order)


def _centered_difference(func, x):
    return (np.real(func(x + GRAD_STEP)) - np.real(func(x - GRAD_STEP))) / (
        2 * GRAD_STEP
    )


@pytest.fixture(
    params=[
        BesselJKernel(0.5),
        SphericalBesselJKernel(1.0),
        ShiftedKernel(BesselJKernel(0.5), 0.2),
        Derivative(BesselJKernel(0.5), 1),
    ],
    ids=["bessel", "spherical", "shifted", "derivative"],
)
def kernel(request):
    return request.param


@pytest.mark.parametrize(
    "s",
    [0.8, 0.8 + 0.3j, np.array([0.6 + 0.2j, 0.8 + 0.3j, 1.1 + 0.1j])],
    ids=["real-scalar", "complex-scalar", "complex-array"],
)
@pytest.mark.parametrize(
    "kind,parameter",
    [("bessel", 0.5), ("spherical", 1.0)],
)
def test_kernel_values_match_scipy(kind, parameter, s):
    if kind == "bessel":
        got = BesselJKernel(parameter)(s)
        expected = _bessel_reference(parameter, s)
    else:
        got = SphericalBesselJKernel(parameter)(s)
        expected = _spherical_reference(parameter, s)

    assert got.shape == np.asarray(s).shape
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


@pytest.mark.parametrize("order", [1, 2])
def test_derivative_matches_mellin_identity(order):
    mu = 0.5
    s = np.array([order + 0.3 + 0.2j, order + 0.7 + 0.3j])
    got = Derivative(BesselJKernel(mu), order)(s)
    expected = _derivative_reference(mu, s, order)
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_shifted_kernel_matches_shifted_argument():
    base = BesselJKernel(0.5)
    s = np.array([0.6 + 0.2j, 0.8 + 0.3j])

    assert_allclose(ShiftedKernel(base, 0.2)(s), base(s + 0.2), rtol=VALUE_RTOL)


@pytest.mark.parametrize("order", [0, -1, 1.5])
def test_derivative_constructor_rejects_invalid_order(order):
    with pytest.raises((ValueError, TypeCheckError), match="positive integer|order"):
        Derivative(BesselJKernel(0.5), order)


@pytest.mark.parametrize(
    "kernel,lower,upper",
    [
        (BesselJKernel(0.5), -0.5, 1.5),
        (SphericalBesselJKernel(1.0), -1.0, 2.0),
        (ShiftedKernel(BesselJKernel(0.5), 0.25), -0.75, 1.25),
        (Derivative(BesselJKernel(0.5), 2), 1.5, 3.5),
    ],
    ids=["bessel", "spherical", "shifted", "derivative"],
)
def test_domain_has_open_bounds_and_uses_real_part(kernel, lower, upper):
    assert_allclose(kernel.domain, (lower, upper), rtol=VALUE_RTOL)
    middle = (lower + upper) / 2
    assert bool(kernel.is_in_domain(middle))
    assert bool(kernel.is_in_domain(middle + 3j))
    assert not bool(kernel.is_in_domain(lower + 3j))
    assert not bool(kernel.is_in_domain(upper - 3j))


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("mu", [0.1, 0.5, 1.0]),
        ("ell", [0.0, 1.0, 2.0]),
        ("nu", [0.1, 0.2, 0.3]),
    ],
)
def test_vmap_over_scalar_kernel_parameters(kind, parameters):
    s = jnp.array([0.8 + 0.2j, 1.0 + 0.3j])

    def evaluate(parameter):
        if kind == "mu":
            return BesselJKernel(parameter)(s)
        if kind == "ell":
            return SphericalBesselJKernel(parameter)(s)
        return ShiftedKernel(BesselJKernel(0.5), parameter)(s)

    got = jax.vmap(evaluate)(jnp.asarray(parameters))
    expected = jnp.stack([evaluate(parameter) for parameter in parameters])
    assert got.shape == (len(parameters), len(s))
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_vmap_over_samples(kernel):
    samples = jnp.array([0.8 + 0.2j, 1.0 + 0.3j])
    got = jax.vmap(kernel)(samples)
    expected = jnp.stack([kernel(sample) for sample in samples])
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert_allclose(got, kernel(samples), rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_nested_vmap_over_parameters_and_samples():
    mus = jnp.array([0.1, 0.5, 1.0])
    samples = jnp.array([0.8 + 0.2j, 1.0 + 0.3j])
    got = jax.vmap(lambda mu: jax.vmap(lambda s: BesselJKernel(mu)(s))(samples))(mus)
    expected = jnp.stack([BesselJKernel(mu)(samples) for mu in mus])
    assert got.shape == (len(mus), len(samples))
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_kernel_is_a_jittable_pytree_argument(kernel):
    samples = jnp.array([0.8 + 0.2j, 1.0 + 0.3j])
    got = jax.jit(lambda k, s: k(s))(kernel, samples)
    assert_allclose(got, kernel(samples), rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_bessel_grad_with_respect_to_argument_matches_scipy():
    mu = 0.5
    s = 0.8
    a = (mu + s) / 2
    b = (mu + 2 - s) / 2
    expected = _bessel_reference(mu, s) * (
        np.log(2) + (scipy_digamma(a) + scipy_digamma(b)) / 2
    )
    got = jax.grad(lambda x: jnp.real(BesselJKernel(mu)(x)))(s)
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_bessel_complex_parameter_grad_matches_scipy():
    mu = 0.5
    s = 0.8 + 0.3j
    a = (mu + s) / 2
    b = (mu + 2 - s) / 2
    expected = np.real(
        _bessel_reference(mu, s) * (scipy_digamma(a) - scipy_digamma(b)) / 2
    )
    got = jax.grad(lambda parameter: jnp.real(BesselJKernel(parameter)(s)))(mu)
    assert_allclose(got, expected, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_spherical_parameter_grad_matches_scipy_difference():
    ell = 1.0
    s = 0.8 + 0.3j
    got = jax.grad(lambda parameter: jnp.real(SphericalBesselJKernel(parameter)(s)))(
        ell
    )
    expected = _centered_difference(
        lambda parameter: _spherical_reference(parameter, s), ell
    )
    assert_allclose(got, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_shift_parameter_grad_matches_scipy_difference():
    nu = 0.2
    s = 0.8 + 0.3j
    got = jax.grad(
        lambda parameter: jnp.real(ShiftedKernel(BesselJKernel(0.5), parameter)(s))
    )(nu)
    expected = _centered_difference(
        lambda parameter: _bessel_reference(0.5, s + parameter), nu
    )
    assert_allclose(got, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_derivative_base_parameter_grad_matches_scipy_difference():
    mu = 0.5
    s = 2.3 + 0.3j
    got = jax.grad(
        lambda parameter: jnp.real(Derivative(BesselJKernel(parameter), 2)(s))
    )(mu)
    expected = _centered_difference(
        lambda parameter: _derivative_reference(parameter, s, 2), mu
    )
    assert_allclose(got, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)


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
