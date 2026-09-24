"""Numerical and JAX transformation tests for the public FFTLog API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import poch

from fftloggin import (
    BesselJKernel,
    Derivative,
    ShiftedKernel,
    forward,
    get_paired_grids,
    inverse,
    lowring_log_kr,
    plan,
)


@pytest.fixture
def scipy_input():
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    a = r ** (mu + 1) * np.exp(-(r**2) / 2)
    return a, BesselJKernel(mu), np.log(r[1] / r[0])


# Reference values from SciPy's scipy/special/tests/test_fftlog.py. They also
# cover fractional order and biases absent from the optional Fortran matrix.
SCIPY_CASES = (
    (
        "ordinary",
        0.0,
        False,
        (
            -0.1159922613593045e-02,
            +0.1625822618458832e-02,
            -0.1949518286432330e-02,
            +0.3789220182554077e-02,
            +0.5093959119952945e-03,
            +0.2785387803618774e-01,
            +0.9944952700848897e-01,
            +0.4599202164586588e00,
            +0.3157462160881342e00,
            -0.8201236844404755e-03,
            -0.7834031308271878e-03,
            +0.3931444945110708e-03,
            -0.2697710625194777e-03,
            +0.3568398050238820e-03,
            -0.5554454827797206e-03,
            +0.8286331026468585e-03,
        ),
    ),
    (
        "lowring",
        0.0,
        True,
        (
            +0.4353768523152057e-04,
            -0.9197045663594285e-05,
            +0.3150140927838524e-03,
            +0.9149121960963704e-03,
            +0.5808089753959363e-02,
            +0.2548065256377240e-01,
            +0.1339477692089897e00,
            +0.4821530509479356e00,
            +0.2659899781579785e00,
            -0.1116475278448113e-01,
            +0.1791441617592385e-02,
            -0.4181810476548056e-03,
            +0.1314963536765343e-03,
            -0.5422057743066297e-04,
            +0.3208681804170443e-04,
            -0.2696849476008234e-04,
        ),
    ),
    (
        "positive_bias",
        0.8,
        True,
        (
            -7.3436673558316850e00,
            +0.1710271207817100e00,
            +0.1065374386206564e00,
            -0.5121739602708132e-01,
            +0.2636649319269470e-01,
            +0.1697209218849693e-01,
            +0.1250215614723183e00,
            +0.4739583261486729e00,
            +0.2841149874912028e00,
            -0.8312764741645729e-02,
            +0.1024233505508988e-02,
            -0.1644902767389120e-03,
            +0.3305775476926270e-04,
            -0.7786993194882709e-05,
            +0.1962258449520547e-05,
            -0.8977895734909250e-06,
        ),
    ),
    (
        "negative_bias",
        -0.8,
        True,
        (
            +0.8985777068568745e-05,
            +0.4074898209936099e-04,
            +0.2123969254700955e-03,
            +0.1009558244834628e-02,
            +0.5131386375222176e-02,
            +0.2461678673516286e-01,
            +0.1235812845384476e00,
            +0.4719570096404403e00,
            +0.2893487490631317e00,
            -0.1686570611318716e-01,
            +0.2231398155172505e-01,
            -0.1480742256379873e-01,
            +0.1692387813500801e00,
            +0.3097490354365797e00,
            +2.7593607182401860e00,
            +10.5251075070045800e00,
        ),
    ),
)


@pytest.mark.parametrize(
    "name,bias,lowring,expected", SCIPY_CASES, ids=[case[0] for case in SCIPY_CASES]
)
def test_scipy_reference(x64, scipy_input, name, bias, lowring, expected):
    a, kernel, dlog = scipy_input
    log_kr = lowring_log_kr(kernel, dlog=dlog, bias=bias) if lowring else 0.0
    result = forward(a, kernel, dlog=dlog, bias=bias, log_kr=log_kr)
    assert_allclose(result, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("n", [64, 63])
def test_power_law_matches_analytic_transform(x64, n):
    rng = np.random.RandomState(3491349965)
    mu = rng.uniform(0, 3)
    gamma = rng.uniform(-1 - mu, 0.5)
    r = np.logspace(-2, 2, n)
    dlog = np.log(r[1] / r[0])
    kernel = BesselJKernel(mu)
    log_kr = lowring_log_kr(kernel, dlog=dlog, bias=gamma)
    _, k = get_paired_grids(r=r, log_kr=log_kr)

    result = forward(r**gamma, kernel, dlog=dlog, bias=gamma, log_kr=log_kr)
    expected = (2 / np.asarray(k)) ** gamma * poch((mu + 1 - gamma) / 2, gamma)
    assert_allclose(result, expected, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("n", [64, 63])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("bias", [0.1, -0.1])
@pytest.mark.parametrize("kr", [1.0, np.exp(1.0), np.exp(-1.0)])
def test_forward_inverse_round_trip(x64, n, order, bias, kr):
    rng = np.random.RandomState(3491349965)
    a = rng.standard_normal(n)
    mu = rng.uniform(3, 5)
    kernel = Derivative(BesselJKernel(mu), order) if order else BesselJKernel(mu)
    transformed = forward(a, kernel, dlog=0.1, bias=bias, log_kr=np.log(kr))
    restored = inverse(transformed, kernel, dlog=0.1, bias=bias, log_kr=np.log(kr))
    assert_allclose(restored, a, rtol=1.5e-7, atol=1e-12)


@pytest.mark.parametrize("n", [63, 64])
def test_jit_forward_and_inverse(n):
    a = jnp.linspace(0.1, 1.0, n)
    kernel = BesselJKernel(jnp.asarray(0.3))
    params = {"dlog": 0.1, "bias": 0.1, "log_kr": 0.2}
    eager = forward(a, kernel, **params)
    compiled = jax.jit(forward)(a, kernel, **params)
    assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)
    assert_allclose(
        jax.jit(inverse)(compiled, kernel, **params),
        inverse(eager, kernel, **params),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("parameter", ["mu", "dlog", "bias", "log_kr"])
def test_vmap_matches_individual_transforms(parameter):
    n = 32
    r = jnp.exp(0.1 * (jnp.arange(n) - (n - 1) / 2))
    a = r**1.3 * jnp.exp(-(r**2) / 2)
    values = jnp.asarray([0.2, 0.3, 0.4])

    def transform(value):
        params = {"dlog": 0.1, "bias": 0.1, "log_kr": 0.2}
        kernel = BesselJKernel(value if parameter == "mu" else 0.3)
        if parameter != "mu":
            params[parameter] = value
        return forward(a, kernel, **params)

    expected = jnp.stack([transform(value) for value in values])
    actual = jax.jit(jax.vmap(transform))(values)
    assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("parameter", ["mu", "dlog", "bias", "log_kr"])
def test_scalar_gradient_matches_finite_difference(x64, parameter):
    n = 32
    r = jnp.exp(0.1 * (jnp.arange(n) - (n - 1) / 2))
    a = r**1.3 * jnp.exp(-(r**2) / 2)
    weights = jnp.linspace(0.4, 1.2, n)
    values = {"mu": 0.3, "dlog": 0.1, "bias": 0.1, "log_kr": 0.2}

    def loss(value):
        params = values | {parameter: value}
        return jnp.sum(
            forward(
                a,
                BesselJKernel(params["mu"]),
                dlog=params["dlog"],
                bias=params["bias"],
                log_kr=params["log_kr"],
            )
            * weights
        )

    point = jnp.asarray(values[parameter])
    step = 1e-5
    numerical = (loss(point + step) - loss(point - step)) / (2 * step)
    automatic = jax.jit(jax.grad(loss))(point)
    assert_allclose(automatic, numerical, rtol=2e-4, atol=2e-6)


def test_input_gradient_matches_finite_difference(x64):
    a = jnp.linspace(0.2, 1.1, 32)
    weights = jnp.linspace(0.4, 1.2, 32)

    def loss(samples):
        return jnp.sum(
            forward(samples, BesselJKernel(0.3), dlog=0.1, bias=0.1) * weights
        )

    index = 9
    step = 1e-5
    numerical = (loss(a.at[index].add(step)) - loss(a.at[index].add(-step))) / (
        2 * step
    )
    automatic = jax.jit(jax.grad(loss))(a)
    assert_allclose(automatic[index], numerical, rtol=2e-4, atol=2e-6)


def test_shifted_kernel_matches_weighted_input(x64):
    r = np.logspace(-3, 3, 128)
    a = np.exp(-(r**2))
    mu, nu, bias = 0.8, 0.2, -0.4
    dlog = np.log(r[1] / r[0])
    base = BesselJKernel(mu)
    _, k = get_paired_grids(r=r)

    shifted = forward(a, ShiftedKernel(base, nu), dlog=dlog, bias=bias)
    direct = forward(a * r**nu, base, dlog=dlog, bias=bias + nu)
    assert_allclose(shifted * k**-nu, direct, rtol=1e-7, atol=1e-12)


@pytest.fixture
def smooth_input():
    n = 32
    r = jnp.exp(0.1 * (jnp.arange(n) - (n - 1) / 2))
    return r**1.3 * jnp.exp(-(r**2) / 2)


@pytest.mark.parametrize("n", [64, 63])
@pytest.mark.parametrize(("bias", "log_kr"), [(0.0, 0.0), (0.1, 0.2), (-0.1, -0.5)])
def test_plan_matches_kernel_transform(x64, n, bias, log_kr):
    rng = np.random.RandomState(20260924)
    a = rng.standard_normal(n)
    kernel = BesselJKernel(0.3)
    params = {"dlog": 0.1, "bias": bias, "log_kr": log_kr}
    p = plan(kernel, n, **params)
    assert_allclose(forward(a, p), forward(a, kernel, **params), rtol=1e-14)
    assert_allclose(inverse(a, p), inverse(a, kernel, **params), rtol=1e-14)


def test_vmap_over_inputs_with_plan(smooth_input):
    batch = jnp.stack([smooth_input * scale for scale in (0.5, 1.0, 2.0)])
    p = plan(BesselJKernel(0.3), batch.shape[1], dlog=0.1, bias=0.1)
    expected = jnp.stack([forward(a, p) for a in batch])
    actual = jax.jit(jax.vmap(forward, in_axes=(0, None)))(batch, p)
    assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_input_gradient_with_precomputed_plan(x64, smooth_input):
    weights = jnp.linspace(0.4, 1.2, smooth_input.shape[0])
    p = plan(BesselJKernel(0.3), smooth_input.shape[0], dlog=0.1, bias=0.1)

    def loss(samples, p):
        return jnp.sum(forward(samples, p) * weights)

    kernel_loss = jnp.sum(
        forward(smooth_input, BesselJKernel(0.3), dlog=0.1, bias=0.1) * weights
    )
    assert_allclose(jax.jit(loss)(smooth_input, p), kernel_loss, rtol=1e-14)
    index, step = 9, 1e-5
    numerical = (
        loss(smooth_input.at[index].add(step), p)
        - loss(smooth_input.at[index].add(-step), p)
    ) / (2 * step)
    automatic = jax.jit(jax.grad(loss))(smooth_input, p)
    assert_allclose(automatic[index], numerical, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("parameter", ["mu", "bias"])
def test_gradient_through_plan_matches_kernel_path(x64, smooth_input, parameter):
    n = smooth_input.shape[0]
    values = {"mu": 0.3, "bias": 0.1}

    def loss(value, use_plan):
        params = values | {parameter: value}
        kernel = BesselJKernel(params["mu"])
        if use_plan:
            return jnp.sum(
                forward(smooth_input, plan(kernel, n, dlog=0.1, bias=params["bias"]))
            )
        return jnp.sum(forward(smooth_input, kernel, dlog=0.1, bias=params["bias"]))

    point = jnp.asarray(values[parameter])
    assert_allclose(
        jax.grad(loss)(point, True), jax.grad(loss)(point, False), rtol=1e-12
    )


@pytest.mark.parametrize("keyword", ["dlog", "bias", "log_kr"])
@pytest.mark.parametrize("transform", [forward, inverse])
def test_plan_rejects_grid_keywords(smooth_input, transform, keyword):
    p = plan(BesselJKernel(0.3), smooth_input.shape[0], dlog=0.1)
    with pytest.raises(TypeError, match="Plan"):
        transform(smooth_input, p, **{keyword: 0.1})


@pytest.mark.parametrize("transform", [forward, inverse])
def test_plan_rejects_other_sample_count(smooth_input, transform):
    p = plan(BesselJKernel(0.3), smooth_input.shape[0] + 1, dlog=0.1)
    with pytest.raises(ValueError, match="n="):
        transform(smooth_input, p)


@pytest.mark.parametrize("transform", [forward, inverse])
def test_kernel_requires_dlog(smooth_input, transform):
    with pytest.raises(TypeError, match="dlog"):
        transform(smooth_input, BesselJKernel(0.3))


def test_plan_rejects_too_few_samples():
    with pytest.raises(ValueError):
        plan(BesselJKernel(0.3), 1, dlog=0.1)
