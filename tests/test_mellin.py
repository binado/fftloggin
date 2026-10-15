"""Public Mellin conventions, numerical references and JAX compatibility."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from fftloggin import mellin


@pytest.fixture
def samples():
    rng = np.random.default_rng(456)
    return rng.normal(size=(2, 7, 3)), rng.normal(size=(2, 7, 3))


def _scale(n, norm, inverse=False):
    if norm == "ortho":
        return 1 / np.sqrt(n)
    if norm == "forward":
        return 1 if inverse else 1 / n
    return 1 / n if inverse else 1


@pytest.mark.parametrize("n", [5, 7, 8])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("real", [False, True])
def test_analysis_matches_finite_sum(x64, samples, n, norm, axis, real):
    a, b = samples
    a = a if real else a + 1j * b
    moved = np.moveaxis(a, axis, -1)
    resized = np.zeros((*moved.shape[:-1], n), dtype=moved.dtype)
    count = min(n, moved.shape[-1])
    resized[..., :count] = moved[..., :count]
    j = np.arange(n)
    m = np.arange(n // 2 + 1 if real else n)
    modes = np.exp(2j * np.pi * j[:, None] * m / n)
    expected = (resized * np.exp(0.3 * 0.2 * j)) @ modes * _scale(n, norm)
    transform = mellin.rmellin if real else mellin.mellin
    actual = transform(a, n=n, axis=axis, norm=norm, c=0.3, dlog=0.2)
    assert_allclose(actual, np.moveaxis(expected, -1, axis), atol=2e-13)


@pytest.mark.parametrize("n", [5, 7, 8])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_complex_synthesis_matches_finite_sum(x64, samples, n, norm, axis):
    a, b = samples
    a = a + 1j * b
    moved = np.moveaxis(a, axis, -1)
    resized = np.zeros((*moved.shape[:-1], n), dtype=complex)
    count = min(n, moved.shape[-1])
    resized[..., :count] = moved[..., :count]
    j = np.arange(n)
    modes = np.exp(-2j * np.pi * j[:, None] * j / n)
    expected = resized @ modes * _scale(n, norm, inverse=True)
    expected *= np.exp(-0.3 * 0.2 * j)
    actual = mellin.imellin(a, n=n, axis=axis, norm=norm, c=0.3, dlog=0.2)
    assert_allclose(actual, np.moveaxis(expected, -1, axis), atol=2e-13)


@pytest.mark.parametrize("n", [5, 7, 8])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_real_synthesis_resizing(x64, samples, n, norm, axis):
    a, b = samples
    spectrum = a + 1j * b
    expected = np.fft.irfft(np.conj(spectrum), n=n, axis=axis, norm=norm)
    weight_shape = [1] * expected.ndim
    weight_shape[axis] = n
    expected *= np.exp(-0.3 * 0.2 * np.arange(n)).reshape(weight_shape)
    actual = mellin.irmellin(spectrum, n=n, axis=axis, norm=norm, c=0.3, dlog=0.2)
    assert_allclose(actual, expected, atol=1e-13)


@pytest.mark.parametrize("n", [1, 7, 8])
@pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
@pytest.mark.parametrize("c", [-0.4, 0.0, 0.3])
def test_real_spectrum_and_round_trip(x64, n, norm, c):
    a = np.linspace(-0.3, 1.2, n)
    full = mellin.mellin(a, c=c, dlog=0.2, norm=norm)
    half = mellin.rmellin(a, c=c, dlog=0.2, norm=norm)
    assert_allclose(half, full[: n // 2 + 1], atol=1e-13)
    assert_allclose(full[(-np.arange(n)) % n], np.conj(full), atol=1e-13)
    assert_allclose(mellin.imellin(full, c=c, dlog=0.2, norm=norm), a, atol=1e-13)
    assert_allclose(mellin.irmellin(half, n=n, c=c, dlog=0.2, norm=norm), a, atol=1e-13)
    assert_allclose(half[0].imag, 0, atol=1e-13)
    if n % 2 == 0:
        assert_allclose(half[-1].imag, 0, atol=1e-13)
        assert_allclose(mellin.irmellin(half, c=c, dlog=0.2, norm=norm), a, atol=1e-13)


@pytest.mark.parametrize(
    "n,indices", [(5, [0, 1, 2, -2, -1]), (6, [0, 1, 2, -3, -2, -1])]
)
def test_frequency_order(x64, n, indices):
    spacing = 2 * np.pi / (n * 0.2)
    assert_allclose(mellin.mellinfreq(n, 0.2), np.array(indices) * spacing)
    assert_allclose(mellin.rmellinfreq(n, 0.2), np.arange(n // 2 + 1) * spacing)


def test_continuous_mellin_conversion(x64):
    n, dlog, log_t0, c = 256, 1 / 16, -8.0, 0.3
    x = log_t0 + np.arange(n) * dlog
    a = np.exp(-(x**2) / 2)
    eta = mellin.mellinfreq(n, dlog)
    s = c + 1j * eta
    actual = dlog * jnp.exp(log_t0 * s) * mellin.mellin(a, dlog=dlog, c=c)
    expected = np.sqrt(2 * np.pi) * jnp.exp(s**2 / 2)
    assert_allclose(actual, expected, atol=2e-11)


@pytest.mark.parametrize("name", ["mellin", "imellin", "rmellin", "irmellin"])
def test_jax_transformations_and_gradients(x64, name):
    transform = getattr(mellin, name)
    a = jnp.linspace(0.2, 0.8, 7)
    if name != "rmellin":
        a = a + 0.1j * a[::-1]

    def evaluate(scale, c, dlog):
        return transform(scale * a, n=8, c=c, dlog=dlog, norm="ortho")

    params = (1.2, 0.3, 0.2)
    assert_allclose(jax.jit(evaluate)(*params), evaluate(*params), atol=1e-13)
    values = jnp.array([0.2, 0.3, 0.4])
    for parameter in range(3):

        def mapped(value, parameter=parameter):
            args = list(params)
            args[parameter] = value
            return evaluate(*args)

        actual = jax.jit(jax.vmap(mapped))(values)
        assert_allclose(actual, jnp.stack([mapped(v) for v in values]), atol=1e-13)

        def loss(value):
            return jnp.sum(jnp.abs(mapped(value)) ** 2)

        value, step = params[parameter], 1e-5
        finite_difference = (loss(value + step) - loss(value - step)) / (2 * step)
        assert_allclose(jax.grad(loss)(value), finite_difference, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("name", ["mellinfreq", "rmellinfreq"])
def test_frequency_jax_transformations(x64, name):
    frequency = getattr(mellin, name)
    evaluate = lambda d: frequency(8, d)
    values = jnp.array([0.1, 0.2])
    assert_allclose(
        jax.jit(jax.vmap(evaluate))(values), jnp.stack([evaluate(v) for v in values])
    )
    assert_allclose(
        jax.grad(lambda d: jnp.sum(evaluate(d)))(0.2), -jnp.sum(evaluate(0.2)) / 0.2
    )


@pytest.mark.parametrize("name", ["mellin", "imellin", "rmellin", "irmellin"])
def test_single_precision(name):
    transform = getattr(mellin, name)
    a = jnp.linspace(0.2, 0.8, 8, dtype=jnp.float32)
    actual = transform(a, n=8, c=jnp.float32(0.3), dlog=jnp.float32(0.2))
    assert actual.dtype == (jnp.float32 if name == "irmellin" else jnp.complex64)


@pytest.mark.parametrize("name", ["mellin", "imellin", "rmellin", "irmellin"])
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"n": 0}, ValueError),
        ({"axis": 2}, ValueError),
        ({"norm": "invalid"}, ValueError),
        ({"c": 1j}, TypeError),
        ({"c": jnp.ones(2)}, ValueError),
        ({"dlog": jnp.ones(2)}, ValueError),
    ],
)
def test_invalid_arguments(name, kwargs, error):
    with pytest.raises(error):
        getattr(mellin, name)(jnp.ones(8), **kwargs)


def test_real_analysis_rejects_complex_samples():
    with pytest.raises(TypeError, match="real samples"):
        mellin.rmellin(jnp.ones(8, dtype=complex))


def test_weak_double_precision_is_preserved(x64):
    a = jnp.linspace(0.2, 0.8, 8)
    assert mellin.rmellin(a).dtype == jnp.complex128
    assert mellin.mellin(a).dtype == jnp.complex128
    assert mellin.irmellin(mellin.rmellin(a)).dtype == jnp.float64


@pytest.mark.parametrize("name", ["mellin", "imellin", "rmellin", "irmellin"])
def test_sample_batching_and_gradient(x64, name):
    transform = getattr(mellin, name)
    samples = jnp.linspace(0.2, 0.8, 14).reshape(2, 7)
    evaluate = lambda a: transform(a, n=8, c=0.3, dlog=0.2)
    assert_allclose(jax.jit(jax.vmap(evaluate))(samples), evaluate(samples), atol=1e-13)
    loss = lambda a: jnp.sum(jnp.abs(evaluate(a)) ** 2)
    a = samples[0]
    step = 1e-5
    expected = (loss(a.at[3].add(step)) - loss(a.at[3].add(-step))) / (2 * step)
    assert_allclose(jax.jit(jax.grad(loss))(a)[3], expected, rtol=1e-8, atol=1e-8)
