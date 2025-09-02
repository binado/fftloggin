"""
Test suite for vectorized FFTLog implementation.
Adapted from scipy's test suite with additional tests for vectorization and numexpr.
"""

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy.fft import fhtoffset as fhtoffset_scipy
from scipy.fft._fftlog_backend import fhtcoeff as fhtcoeff_scipy
from scipy.special import poch

from fftloggin._backend import _fhtcoeff
from fftloggin._backend_numexpr import NUMEXPR_AVAILABLE
from fftloggin.fht import fht, fhtoffset, ifht


# test function, analytical Hankel transform is of the same form
def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


@pytest.mark.parametrize("bias", [0, 0.8, -0.8])
def test_fht_coefficients_agree_with_scipy(bias: float):
    r = np.logspace(-4, 4, 16)

    dln = math.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0

    a = np.asarray(f(r, mu))
    n = a.shape[-1]
    ours = _fhtcoeff(n, dln, mu, offset=offset, bias=bias)
    theirs = fhtcoeff_scipy(n, dln, mu, offset=offset, bias=bias)
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bias", [0, 0.1, -0.1])
def test_fhtoffset_agrees_with_scipy(offset: float, bias: float):
    r = np.logspace(-4, 4, 16)

    dln = math.log(r[1] / r[0])
    mu = 0.3

    ours = fhtoffset(dln, mu, initial=offset, bias=bias)
    theirs = fhtoffset_scipy(dln, mu, initial=offset, bias=bias)
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fht_agrees_with_fftlog(use_numexpr):
    """Check that fht numerically agrees with the output from Fortran FFTLog."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)

    dln = math.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0

    a = np.asarray(f(r, mu))

    # test 1: compute as given
    ours = fht(a, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)
    theirs = [
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
    ]
    theirs = np.asarray(theirs, dtype=np.float64)
    assert_allclose(ours, theirs)

    # test 2: change to optimal offset (using 'initial' for scipy compatibility)
    offset = fhtoffset(dln, mu, initial=0.0, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)
    theirs = [
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
    ]
    theirs = np.asarray(theirs, dtype=np.float64)
    assert_allclose(ours, theirs)

    # test 3: positive bias
    bias = 0.8
    offset = fhtoffset(dln, mu, initial=0.0, bias=bias)

    ours = fht(a, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)
    theirs = [
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
    ]
    theirs = np.asarray(theirs, dtype=np.float64)
    assert_allclose(ours, theirs)

    # test 4: negative bias
    bias = -0.8
    offset = fhtoffset(dln, mu, initial=0.0, bias=bias)

    ours = fht(a, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)
    theirs = [
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
        10.5251075070045800e00,
    ]
    theirs = np.asarray(theirs, dtype=np.float64)
    assert_allclose(ours, theirs)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_vectorized_fht(use_numexpr):
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    n = 16
    r = np.logspace(-4, 4, n)
    dln = math.log(r[1] / r[0])

    # Test scalar mu
    mu = 0.3
    a = f(r, mu)
    offset = 0.0
    bias = 0.0
    out = fht(a, dln, mu, offset, bias, use_numexpr=use_numexpr)
    assert out.shape == r.shape

    # Test 1d mu
    mu = np.array([0.3]).reshape(1, 1)
    a = f(r, mu)
    offset = 0.0
    bias = 0.0
    out = fht(a, dln, mu, offset, bias, use_numexpr=use_numexpr)
    assert out.shape == (1, n)

    # Test 2d mu
    mu = np.linspace(0.1, 0.3, 3).reshape(3, 1, 1)
    a = f(r, mu)
    offset = 0.0
    bias = 0.0
    out = fht(a, dln, mu, offset, bias, use_numexpr=use_numexpr)
    assert out.shape == (3, 1, n)


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("optimal", [True, False])
@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bias", [0, 0.1, -0.1])
@pytest.mark.parametrize("n", [64, 63])
def test_fht_identity(n, bias, offset, optimal, use_numexpr):
    """Test that ifht is the inverse of fht."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(n))
    dln = rng.uniform(-1, 1)
    mu = rng.uniform(-2, 2)

    if optimal:
        # Note: using 'offset' parameter instead of 'initial' for our implementation
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)

    A = fht(a, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)
    a_ = ifht(A, dln, mu, offset=offset, bias=bias, use_numexpr=use_numexpr)

    assert_allclose(a_, a, rtol=1.5e-7)


@pytest.mark.parametrize("use_numexpr", [True, False])
def test_fht_special_cases(use_numexpr):
    """Test special cases and singularities."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(64))
    dln = rng.uniform(-1, 1)

    # let x = (mu+1+q)/2, y = (mu+1-q)/2, M = {0, -1, -2, ...}

    # case 1: x in M, y in M => well-defined transform
    mu, bias = -4.0, 1.0
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias, use_numexpr=use_numexpr)
        assert not record, "fht warned about a well-defined transform"

    # case 2: x not in M, y in M => well-defined transform
    mu, bias = -2.5, 0.5
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias, use_numexpr=use_numexpr)
        assert not record, "fht warned about a well-defined transform"

    # case 3: x in M, y not in M => singular transform
    mu, bias = -3.5, 0.5
    with pytest.warns(Warning) as record:
        fht(a, dln, mu, bias=bias, use_numexpr=use_numexpr)
        assert record, "fht did not warn about a singular transform"

    # case 4: x not in M, y in M => singular inverse transform
    mu, bias = -2.5, 0.5
    with pytest.warns(Warning) as record:
        ifht(a, dln, mu, bias=bias, use_numexpr=use_numexpr)
        assert record, "ifht did not warn about a singular transform"


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("n", [64, 63])
def test_fht_exact(n, use_numexpr):
    """Test exact transform for power law functions."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1 - mu, 1 / 2)

    r = np.logspace(-2, 2, n)
    a = np.asarray(r**gamma)

    dln = math.log(r[1] / r[0])

    offset = fhtoffset(dln, mu, initial=0.0, bias=gamma)

    A = fht(a, dln, mu, offset=offset, bias=gamma, use_numexpr=use_numexpr)

    k = np.exp(offset) / r[::-1]

    # analytical result
    At = np.asarray((2 / k) ** gamma * poch((mu + 1 - gamma) / 2, gamma))

    assert_allclose(A, At)


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("op", [fht, ifht])
def test_array_like(op, use_numexpr):
    """Test that array-like inputs work."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    x = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    assert_allclose(
        op(x, 1.0, 2.0, use_numexpr=use_numexpr),
        op(np.asarray(x), 1.0, 2.0, use_numexpr=use_numexpr),
    )


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize("n", [128, 129])
def test_gh_21661(n, use_numexpr):
    """Test for github issue 21661."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    one = np.asarray(1.0)
    mu = 0.0
    r = np.logspace(-7, 1, n)
    dln = math.log(r[1] / r[0])
    # Using 'offset' parameter instead of 'initial'
    offset = fhtoffset(dln, initial=-6 * np.log(10), mu=mu)
    r = np.asarray(r, dtype=one.dtype)
    k = math.exp(offset) / np.flip(r, axis=-1)

    def f(x, mu):
        return x ** (mu + 1) * np.exp(-(x**2) / 2)

    a_r = f(r, mu)
    fht_val = fht(a_r, dln, mu=mu, offset=offset, use_numexpr=use_numexpr)
    a_k = f(k, mu)
    rel_err = np.max(np.abs((fht_val - a_k) / a_k))
    assert_array_less(rel_err, np.asarray(7.28e16)[()])


@pytest.mark.parametrize("use_numexpr", [True, False])
@pytest.mark.parametrize(
    "shape, axis",
    [
        ((1, 16, 1), 1),
        ((16, 1, 1), 0),
        ((1, 1, 16), 2),
        ((1, 16, 1), -2),
        ((16, 1, 1), -3),
        ((1, 1, 16), -1),
    ],
)
def test_fht_axis_parameter(shape, axis, use_numexpr):
    """Test that the axis parameter of fht works correctly."""
    if use_numexpr and not NUMEXPR_AVAILABLE:
        pytest.skip("numexpr not available")

    r = np.logspace(-4, 4, 16)
    dln = math.log(r[1] / r[0])
    mu = 0.3
    a = np.asarray(f(r, mu))

    a_reshaped = np.reshape(a, shape)
    A = fht(a_reshaped, dln, mu, axis=axis, use_numexpr=use_numexpr)
    assert A.shape == shape
