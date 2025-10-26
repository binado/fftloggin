"""
Test suite for FFTLog implementation using Grid API.
Adapted from scipy's test suite with Grid-based interface.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less
from scipy.special import poch

from fftloggin.fftlog import FFTLog
from fftloggin.kernels import BesselJKernel, Kernel, SphericalBesselJKernel


# test function, analytical Hankel transform is of the same form
def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


def test_grid_agrees_with_fftlog():
    """Check that Grid numerically agrees with the output from Fortran FFTLog."""
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    logc = 0.0  # offset parameter from old API maps directly to logc
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Test 1: compute as given
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=bias), logc=logc, minimize_ringing=False
    )
    fftlog.create_grid(r=r)
    ours = fftlog.forward(a)

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


def test_grid_with_optimal_logc():
    """Test Grid with optimal logc (minimize_ringing=True)."""
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Create grid with optimal logc
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=bias), logc=0.0, minimize_ringing=True
    )
    fftlog.create_grid(r=r)
    ours = fftlog.forward(a)

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


def test_grid_with_positive_bias():
    """Test Grid with positive bias."""
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = 0.8

    a = np.asarray(f(r, mu))

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=bias), logc=0.0, minimize_ringing=True
    )
    fftlog.create_grid(r=r)
    ours = fftlog.forward(a)

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


def test_grid_with_negative_bias():
    """Test Grid with negative bias."""
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = -0.8

    a = np.asarray(f(r, mu))

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=bias), logc=0.0, minimize_ringing=True
    )
    fftlog.create_grid(r=r)
    ours = fftlog.forward(a)

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


def test_vectorized_grid():
    """Test Grid with vectorized kernel (multiple mu values)."""
    n = 16
    r = np.logspace(-4, 4, n)

    # Test scalar mu
    mu = 0.3
    a = f(r, mu)
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=0.0), logc=0.0, minimize_ringing=False
    )
    fftlog.create_grid(r=r)
    out = fftlog.forward(a)
    assert out.shape == r.shape

    # Test 1d mu (single batch element)
    mu = np.array([0.3])
    a = f(r, mu)
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=0.0), logc=0.0, minimize_ringing=False
    )
    fftlog.create_grid(r=r)
    out = fftlog.forward(a)
    assert out.shape == (n,)

    # Test 1d mu (multiple batch elements)
    mu = np.linspace(0.1, 0.3, 3).reshape(-1, 1)
    a = f(r, mu)
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=0.0), logc=0.0, minimize_ringing=False
    )
    fftlog.create_grid(r=r)
    out = fftlog.forward(a)
    assert out.shape == (3, n)


@pytest.mark.parametrize("logc", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("bias", [0.1, -0.1])
@pytest.mark.parametrize("n", [64, 63])
@pytest.mark.parametrize("kernel_cls", [BesselJKernel, SphericalBesselJKernel])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("minimize_ringing", [False])
def test_grid_identity(
    n: int,
    bias: float,
    logc: float,
    kernel_cls: type[Kernel],
    order: int,
    minimize_ringing: bool,
):
    """Test that inverse is the inverse of forward for various kernels and derivatives."""
    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(n))
    dlog = 0.1

    # Create grid for forward transform
    r = np.exp(np.arange(n) * dlog)

    # Bias correction for derivatives
    # bias = bias + order

    # Create kernel instance based on type
    mu = rng.uniform(3, 5)
    if kernel_cls is BesselJKernel:
        kernel = kernel_cls(mu, bias=bias)
    else:  # SphericalBesselJKernel
        ell = np.ceil(mu - 0.5)
        kernel = kernel_cls(ell, bias=bias)

    # Apply derivative if needed
    if order > 0:
        kernel = kernel.derive(order)

    # Create FFTLog for forward and inverse transforms
    fftlog = FFTLog.from_array(
        r, kernel=kernel, logc=logc, minimize_ringing=minimize_ringing
    )
    grid_fwd = fftlog.create_grid(r=r)
    A = fftlog.forward(a)

    # Create grid for inverse transform with same FFTLog
    fftlog.create_grid(k=grid_fwd.k)
    a_reconstructed = fftlog.inverse(A)

    assert_allclose(a_reconstructed, a, rtol=1.5e-7)


def test_grid_special_cases():
    """Test special cases and singularities."""
    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(64))
    dlog_val = rng.uniform(-1, 1)
    r = np.exp(np.arange(64) * dlog_val)

    # let x = (mu+1+q)/2, y = (mu+1-q)/2, M = {0, -1, -2, ...}

    # case 1: x in M, y in M => well-defined transform
    mu, bias = -4.0, 1.0
    with warnings.catch_warnings(record=True) as record:
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu, bias=bias))
        fftlog.forward(a)
        assert not record, "FFTLog warned about a well-defined transform"

    # case 2: x not in M, y in M => well-defined transform
    mu, bias = -2.5, 0.5
    with warnings.catch_warnings(record=True) as record:
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu, bias=bias))
        fftlog.forward(a)
        assert not record, "FFTLog warned about a well-defined transform"

    # case 3: x in M, y not in M => singular transform
    mu, bias = -3.5, 0.5
    with pytest.warns(Warning) as record:
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu, bias=bias))
        fftlog.forward(a)
        assert record, "FFTLog did not warn about a singular transform"

    # case 4: x not in M, y in M => singular inverse transform
    mu, bias = -2.5, 0.5
    with pytest.warns(Warning) as record:
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu, bias=bias))
        fftlog.inverse(a)
        assert record, "FFTLog did not warn about a singular transform"


@pytest.mark.parametrize("n", [64, 63])
def test_grid_exact(n):
    """Test exact transform for power law functions."""
    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \\gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1 - mu, 1 / 2)

    r = np.logspace(-2, 2, n)
    a = np.asarray(r**gamma)

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=gamma), logc=0.0, minimize_ringing=True
    )
    grid = fftlog.create_grid(r=r)
    A = fftlog.forward(a)

    k = grid.k

    # analytical result
    At = np.asarray((2 / k) ** gamma * poch((mu + 1 - gamma) / 2, gamma))

    assert_allclose(A, At)


def test_array_like():
    """Test that array-like inputs work."""
    x = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    r = np.array([1.0, 2.0])

    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(2.0))
    fftlog.create_grid(r=r)
    result1 = fftlog.forward(x)
    result2 = fftlog.forward(np.asarray(x))

    assert_allclose(result1, result2)


@pytest.mark.parametrize("n", [128, 129])
def test_gh_21661(n):
    """Test for github issue 21661."""
    one = np.asarray(1.0)
    mu = 0.0
    r = np.logspace(-7, 1, n)

    # Using logc parameter (offset from old API)
    logc = -6 * np.log(10)
    r = np.asarray(r, dtype=one.dtype)

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu, bias=0.0), logc=logc, minimize_ringing=False
    )
    grid = fftlog.create_grid(r=r)
    k = grid.k

    def f_test(x, mu):
        return x ** (mu + 1) * np.exp(-(x**2) / 2)

    a_r = f_test(r, mu)
    fht_val = fftlog.forward(a_r)
    a_k = f_test(k, mu)
    rel_err = np.max(np.abs((fht_val - a_k) / a_k))
    assert_array_less(rel_err, np.asarray(7.28e16)[()])
