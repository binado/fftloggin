"""
Test suite for FFTLog implementation using Grid API.
Adapted from scipy's test suite with Grid-based interface.
"""

from numbers import Number

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.special import poch

from fftloggin.fftlog import FFTLog
from fftloggin.kernels import BesselJKernel
from fftloggin.utils import prepare_batch_params


# test function, analytical Hankel transform is of the same form
def f(r, mu):
    return r ** (mu + 1) * np.exp(-(r**2) / 2)


def test_fftlog_agrees_with_fortran():
    """
    Check that FFTLog numerically agrees with the output from Fortran FFTLog.
    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    kr = 1.0  # product k*r at the geometric center of the grid
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Test 1: compute as given
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu), bias=bias, kr=kr, lowring=False
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


def test_fftlog_with_optimal_kr():
    """
    Test fftlog with optimal kr (lowring=True).
    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = 0.0

    a = np.asarray(f(r, mu))

    # Create grid with optimal kr
    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu), bias=bias, kr=1.0, lowring=True
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


def test_fftlog_with_positive_bias():
    """
    Test fftlog with positive bias.
    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = 0.8

    a = np.asarray(f(r, mu))

    # This value for the bias lies outside the strip of definition
    # of the kernel, but we skip the bound checking here for
    # compatibility with scipy's implementation.
    kernel = BesselJKernel(mu, check_bounds=False)
    fftlog = FFTLog.from_array(
        r,
        kernel=kernel,
        bias=bias,
        kr=1.0,
        lowring=True,
    )
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


def test_fftlog_with_negative_bias():
    """
    Test Grid with negative bias.

    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    r = np.logspace(-4, 4, 16)
    mu = 0.3
    bias = -0.8

    a = np.asarray(f(r, mu))

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu), bias=bias, kr=1.0, lowring=True
    )
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


def test_fftlog_with_vectorized_kernel():
    """Test Grid with vectorized kernel (multiple mu values)."""
    n = 16
    r = np.logspace(-4, 4, n)

    # Test scalar mu
    mu = 0.3
    a = f(r, mu)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=1.0, lowring=False)
    out = fftlog.forward(a)
    assert out.shape == r.shape

    # Test 1d mu (single batch element)
    mu = np.array([0.3])
    a = f(r, mu)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=1.0, lowring=False)
    out = fftlog.forward(a)
    assert out.shape == (n,)

    # Test 1d mu (multiple batch elements)
    mu = np.linspace(0.1, 0.3, 3).reshape(-1, 1)
    a = f(r, mu)
    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=1.0, lowring=False)
    out = fftlog.forward(a)
    assert out.shape == (3, n)


@pytest.mark.parametrize("kr", [1.0, np.exp(1.0), np.exp(-1.0)])
@pytest.mark.parametrize("bias", [0.1, -0.1])
@pytest.mark.parametrize("n", [64, 63])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("lowring", [False])
def test_fftlog_identity(
    n: int,
    bias: float,
    kr: float,
    order: int,
    lowring: bool,
):
    """Test that inverse is the inverse of forward for various kernels and derivatives."""
    rng = np.random.RandomState(3491349965)

    a = np.asarray(rng.standard_normal(n))
    dlog = 0.1

    # Create grid for forward transform
    r = np.exp(np.arange(n) * dlog)

    # Create kernel instance
    mu = rng.uniform(3, 5)
    kernel = BesselJKernel(mu)

    # Apply derivative if needed
    if order > 0:
        kernel = kernel.derive(order)

    # Create FFTLog for forward and inverse transforms
    fftlog = FFTLog.from_array(r, kernel=kernel, bias=bias, kr=kr, lowring=lowring)
    A = fftlog.forward(a)

    # Create grid for inverse transform with same FFTLog
    a_reconstructed = fftlog.inverse(A)

    assert_allclose(a_reconstructed, a, rtol=1.5e-7)


@pytest.mark.parametrize("n", [64, 63])
def test_fftlog_exact(n):
    """
    Test exact transform for power law functions.
    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    rng = np.random.RandomState(3491349965)

    # for a(r) a power law r^\\gamma, the fast Hankel transform produces the
    # exact continuous Hankel transform if biased with q = \\gamma

    mu = rng.uniform(0, 3)

    # convergence of HT: -1-mu < gamma < 1/2
    gamma = rng.uniform(-1 - mu, 1 / 2)

    r = np.logspace(-2, 2, n)
    a = np.asarray(r**gamma)

    fftlog = FFTLog.from_array(
        r, kernel=BesselJKernel(mu), bias=gamma, kr=1.0, lowring=True
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
    result1 = fftlog.forward(x)
    result2 = fftlog.forward(np.asarray(x))

    assert_allclose(result1, result2)


@pytest.mark.parametrize("n", [128, 129])
def test_gh_21661(n):
    """
    Test for github issue 21661.
    This test is adapted from scipy's test suite, see
    https://github.com/scipy/scipy/blob/main/scipy/special/tests/test_fftlog.py
    """
    one = np.asarray(1.0)
    mu = 0.0
    r = np.logspace(-7, 1, n)

    # Using kr parameter (product k*r at geometric center)
    kr = np.exp(-6 * np.log(10))
    r = np.asarray(r, dtype=one.dtype)

    fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=kr, lowring=False)
    grid = fftlog.create_grid(r=r)
    k = grid.k

    def f_test(x, mu):
        return x ** (mu + 1) * np.exp(-(x**2) / 2)

    a_r = f_test(r, mu)
    fht_val = fftlog.forward(a_r)
    a_k = f_test(k, mu)
    rel_err = np.max(np.abs((fht_val - a_k) / a_k))
    assert_array_less(rel_err, np.asarray(7.28e16)[()])


class TestBatchedTransforms:
    """Tests for batched transforms with array parameters."""

    def test_batched_kr_only(self):
        """Test batching with array kr, scalar dlog and bias."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        # Batch over 3 different kr values
        kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=kr, lowring=False)
        assert isinstance(fftlog.dlog, Number)
        assert isinstance(fftlog.bias, Number)

        assert isinstance(fftlog.kr, np.ndarray)
        assert fftlog.kr.shape == kr.shape
        assert isinstance(fftlog.logc, np.ndarray)
        assert fftlog.logc.shape == kr.shape

        grid = fftlog.create_grid(r=r)
        assert grid.r.shape == r.shape
        assert grid.k.shape == (3, 128)

        result = fftlog.forward(a)
        assert result.shape == (3, 128)

    def test_batched_dlog_only(self):
        """Test batching with array dlog, scalar bias and kr."""
        r = np.logspace(-2, 2, 64)
        mu = 0.3

        # Batch over 2 different dlog values
        dlog1 = np.log(r[1] / r[0])
        dlog2 = dlog1 * 1.5  # Different spacing
        dlog = np.array([dlog1, dlog2]).reshape(-1, 1)

        # Test with lowring=False
        kr_scalar = 1.0
        fftlog = FFTLog(
            kernel=BesselJKernel(mu), n=64, dlog=dlog, kr=kr_scalar, lowring=False
        )
        assert isinstance(fftlog.bias, Number)
        assert isinstance(fftlog.kr, Number)
        assert isinstance(fftlog.dlog, np.ndarray)
        assert fftlog.dlog.shape == dlog.shape
        assert_array_equal(fftlog.dlog, dlog)

        # Test with lowring=True
        a = f(r, mu)
        result = fftlog.forward(a)
        assert result.shape == (2, 64)

        kr_scalar = 1.0
        fftlog = FFTLog(
            kernel=BesselJKernel(mu), n=64, dlog=dlog, kr=kr_scalar, lowring=True
        )
        assert isinstance(fftlog.bias, Number)
        # kr now should have the same shape as dlog
        assert isinstance(fftlog.kr, np.ndarray)
        assert fftlog.kr.shape == dlog.shape
        assert isinstance(fftlog.dlog, np.ndarray)
        assert fftlog.dlog.shape == dlog.shape
        assert_array_equal(fftlog.dlog, dlog)

    def test_batched_bias_only(self):
        """Test batching with array bias, scalar dlog and kr."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        # Batch over 2 different bias values
        bias = np.array([0.0, 0.5]).reshape(-1, 1)
        kr_scalar = 1.0
        fftlog = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), bias=bias, kr=kr_scalar, lowring=False
        )

        result = fftlog.forward(a)
        assert result.shape == (2, 128)
        assert isinstance(fftlog.dlog, Number)
        assert isinstance(fftlog.kr, Number)
        assert kr_scalar == fftlog.kr

        kr_scalar = 1.0
        fftlog = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), bias=bias, kr=kr_scalar, lowring=True
        )
        result = fftlog.forward(a)
        assert result.shape == (2, 128)
        assert isinstance(fftlog.dlog, Number)
        # Bias should promote kr to an array
        assert isinstance(fftlog.kr, np.ndarray)
        assert fftlog.kr.shape == bias.shape

    def test_all_params_batched_compatible_shapes(self):
        """Test batching with all three parameters having compatible shapes."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        # All have shape (2, 1) - compatible
        dlog, bias, kr = prepare_batch_params([0.04, 0.05], [0.0, 0.1], [1.0, 2.0])

        fftlog = FFTLog(
            kernel=BesselJKernel(mu), n=128, dlog=dlog, bias=bias, kr=kr, lowring=False
        )

        result = fftlog.forward(a)
        assert result.shape == (2, 128)

    def test_prepare_batch_params_helper(self):
        """Test using prepare_batch_params helper function."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        # Use helper to prepare parameters
        dlog, bias, kr = prepare_batch_params(0.05, 0.0, [0.5, 1.0, 2.0])

        fftlog = FFTLog(
            kernel=BesselJKernel(mu), n=128, dlog=dlog, bias=bias, kr=kr, lowring=False
        )

        result = fftlog.forward(a)
        assert result.shape == (3, 128)

        # kr is scalar, so dlog and bias stay scalar
        assert dlog.shape == ()
        assert bias.shape == ()
        assert kr.shape == (3, 1)

    def test_batched_round_trip(self):
        """Test forward and inverse transforms with batched parameters."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=kr, lowring=False)

        # Forward transform with 1D input broadcasts to (3, 128)
        A = fftlog.forward(a)
        assert A.shape == (3, 128)

        # Inverse transform - passing full batch
        a_reconstructed = fftlog.inverse(A)
        assert a_reconstructed.shape == (3, 128)

        # Each batch should approximately reconstruct original
        # Note: FFTLog is not exact, so we use a loose tolerance
        for i in range(3):
            assert_allclose(a_reconstructed[i], a, rtol=1e-2, atol=1e-2)

    def test_batched_backward_compat_scalar(self):
        """Test that scalar parameters still work (backward compatibility)."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        # All scalars - should work as before
        fftlog = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), bias=0.0, kr=1.0, lowring=False
        )

        result = fftlog.forward(a)
        assert result.shape == (128,)  # No batch dimension

    def test_batched_vs_loop_consistency(self):
        """Test that batched transforms match individual transforms in a loop."""
        r = np.logspace(-2, 2, 128)
        a = f(r, 0.3)
        mu = 0.3

        kr_values = [0.5, 1.0, 2.0]

        # Batched transform
        kr_batch = np.array(kr_values).reshape(-1, 1)
        fftlog_batch = FFTLog.from_array(
            r, kernel=BesselJKernel(mu), kr=kr_batch, lowring=False
        )
        result_batch = fftlog_batch.forward(a)

        # Individual transforms in a loop
        results_loop = []
        for kr_val in kr_values:
            fftlog_single = FFTLog.from_array(
                r, kernel=BesselJKernel(mu), kr=kr_val, lowring=False
            )
            result_single = fftlog_single.forward(a)
            results_loop.append(result_single)
        results_loop = np.array(results_loop)

        # Should match
        assert_allclose(result_batch, results_loop, rtol=1e-10)

    def test_batched_kernel_coefficients_shape(self):
        """Test that kernel_coefficients have correct shape for batched params."""
        r = np.logspace(-2, 2, 128)
        mu = 0.3

        kr = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
        fftlog = FFTLog.from_array(r, kernel=BesselJKernel(mu), kr=kr, lowring=False)

        coeffs = fftlog.kernel_coefficients
        ns = 128 // 2 + 1
        assert coeffs.shape == (3, ns)

    def test_batched_optimal_logcenter(self):
        """Test optimal_logcenter with batched dlog."""
        mu = 0.3

        dlog = np.array([0.04, 0.05]).reshape(-1, 1)
        fftlog = FFTLog(
            kernel=BesselJKernel(mu), n=128, dlog=dlog, kr=1.0, lowring=False
        )

        logc_opt = fftlog.optimal_logcenter()
        assert logc_opt.shape == (2, 1)
