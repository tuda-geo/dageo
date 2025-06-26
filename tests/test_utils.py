import numpy as np
import scooby
from numpy.testing import assert_allclose

from dageo import utils


def test_gaussian_covariance():
    # Small length, no variance => eye
    assert_allclose(
        utils.gaussian_covariance(4, 4, [0.1, 0.1], 0, 0), np.eye(4 * 4)
    )

    # Small length, some variance => still eye
    assert_allclose(
        utils.gaussian_covariance(
            nx=4, ny=4, length=[0.1, 0.1], theta=0, variance=1
        ),
        np.eye(4 * 4),
    )

    # Simply check some values with variance
    x = utils.gaussian_covariance(4, 4, [2, 1], 30, 2)
    assert_allclose(np.diag(x), 1.0)
    assert_allclose(np.diag(x, -1)[3::4], 0)
    for i in range(3):
        assert_allclose(np.diag(x, -1)[i::4], 0.00024027168)
    assert_allclose(np.diag(x, -4), 0.58600724)
    for i in [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        assert_allclose(np.diag(x, -i), 0)

    # Simply check some values without variance
    x = utils.gaussian_covariance(4, 4, [1, 2], 0, 0)
    assert_allclose(np.diag(x), 1.0)
    assert_allclose(np.diag(x, -1)[0], 0.20833333)
    assert_allclose(np.diag(x, -3)[1], 0.13466999)
    assert_allclose(np.diag(x, -4)[0], 0.6848958)
    assert_allclose(np.diag(x, -5)[0], 0.13466999)
    assert_allclose(np.diag(x, -7)[1], 0.030032475)
    assert_allclose(np.diag(x, -8)[0], 0.20833333)
    assert_allclose(np.diag(x, -9)[0], 0.030032475)
    assert_allclose(np.diag(x, -11)[1], 0.0004445008)
    assert_allclose(np.diag(x, -12)[0], 0.016493056)
    assert_allclose(np.diag(x, -13)[0], 0.0004445008)
    for i in [2, 6, 10, 14, 15]:
        assert_allclose(np.diag(x, -i), 0)


def test_localization_matrix():
    # [[ 0,  0,  0,  0],
    #  [ 4,  5,  0,  0],
    #  [ 8,  9, 10,  0],
    #  [12, 13, 14, 15]]
    tril = np.tril(np.arange(16).reshape(4, 4))
    full = tril + np.tril(tril, -1).T
    triu = np.triu(full)

    data_positions = np.array([[1, 0]], dtype=int)
    solution = np.array([[[4]], [[5]], [[9]], [[13]]])
    outtril = utils.localization_matrix(tril, data_positions, (4, 1))
    assert_allclose(solution, outtril)
    outtriu = utils.localization_matrix(triu, data_positions, (4, 1), "upper")
    assert_allclose(solution, outtriu)
    outfull = utils.localization_matrix(full, data_positions, (4, 1), "full")
    assert_allclose(solution, outfull)


def test_random():
    assert isinstance(utils.rng(), np.random.Generator)
    assert isinstance(utils.rng(11), np.random.Generator)
    rng = np.random.default_rng()
    assert rng == utils.rng(rng)


def test_Report(capsys):
    out, _ = capsys.readouterr()  # Empty capsys

    # Reporting is done by the external package scooby.
    # We just ensure the shown packages do not change (core and optional).
    out1 = utils.Report()
    out2 = scooby.Report(
        core=["dageo", "numpy", "scipy"],
        optional=["matplotlib", "IPython"],
        ncol=3,
    )

    # Ensure they're the same; exclude time to avoid errors.
    assert out1.__repr__()[115:] == out2.__repr__()[115:]


def test_all_dir():
    assert set(utils.__all__) == set(dir(utils))
