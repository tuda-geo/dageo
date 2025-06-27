import numpy as np
import scipy as sp
from numpy.testing import assert_allclose

import dageo
from dageo import reservoir_simulator


class TestSimulator:
    def test_input_and_defaults(self):
        RS = reservoir_simulator.Simulator(3, 2)
        assert RS.size == 6
        assert RS.shape == (3, 2)
        assert RS.nx == 3
        assert RS.ny == 2

        assert RS.phi == 0.2
        assert RS.c_f == 1e-5
        assert RS.p0 == 1.0
        assert RS.rho0 == 1.0
        assert RS.mu_w == 1.0
        assert RS.rw == 0.15
        assert RS.dx == 50.0
        assert RS.dz == 10.0
        assert RS.pres_ini == 150.0
        assert_allclose([0, 0, 180, 2, 1, 120], RS.wells.ravel())
        volumes = np.ones(6) * 25000
        assert_allclose(RS.volumes, volumes)
        assert_allclose(RS._vol_phi_cf, volumes * 0.2e-5)
        assert_allclose(RS.locs, [0, 5])

        wells = np.array([[1, 1, 130], [0, 3, 200]])
        RS = reservoir_simulator.Simulator(
            nx=2,
            ny=4,
            phi=0.3,
            c_f=1e-4,
            p0=0.8,
            rho0=0.9,
            mu_w=0.7,
            rw=0.13,
            pres_ini=140.0,
            dx=30.0,
            dz=20.0,
            wells=wells,
        )
        assert RS.size == 8
        assert RS.shape == (2, 4)
        assert RS.nx == 2
        assert RS.ny == 4

        assert RS.phi == 0.3
        assert RS.c_f == 1e-4
        assert RS.p0 == 0.8
        assert RS.rho0 == 0.9
        assert RS.mu_w == 0.7
        assert RS.rw == 0.13
        assert RS.dx == 30.0
        assert RS.dz == 20.0
        assert RS.pres_ini == 140.0
        assert_allclose(wells, RS.wells)
        assert_allclose(RS.locs, [3, 6])

    def test_check_status_quo(self):
        # Here we are only checking the Status Quo.
        # It would be great to also have some checks with analytical solutions!
        nx, ny = 3, 2
        RS = reservoir_simulator.Simulator(nx, ny)

        result = np.array(
            [
                [
                    [150.00000000, 150.00000000],
                    [150.00000000, 150.00000000],
                    [150.00000000, 150.00000000],
                ],
                [
                    [154.18781512, 150.91103108],
                    [150.54640165, 149.45359835],
                    [149.08896892, 145.81218488],
                ],
                [
                    [158.73203532, 153.71247787],
                    [151.25155563, 148.74874726],
                    [146.28778058, 141.26794637],
                ],
            ]
        )

        perm_fields = np.ones((nx, ny))
        dt = np.array([0.001, 0.1])
        out1 = RS(perm_fields, dt=dt)

        assert_allclose(out1, result)

        out2 = RS(np.array([perm_fields, perm_fields]), dt=dt)
        assert_allclose(out1, out2[0])
        assert_allclose(out1, out2[1])

        out3 = RS(perm_fields, dt=dt, data=[1, 1])
        assert_allclose(out1[:, 1, 1], out3)

    def test_check_1d(self):
        nx, ny = 1, 10
        wells1 = np.array([[0, 0, 180], [0, 9, 120]])
        wells2 = np.array([[0, 0, 180], [9, 0, 120]])
        perm_fields = np.ones((nx, ny))
        dt = np.array([0.001, 0.1])

        RS1 = reservoir_simulator.Simulator(nx, ny, wells=wells1)
        RS2 = reservoir_simulator.Simulator(ny, nx, wells=wells2)

        out1 = RS1(perm_fields, dt=dt)
        out2 = RS2(perm_fields, dt=dt)

        assert_allclose(
            out1.reshape(dt.size + 1, -1),
            out2.reshape(dt.size + 1, -1, order="F"),
        )


class TestRandomPermeability:
    def test_input_and_defaults(self):
        RP = reservoir_simulator.RandomPermeability(3, 2, 0.5, 0.2, 0.8)
        assert RP.nx == 3
        assert RP.ny == 2
        assert RP.nc == 6
        assert RP.perm_mean == 0.5
        assert RP.perm_min == 0.2
        assert RP.perm_max == 0.8
        assert RP.length == (10.0, 10.0)
        assert RP.theta == 45.0
        assert RP.variance == 1.0
        assert RP.dtype == "float32"

        RP = reservoir_simulator.RandomPermeability(
            3, 2, 0.5, 0.2, 0.8, (3.0, 4.0), 0.0, 2.1, "float64"
        )
        assert RP.length == (3.0, 4.0)
        assert RP.theta == 0.0
        assert RP.variance == 2.1
        assert RP.dtype == "float64"

    def test_cov_lcho(self):
        # cov is just a call to utils.gaussian_covariance - check.
        RP = reservoir_simulator.RandomPermeability(3, 2, 0.5, 0.2, 0.8)
        assert_allclose(
            RP.cov,
            dageo.utils.gaussian_covariance(
                3, 2, RP.length, RP.theta, RP.variance
            ),
        )
        # lcho is just a call to sp.linalg.cholesky - check.
        assert_allclose(RP.lcho, sp.linalg.cholesky(RP.cov, lower=True))

    def test_call(self):
        RP = reservoir_simulator.RandomPermeability(3, 2, 0.5, 0.0, 1.0)
        assert_allclose(RP(1, 0.5, 0.5, 0.5), 0.5)

        rng = dageo.utils.rng(4)
        result = np.array(
            [
                [
                    [0.00000000, 0.29639514],
                    [0.00000000, 0.00000000],
                    [0.83044794, 0.27682457],
                ]
            ]
        )
        assert_allclose(RP(1, random=rng), result, rtol=1e-6)


def test_all_dir():
    assert set(reservoir_simulator.__all__) == set(dir(reservoir_simulator))
