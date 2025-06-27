# Copyright 2024 D. Werthmüller, G. Serrao Seabra, F.C. Vossepoel
#
# This file is part of dageo.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

import numpy as np
import scipy as sp

from dageo import utils

__all__ = ["Simulator", "RandomPermeability"]


def __dir__():
    return __all__


class Simulator:
    """A small 2D Reservoir Simulator.

    2D connection-based, single-phase, single-component system using the Darcy
    assumption, employing backward Euler for time discretization and finite
    volume for space discretization. It simulates a single-phase fluid (likely
    water) with compressibility in a reservoir with constant porosity and a
    heterogeneous permeability field. The simulator utilizes SciPy's
    ``sparse.linalg.solve`` for pressure solutions and calculates
    transmissibilities for inter-block connections. Well modeling is handled
    through the Peaceman well model for well indices, with constant pressure
    boundary conditions for injection and production wells. The simulation
    operates on a 2D grid with user-defined dimensions (nx, ny), uses flexible
    time steps, and starts from a specified initial pressure condition.
    Physical processes accounted for include fluid density changes with
    pressure (modeling a slightly compressible fluid) while assuming constant
    fluid viscosity."

    Created by following the course
    **AESM304A - Flow and Simulation of Subsurface processes** at
    Delft University of Technology (TUD); this particular part was taught by
    Dr. D.V. Voskov, https://orcid.org/0000-0002-5399-1755.


    Parameters
    ----------
    nx, ny : int
        Dimension of field (number of cells).
    phi : float, default: 0.2
        Porosity (-).
    c_f : float, default: 1e-5
        Formation compressibility (1/kPa).
    p0 : float, default: 1.0
        Zero pressure (bar).
    rho0 : float, default: 1.0
        Fixed density (kg/m3).
    mu_w : float, default: 1.0
        Viscosity (cP).
    rw : float, default: 0.15
        Well radius (m).
    pres_ini : float, default: 150.0
        Initial well pressure (bar).
    wells : {ndarray, None}, default: None
        Nd array of shape ``(nwells, 3)``, indicating well locations (with cell
        indices) and pressure. If None, the default is used, which is

            np.array([[0, 0, 180], [self.nx-1, self.ny-1, 120]])

        corresponding to a well in the first and in the last cell, with a
        pressure of 180 and 120, respectively.
    dx, dz : floats, default: 50.0, 10.0
        Cell dimensions in horizontal (dx) and vertical (dz)
        directions (m).

    """

    def __init__(
        self,
        nx,
        ny,
        phi=0.2,
        c_f=1e-5,
        p0=1.0,
        rho0=1.0,
        mu_w=1.0,
        rw=0.15,
        pres_ini=150.0,
        wells=None,
        dx=50.0,
        dz=10.0,
    ):
        """Initialize a Simulation instance."""

        self.size = nx * ny
        self.shape = (nx, ny)
        self.nx = nx
        self.ny = ny

        self.phi = phi
        self.c_f = c_f
        self.p0 = p0
        self.rho0 = rho0
        self.mu_w = mu_w
        self.rw = rw
        self.dx = dx
        self.dz = dz
        self.pres_ini = pres_ini

        # Store volumes (needs adjustment for arbitrary cell volumes)
        self.volumes = np.ones(self.size) * self.dx * self.dx * self.dz
        self._vol_phi_cf = self.volumes * self.phi * self.c_f

        if wells is None:
            # Default wells setup if none provided. Each well is specified by
            # its grid coordinates followed by its pressure. The first well
            # ([0, 0, 180]) is placed at the bottom-left corner with a pressure
            # of 180 units, representing an injection pressure. The second well
            # ([self.nx-1, self.ny-1, 120]), is located at the top-right
            # corner, and has a pressure of 120 units, possibly a lower
            # pressure or production scenario.
            self.wells = np.array(
                [[0, 0, 180], [self.nx - 1, self.ny - 1, 120]]
            )
        else:
            self.wells = np.array(wells)

        # Get well locations and set terms
        self.locs = self.wells[:, 1] * self.nx + self.wells[:, 0]

    @property
    def _set_well_terms(self):
        """Set well terms.

        Calculate well terms based on current permeability field, to be used in
        the simulation. Adjust well impacts using calculated terms.
        """
        wi = 2 * np.pi * self.perm_field[self.locs] * self.dz
        wi /= self.mu_w * np.log(0.208 * self.dx / self.rw)

        # Add wells
        self._add_wells_f = self.wells[:, 2] * wi
        self._add_wells_d = wi

    def solve(self, pressure, dt):
        """Construct & solve K-matrix for the simulation of pressure over time.

        Parameters
        ----------
        pressure : ndarray
            Current pressure state of the reservoir of size ``self.size``.

        dt : float
            Time step for the simulation (days).

        Returns
        -------
        pressure : ndarray
            Pressure state after applying the time step, of size ``self.size``.

        """

        # Mobility ratio without permeability
        phi = self.rho0 * (1 + self.c_f * (pressure - self.p0)) / self.mu_w

        # Compr. and right-hand side f
        compr = self._vol_phi_cf / dt
        f = compr * pressure

        # Pre-allocate diagonals.
        mn = np.zeros(self.size)
        m1 = np.zeros(self.size)
        d = compr
        p1 = np.zeros(self.size)
        pn = np.zeros(self.size)

        t1 = self.dx * self.perm_field[:-1] * self.perm_field[1:]
        t1 /= self.perm_field[:-1] + self.perm_field[1:]
        t1 *= (phi[:-1] + phi[1:]) / 2
        t1[self.nx - 1:: self.nx] = 0.0
        d[:-1] += t1
        d[1:] += t1
        m1[:-1] -= t1
        p1[1:] -= t1

        t2 = self.dx * self.perm_field[: -self.nx] * self.perm_field[self.nx:]
        t2 /= self.perm_field[: -self.nx] + self.perm_field[self.nx:]
        t2 *= (phi[: -self.nx] + phi[self.nx:]) / 2
        d[: -self.nx] += t2
        d[self.nx:] += t2
        mn[: -self.nx] -= t2
        pn[self.nx:] -= t2

        # Add wells.
        f[self.locs] += self._add_wells_f
        d[self.locs] += self._add_wells_d

        # Bring to sparse matrix
        if self.nx > 1:
            offsets = np.array([-self.nx, -1, 0, 1, self.nx])
            data = np.array([mn, m1, d, p1, pn])
        else:
            offsets = np.array([-1, 0, 1])
            data = np.array([mn + m1, d, p1 + pn])
        K = sp.sparse.dia_array((data, offsets), shape=(self.size, self.size))

        # Solve the system
        return sp.sparse.linalg.spsolve(K.tocsc(), f, use_umfpack=False)

    def __call__(self, perm_fields, dt=np.ones(10) * 0.0001, data=False):
        """Run simulator.

        Run the simulation across multiple time steps and possibly multiple
        permeability scenarios.

        Parameters
        ----------
        perm_fields : ndarray
            Permeability fields (mD) to simulate, either of dimension
            (ne, nx, ny), or of dimension (nx, ny).

        dt : ndarray, default: np.ones(10)*0.0001
            Time steps to use for simulation (days).

        data : {False, [ndarray, ndarray]}, default: False
            Specific indices [nx, ny] to output data for;
            if False, return all data

        Returns
        -------
        simulation : ndarray
            Simulation results over time for given permeability fields.

        """
        if perm_fields.ndim == 2:
            ne = 1
            perm_fields = [
                perm_fields,
            ]
        else:
            ne = perm_fields.shape[0]
        nt = dt.size + 1

        out = np.zeros((ne, nt, self.nx, self.ny))
        for n, perm_field in enumerate(perm_fields):

            self.perm_field = perm_field.ravel("F")
            self._set_well_terms

            pressure = np.ones((dt.size + 1, self.size)) * self.pres_ini
            for i, d in enumerate(dt):
                pressure[i + 1, :] = self.solve(pressure[i, :], d)
            out[n, ...] = pressure.reshape(
                (dt.size + 1, *self.shape), order="F"
            )

        if ne == 1:
            out = out[0, ...]

        if data:
            return out[..., data[0], data[1]]
        else:
            return out


class RandomPermeability:
    """Return random permeability fields with certain statistical props.


    Parameters
    ----------
    nx, ny : int
        Dimensions of the grid.
    perm_mean : float
        Mean permeability (mD).
    perm_min, perm_max : float
        Minimum and maximum values for permeability (mD).
    length : tuple of two floats, default: (10.0, 10.0)
        Length scales for the correlation of permeability (-).
    theta : float, default: 45.0
        Rotation angle for the anisotropy in the permeability field (°).
    variance : float, default: 1.0
        Variance scale for the permeability.
    dtype : str, default: 'float32'
        Data type for computations, for precision and performance tuning.

    """

    def __init__(
        self,
        nx,
        ny,
        perm_mean,
        perm_min,
        perm_max,
        length=(10.0, 10.0),
        theta=45.0,
        variance=1.0,
        dtype="float32",
    ):
        """Initialize parameters for generating random permeability fields."""
        self.nx, self.ny = nx, ny  # Grid dimensions
        self.nc = nx * ny  # Total number of cells
        self.perm_mean = perm_mean  # Permeability statistics
        self.perm_min, self.perm_max = perm_min, perm_max
        self.length, self.theta = length, theta  # Anisotropy parameters
        self.variance = variance  # Variance
        self.dtype = dtype  # Data type

    @property
    def cov(self):
        """Covariance matrix

        Lazy-loaded covariance matrix, calculated based on anisotropy and
        statistical parameters.
        """
        if not hasattr(self, "_cov"):
            self._cov = utils.gaussian_covariance(
                nx=self.nx,
                ny=self.ny,
                length=self.length,
                theta=self.theta,
                variance=self.variance,
                dtype=self.dtype,
            )
        return self._cov

    @property
    def lcho(self):
        """Lower Cholesky decomposition

        Lower Cholesky decomposition of the covariance matrix, used for
        generating random fields.
        """
        if not hasattr(self, "_lcho"):
            self._lcho = sp.linalg.cholesky(self.cov, lower=True)
        return self._lcho

    def __call__(
        self, n, perm_mean=None, perm_min=None, perm_max=None, random=None
    ):
        """Generate n random permeability fields

        Generate n random permeability fields using the specified statistical
        parameters and random seed.

        Parameters
        ----------
        n : int
            Number of fields to generate
        perm_mean : {float, None}, default: None
            Mean permeability to override the initialized value (mD).
        perm_min, perm_max : {float, None}, default: None
            Min and max permeability values to clip the fields (mD).
        random : {None, int,  np.random.Generator}, default: None
            Seed or random generator for reproducibility.

        Returns
        -------
        perm_fields : ndarray
            An array of generated permeability fields.

        """

        if perm_mean is None:
            perm_mean = self.perm_mean
        if perm_min is None:
            perm_min = self.perm_min
        if perm_max is None:
            perm_max = self.perm_max

        # Initialize fields with mean permeability
        out = np.full((n, self.nx, self.ny), perm_mean, order="F")
        for i in range(n):
            z = utils.rng(random).normal(size=self.nc)  # Random numbers
            # Apply the Cholesky transform
            out[i, ...] += (self.lcho @ z).reshape(
                (self.nx, self.ny), order="F"
            )

        # Clip the results to stay within specified bounds
        return out.clip(perm_min, perm_max)
