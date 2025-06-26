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

import math

import numpy as np
import torch

__all__ = [
    "Geertsma",
    "GeertsmaFullGrid",
    "TorchGeertsma",
    "TorchGeertsmaFullGrid",
]


def __dir__():
    return __all__


class Geertsma:
    """Geertsma's analytical solution for subsidence due to reservoir
    compaction.

    This implements Geertsma's model for surface subsidence due to reservoir
    compaction, providing a simple forward model that can be used with the
    ESMDA algorithm. The model calculates the vertical displacement
    (subsidence) at specified observation points on the surface, caused by
    pressure changes in a disc-shaped reservoir.

    Background Theory:
    -----------------
    When pressure in a reservoir decreases (e.g., during hydrocarbon
    production), the effective stress on the reservoir rock increases. This
    increased stress leads to compaction of the reservoir. The compaction then
    propagates to the surface, causing subsidence.

    Geertsma (1973) provided an analytical solution to relate reservoir
    pressure changes to surface displacement using linear poroelasticity
    theory. This model has been widely used in the petroleum and geothermal
    industries to predict surface subsidence due to fluid extraction.

    For a uniform, disc-shaped reservoir, surface subsidence can be calculated
    using Geertsma's closed-form solution, which is implemented in this class.
    For more complex, heterogeneous reservoirs, see the GeertsmaFullGrid class.

    Mathematical Formulation:
    ------------------------
    The model follows these steps:
    1. Calculate pressure change in the reservoir (ΔP)
    2. Compute reservoir compaction: Δh = cm * h * ΔP
       (where cm is the compaction coefficient, h is reservoir thickness)
    3. Apply Geertsma's solution to propagate this compaction to the surface,
       accounting for the depth and lateral position of observation points

    The Geertsma solution is based on a nucleus-of-strain approach and assumes:
    - Linear poroelasticity
    - Homogeneous, isotropic elastic properties
    - Reservoir compaction proportional to pressure change
    - Semi-infinite, elastic half-space
    - No horizontal displacement at the surface

    Parameters
    ----------
    depth : float
        Depth to the center of the reservoir (m).
    radius : float, default: 1000.0
        Radius of the disc-shaped reservoir (m).
    thickness : float, default: 50.0
        Thickness of the reservoir (m).
    cm : float, default: 1.0e-10
        Uniaxial compaction coefficient (1/Pa).
    nu : float, default: 0.25
        Poisson's ratio (-).
    p0 : float, default: 0.0
        Initial reservoir pressure (Pa).
    obs_points : {ndarray, None}, default: None
        Array of shape (npoints, 2) with x,y-coordinates of observation points.
        If None, a default grid of points is created.
    npoints : int, default: 20
        Number of observation points in each direction when creating default
        grid.
    obs_range : float, default: 5000.0
        Extent of default observation grid (m).

    References
    ----------
    Geertsma, J. (1973). Land subsidence above compacting oil and gas
    reservoirs. Journal of Petroleum Technology, 25(06), 734-744.
    """

    def __init__(
        self,
        depth,
        radius=1000.0,
        thickness=50.0,
        cm=1.0e-10,
        nu=0.25,
        p0=0.0,
        obs_points=None,
        npoints=20,
        obs_range=5000.0,
    ):
        """Initialize a Geertsma model instance."""
        self.depth = depth
        self.radius = radius
        self.thickness = thickness
        self.cm = cm
        self.nu = nu
        self.p0 = p0

        # Create observation points if not provided
        if obs_points is None:
            x = np.linspace(-obs_range, obs_range, npoints)
            y = np.linspace(-obs_range, obs_range, npoints)
            X, Y = np.meshgrid(x, y)
            self.obs_points = np.column_stack((X.flatten(), Y.flatten()))
        else:
            self.obs_points = np.asarray(obs_points)

        # Store the number of observation points
        self.nobs = self.obs_points.shape[0]

    def _nucleus_solution(self, x, y, depth, nu):
        """Calculate nucleus-of-strain solution.

        Parameters
        ----------
        x, y : ndarray
            Horizontal distances from the nucleus (m).
        depth : float
            Depth of the nucleus (m).
        nu : float
            Poisson's ratio (-).

        Returns
        -------
        uz : ndarray
            Vertical displacement at the observation points (m).
        """
        r = np.sqrt(x**2 + y**2)
        R = np.sqrt(r**2 + depth**2)

        # Geertsma's solution for vertical displacement
        uz = (1 - 2 * nu) * depth / (R**3)
        return uz

    def _disc_solution(self, pressure_field):
        """Calculate the displacement for a disc-shaped reservoir.

        Parameters
        ----------
        pressure_field : ndarray
            Pressure field inside the reservoir (Pa).

        Returns
        -------
        uz : ndarray
            Vertical displacement at each observation point (m).
        """
        # Prepare result array
        uz = np.zeros(self.nobs)

        # For a disc-shaped reservoir, we use the nucleus solution with
        # numerical integration over the reservoir volume
        x_obs = self.obs_points[:, 0]
        y_obs = self.obs_points[:, 1]

        # Get average pressure change
        delta_p = np.mean(pressure_field) - self.p0

        # Calculate compaction for the entire reservoir volume
        # For a uniform disc-shaped reservoir, we can use a closed-form
        # solution
        volume = np.pi * self.radius**2 * self.thickness
        compaction = self.cm * delta_p * volume

        # Calculate the distance from each observation point to the center of
        # the reservoir
        r = np.sqrt(x_obs**2 + y_obs**2)

        # Apply the Geertsma solution for a disc
        # For r > 0, we use the exact solution
        mask = r > 0
        if np.any(mask):
            r_mask = r[mask]
            uz[mask] = (
                compaction
                * (3 - 4 * self.nu)
                * self.depth
                / (2 * np.pi)
                * (
                    (r_mask**2 + self.depth**2) ** (-3 / 2)
                    - (r_mask**2 + self.depth**2 + self.radius**2) ** (-3 / 2)
                )
            )

        # For r = 0 (observation point directly above reservoir center)
        # We need to handle the special case to avoid division by zero
        mask_center = r == 0
        if np.any(mask_center):
            uz[mask_center] = (
                compaction
                * (3 - 4 * self.nu)
                * self.depth
                / (2 * np.pi)
                * (
                    (self.depth**2) ** (-3 / 2)
                    - (self.depth**2 + self.radius**2) ** (-3 / 2)
                )
            )

        return uz

    def __call__(self, pressure_fields, time_steps=None):
        """Run Geertsma model to calculate surface subsidence.

        Parameters
        ----------
        pressure_fields : ndarray
            Pressure fields to simulate, either of dimension (ne, nx, ny),
            or of dimension (nx, ny). These represent reservoir pressure (Pa).

        time_steps : {ndarray, None}, default: None
            For compatibility with the Simulator class interface. Not used in
            this implementation as the Geertsma model provides a static
            solution.

        Returns
        -------
        subsidence : ndarray
            Subsidence at observation points for given pressure fields.
        """
        # Handle single or multiple fields
        if pressure_fields.ndim == 2:
            ne = 1
            pressure_fields = np.expand_dims(pressure_fields, 0)
        else:
            ne = pressure_fields.shape[0]

        # Calculate subsidence for each field
        subsidence = np.zeros((ne, self.nobs))
        for i, pressure_field in enumerate(pressure_fields):
            subsidence[i, :] = self._disc_solution(pressure_field)

        # Return results
        if ne == 1:
            return subsidence[0, :]
        else:
            return subsidence


class GeertsmaFullGrid:
    """Full-grid implementation of Geertsma's solution for subsidence.

    This class extends the basic Geertsma model to account for heterogeneous
    pressure changes across a gridded reservoir. Each cell contributes to the
    total subsidence according to its pressure change, size, and position.

    Unlike the basic Geertsma model which assumes a uniform disc-shaped
    reservoir, this implementation:

    1. Handles arbitrary reservoir geometry defined by a grid
    2. Accounts for spatially varying pressure changes
    3. Uses matrix operations for efficient calculation with multiple models
    4. Is particularly well-suited for ESMDA or other ensemble-based methods

    The implementation uses a nucleus-of-strain approach, where each grid cell
    is treated as a nucleus that contributes to surface subsidence. These
    contributions are pre-computed in an influence matrix for efficiency.

    When subsidence calculations must be performed for many different pressure
    scenarios (as in ensemble methods), this approach is much more
    computationally efficient than recalculating from first principles each
    time.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y directions.
    depth : float
        Depth to the top of the reservoir (m).
    dx, dy : float, default: 100.0, 100.0
        Cell dimensions in x and y directions (m).
    thickness : float, default: 50.0
        Thickness of the reservoir (m).
    cm : float, default: 1.0e-10
        Uniaxial compaction coefficient (1/Pa).
    nu : float, default: 0.25
        Poisson's ratio (-).
    p0 : float, default: 0.0
        Initial reservoir pressure (Pa).
    obs_points : {ndarray, None}, default: None
        Array of shape (npoints, 2) with x,y-coordinates of observation points.
        If None, a default grid of points is created.
    npoints : int, default: 20
        Number of observation points in each direction when creating default
        grid.
    obs_range : float, default: None
        Extent of default observation grid (m). If None, it's set to 2 times
        the reservoir extent.
    """

    def __init__(
        self,
        nx,
        ny,
        depth,
        dx=100.0,
        dy=100.0,
        thickness=50.0,
        cm=1.0e-10,
        nu=0.25,
        p0=0.0,
        obs_points=None,
        npoints=20,
        obs_range=None,
    ):
        """Initialize a GeertsmaFullGrid instance."""
        self.nx = nx
        self.ny = ny
        self.depth = depth
        self.dx = dx
        self.dy = dy
        self.thickness = thickness
        self.cm = cm
        self.nu = nu
        self.p0 = p0

        # Calculate reservoir extent
        self.x_extent = nx * dx
        self.y_extent = ny * dy

        # Create observation points if not provided
        if obs_points is None:
            if obs_range is None:
                # Default to a range that's 2x the reservoir extent
                obs_range = max(self.x_extent, self.y_extent)

            x = np.linspace(-obs_range, obs_range, npoints)
            y = np.linspace(-obs_range, obs_range, npoints)
            X, Y = np.meshgrid(x, y)
            self.obs_points = np.column_stack((X.flatten(), Y.flatten()))
        else:
            self.obs_points = np.asarray(obs_points)

        # Store the number of observation points
        self.nobs = self.obs_points.shape[0]

        # Create cell centers for the reservoir grid
        x_centers = np.linspace(dx / 2, self.x_extent - dx / 2, nx)
        y_centers = np.linspace(dy / 2, self.y_extent - dy / 2, ny)
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers)

        # Center the reservoir grid around the origin
        X_centers -= self.x_extent / 2
        Y_centers -= self.y_extent / 2

        self.cell_centers = np.column_stack(
            (X_centers.flatten(), Y_centers.flatten())
        )
        self.cell_volume = dx * dy * thickness

    def _calculate_influence(self, x_obs, y_obs, x_cell, y_cell):
        """Calculate influence coefficient for a cell on an observation point.

        Parameters
        ----------
        x_obs, y_obs : float
            Coordinates of the observation point (m).
        x_cell, y_cell : float
            Coordinates of the cell center (m).

        Returns
        -------
        influence : float
            Influence coefficient for vertical displacement.
        """
        # Calculate horizontal distance from cell to observation point
        dx = x_obs - x_cell
        dy = y_obs - y_cell
        r = np.sqrt(dx**2 + dy**2)

        # Use Geertsma's nucleus-of-strain solution
        if r == 0:  # Observation point directly above cell
            influence = (
                (3 - 4 * self.nu) * self.depth / (2 * np.pi * self.depth**3)
            )
        else:
            R = np.sqrt(r**2 + self.depth**2)
            influence = (3 - 4 * self.nu) * self.depth / (2 * np.pi * R**3)

        return influence

    def _build_influence_matrix(self):
        """Build the influence matrix for all cells and observation points.

        The influence matrix is a key efficiency component of the model. It
        represents how each reservoir cell influences subsidence at each
        observation point.

        For a grid-based reservoir with many cells, calculating the
        contribution of each cell to each observation point can be
        computationally expensive. By pre-computing these influence
        coefficients once in a matrix, subsequent subsidence calculations
        for different pressure fields become a simple matrix-vector
        multiplication.

        The influence matrix has dimensions (nobs, nx*ny) where:
        - Each row represents an observation point on the surface
        - Each column represents a cell in the reservoir grid
        - Each matrix element A[i,j] represents how much a unit pressure change
          in cell j contributes to subsidence at observation point i

        The influence coefficients are derived from Geertsma's
        nucleus-of-strain solution, which depends on:
        - The horizontal distance between observation point and cell center
        - The depth of the reservoir
        - The Poisson's ratio of the material
        - The compaction coefficient
        - The volume of the cell

        This method calculates this matrix once, making later subsidence
        calculations much more efficient, particularly when processing
        multiple pressure fields in ensemble-based methods like ESMDA.

        Returns
        -------
        influence_matrix : ndarray
            Matrix of shape (nobs, nx*ny) containing influence coefficients.
            When multiplied by a vector of pressure changes, it produces the
            resulting subsidence at each observation point.
        """
        # Initialize influence matrix
        influence_matrix = np.zeros((self.nobs, self.nx * self.ny))

        # Calculate influence for each observation point and cell combination
        for i, (x_obs, y_obs) in enumerate(self.obs_points):
            for j, (x_cell, y_cell) in enumerate(self.cell_centers):
                influence_matrix[i, j] = self._calculate_influence(
                    x_obs, y_obs, x_cell, y_cell
                )

        # Scale by cell volume and compaction coefficient
        # This converts from pressure change to subsidence
        return influence_matrix * self.cell_volume * self.cm

    def __call__(self, pressure_fields, time_steps=None):
        """Run Geertsma model to calculate surface subsidence.

        This method calculates surface subsidence caused by pressure changes
        in the reservoir. For grid-based reservoirs, it efficiently uses the
        pre-computed influence matrix to transform pressure changes into
        surface displacements.

        The calculation follows these steps:
        1. Ensure the influence matrix is built (only done once)
        2. For each pressure field:
           a. Calculate the pressure change at each cell (current pressure -
              initial pressure)
           b. Multiply the pressure changes by the influence matrix to get
              subsidence
           c. Apply a negative sign to follow the convention that subsidence
              is downward

        The matrix multiplication approach is much more efficient than
        recalculating
        the contribution of each cell for every pressure field, especially when
        working with multiple ensemble members in data assimilation.

        Parameters
        ----------
        pressure_fields : ndarray
            Pressure fields to simulate, either of dimension (ne, nx, ny),
            or of dimension (nx, ny). These represent reservoir pressure (Pa).

        time_steps : {ndarray, None}, default: None
            For compatibility with the Simulator class interface. Not used in
            this implementation as the Geertsma model provides a static
            solution.

        Returns
        -------
        subsidence : ndarray
            Subsidence at observation points for given pressure fields.
            Shape is (ne, nobs) for multiple fields or (nobs,) for a single
            field.
            Negative values indicate downward movement (subsidence).
        """
        # Handle single or multiple fields
        if pressure_fields.ndim == 2:
            ne = 1
            pressure_fields = np.expand_dims(pressure_fields, 0)
        else:
            ne = pressure_fields.shape[0]

        # Calculate influence matrix only once for efficiency
        if not hasattr(self, "_influence_matrix"):
            self._influence_matrix = self._build_influence_matrix()

        # Calculate subsidence for each field
        subsidence = np.zeros((ne, self.nobs))
        for i, pressure_field in enumerate(pressure_fields):
            # Calculate pressure change for each cell relative to initial
            # pressure
            delta_p = pressure_field.flatten() - self.p0

            # Calculate subsidence using the influence matrix
            # This efficiently computes how each cell's pressure change
            # contributes
            # to subsidence at all observation points
            # Negative sign convention: subsidence is downward displacement
            subsidence[i, :] = -self._influence_matrix @ delta_p

        # Return results, keeping appropriate dimensions
        if ne == 1:
            return subsidence[0, :]
        else:
            return subsidence


class TorchGeertsma:
    """
    Torch differentiable implementation of Geertsma's model for subsidence.

    This variation uses PyTorch operations for differentiability and is
    intended for
    gradient-based inversion methods (e.g., RML). It computes the predicted
    vertical
    subsidence at the reservoir center (r=0) for a constant pressure drop.

    Parameters
    ----------
    depth : float or torch scalar
        Depth to the reservoir (m).
    radius : float or torch scalar
        Radius of the disc-shaped reservoir (m).
    thickness : float or torch scalar
        Thickness of the reservoir (m).
    cm : float or torch scalar
        Uniaxial compaction coefficient (1/Pa).
    nu : float or torch scalar
        Poisson's ratio.
    p0 : float
        Initial reservoir pressure (Pa).
    p : float
        Current reservoir pressure (Pa).
    """

    def __init__(
        self,
        depth,
        radius=1000.0,
        thickness=50.0,
        cm=1.0e-10,
        nu=0.25,
        p0=20e6,
        p=15e6,
    ):
        self.depth = (
            torch.tensor(depth, dtype=torch.float32)
            if not torch.is_tensor(depth)
            else depth
        )
        self.radius = (
            torch.tensor(radius, dtype=torch.float32)
            if not torch.is_tensor(radius)
            else radius
        )
        self.thickness = (
            torch.tensor(thickness, dtype=torch.float32)
            if not torch.is_tensor(thickness)
            else thickness
        )
        self.cm = (
            torch.tensor(cm, dtype=torch.float32)
            if not torch.is_tensor(cm)
            else cm
        )
        self.nu = (
            torch.tensor(nu, dtype=torch.float32)
            if not torch.is_tensor(nu)
            else nu
        )
        self.p0 = p0
        self.p = p  # current pressure (assumed constant)

    def __call__(self):
        """
        Computes the predicted vertical subsidence at the reservoir center
        (r=0).

        Returns
        -------
        subsidence : torch tensor of shape (1,)
            Predicted subsidence.
        """
        # Compute pressure change
        delta_p = (
            self.p - self.p0
        )  # expected to be negative if pressure dropped

        # Compute reservoir volume for a disc
        volume = math.pi * self.radius**2 * self.thickness

        # Compute compaction: compaction = cm * delta_p * volume
        compaction = self.cm * delta_p * volume

        # At r=0, the subsidence is given by:
        # u_z = (3 - 4*nu)*depth/(2*pi) * compaction *
        #       [ (depth^2)^(-3/2) - (depth^2+radius^2)^(-3/2) ]
        term_center = (self.depth**2) ** (-1.5) - (
            self.depth**2 + self.radius**2
        ) ** (-1.5)
        constant = (3 - 4 * self.nu) * self.depth / (2 * math.pi)

        subsidence = constant * compaction * term_center
        return subsidence.unsqueeze(0)


class TorchGeertsmaFullGrid:
    """
    Torch differentiable full-grid implementation of Geertsma's model for
    subsidence.

    This variation uses PyTorch operations to compute the subsidence at a set
    of observation points
    for a grid-based reservoir. It is intended for gradient-based inversion
    methods (e.g., RML).
    The subsidence is computed from a uniform pressure field with a constant
    pressure drop.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    depth : float or torch scalar
        Depth to the reservoir (m).
    dx : float
        Cell dimension in the x-direction (m).
    dy : float
        Cell dimension in the y-direction (m).
    thickness : float or torch scalar
        Thickness of the reservoir (m).
    cm : float or torch scalar
        Uniaxial compaction coefficient (1/Pa).
    nu : float or torch scalar
        Poisson's ratio.
    p0 : float
        Initial reservoir pressure (Pa).
    p : float
        Current reservoir pressure (Pa).
    obs_points : array-like or torch tensor, optional
        Coordinates of observation points, shape (n_obs, 2). If None, a
        default grid is created.
    npoints : int, default: 20
        Number of observation points in each direction for the default grid.
    obs_range : float, optional
        Extent of the observation grid in each direction. If None, set to
        2 * max(nx*dx, ny*dy).
    """

    def __init__(
        self,
        nx,
        ny,
        depth,
        dx=100.0,
        dy=100.0,
        thickness=50.0,
        cm=1.0e-10,
        nu=0.25,
        p0=0.0,
        p=15e6,
        obs_points=None,
        npoints=20,
        obs_range=None,
    ):
        self.nx = nx
        self.ny = ny
        self.depth = (
            torch.tensor(depth, dtype=torch.float32)
            if not torch.is_tensor(depth)
            else depth
        )
        self.dx = dx
        self.dy = dy
        self.thickness = (
            torch.tensor(thickness, dtype=torch.float32)
            if not torch.is_tensor(thickness)
            else thickness
        )
        self.cm = (
            torch.tensor(cm, dtype=torch.float32)
            if not torch.is_tensor(cm)
            else cm
        )
        self.nu = (
            torch.tensor(nu, dtype=torch.float32)
            if not torch.is_tensor(nu)
            else nu
        )
        self.p0 = p0
        self.p = p

        # Define cell centers for the reservoir grid (shape: [nx*ny, 2])
        x_centers = (
            torch.linspace(dx / 2, nx * dx - dx / 2, nx) - (nx * dx) / 2
        )
        y_centers = (
            torch.linspace(dy / 2, ny * dy - dy / 2, ny) - (ny * dy) / 2
        )
        Xc, Yc = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self.cell_centers = torch.stack(
            [Xc.flatten(), Yc.flatten()], dim=1
        )  # shape (nx*ny, 2)

        # Cell volume
        self.cell_volume = dx * dy * thickness

        # Define observation points
        if obs_points is None:
            if obs_range is None:
                obs_range = 2 * max(nx * dx, ny * dy)
            x_obs = torch.linspace(-obs_range, obs_range, npoints)
            y_obs = torch.linspace(-obs_range, obs_range, npoints)
            Xo, Yo = torch.meshgrid(x_obs, y_obs, indexing="ij")
            self.obs_points = torch.stack(
                [Xo.flatten(), Yo.flatten()], dim=1
            )  # shape (n_obs, 2)
        else:
            if not torch.is_tensor(obs_points):
                self.obs_points = torch.tensor(obs_points, dtype=torch.float32)
            else:
                self.obs_points = obs_points

    def __call__(self):
        """
        Computes the subsidence at observation points using the
        differentiable full-grid Geertsma model.

        Returns
        -------
        subsidence : torch tensor of shape (n_obs,)
            Predicted subsidence at each observation point (negative
            indicates downward displacement).
        """
        n_cells = self.cell_centers.shape[0]

        # Compute differences between observation points and cell centers
        # diff shape: (n_obs, n_cells, 2)
        diff = self.obs_points.unsqueeze(1) - self.cell_centers.unsqueeze(0)
        r = torch.norm(
            diff, dim=2
        )  # horizontal distances, shape: (n_obs, n_cells)

        # Base term: (3 - 4*nu)*depth/(2*pi)
        base = (3 - 4 * self.nu) * self.depth / (2 * math.pi)

        # Compute denominator: if r==0 use depth^3, else (r^2 + depth^2)^(3/2)
        denom = torch.where(
            r == 0, self.depth**3, (r**2 + self.depth**2) ** 1.5
        )
        influence = base / denom  # shape: (n_obs, n_cells)

        # Scale by cell volume and compaction coefficient
        influence_matrix = (
            influence * self.cell_volume * self.cm
        )  # shape: (n_obs, n_cells)

        # Compute pressure change for each cell (assume uniform pressure field)
        delta_p = self.p - self.p0  # scalar
        delta_p_vec = delta_p * torch.ones(n_cells, dtype=torch.float32)

        # Predicted subsidence at observation points (negative sign for
        # downward displacement)
        subsidence = -torch.matmul(
            influence_matrix, delta_p_vec
        )  # shape: (n_obs,)
        return subsidence
