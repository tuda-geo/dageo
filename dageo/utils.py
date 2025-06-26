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

from datetime import datetime

import numpy as np
import scipy as sp
from scooby import Report as ScoobyReport

try:
    from dageo.version import version as __version__
except ImportError:
    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")


__all__ = ["gaussian_covariance", "localization_matrix", "Report", "rng"]


def __dir__():
    return __all__


def gaussian_covariance(nx, ny, length, theta, variance, dtype="float32"):
    """Return covariance matrix with Gaussian properties

    Generate covariance matrix based on grid size, anisotropy, and statistical
    parameters.


    Parameters
    ----------
    nx, ny : int
        Dimensions of the grid.
    length : float
        Length scales for the correlation of the property.
    theta : float
        Rotation angle for the anisotropy in the property field.
    variance : float
        Variance of the property.
    dtype : str, default: 'float32'
        Data type for computations.


    Returns
    -------
    cov : ndarray
        Covariance matrix for the property field.

    """
    nc = nx * ny  # Total number of cells
    # Precompute cosine and sine of the rotation angle
    cost, sint = np.cos(theta), np.sin(theta)

    # 1. Fill the first row of the covariance matrix
    tmp1 = np.zeros([nx, nc], dtype=dtype)
    for i in range(nx):
        tmp1[i, 0] = 1.0  # Set diagonal
        for j in range(i + 1, nc):
            # Distance in the x and y directions
            d0 = (j % nx) - i
            d1 = j // nx
            # Rotate coordinates
            rot0 = cost * d0 - sint * d1
            rot1 = sint * d0 + cost * d1
            # Calculate the scaled distance
            hl = np.sqrt((rot0 / length[0]) ** 2 + (rot1 / length[1]) ** 2)

            # Sphere formula for covariance, modified for anisotropy
            if variance:  # Non-zero variance scale
                if hl <= 1:
                    tmp1[i, j - i] = variance * (1 - 1.5 * hl + hl**3 / 2)

            else:  # Gaspari-Cohn function for smoothness
                if hl < 1:
                    tmp1[i, j - i] = (
                        -(hl**5) / 4
                        + (hl**4) / 2
                        + (hl**3) * 5 / 8
                        - (hl**2) * 5 / 3
                        + 1
                    )
                elif hl >= 1 and hl < 2:
                    tmp1[i, j - i] = (
                        (hl**5) / 12
                        - (hl**4) / 2
                        + (hl**3) * 5 / 8
                        + (hl**2) * 5 / 3
                        - hl * 5
                        + 4
                        - (1 / hl) * 2 / 3
                    )

    # 2. Get the indices of the non-zero columns
    ind = np.where(tmp1.sum(axis=0))[0]

    # 3. Expand the non-zero colums ny-times
    tmp2 = np.zeros([nc, ind.size], dtype=dtype)
    for i, j in enumerate(ind):
        n = j // nx
        tmp2[: nc - n * nx, i] = np.tile(tmp1[:, j], ny - n)

    # 4. Construct array through sparse diagonal array
    cov = sp.sparse.dia_array((tmp2.T, -ind), shape=(nc, nc))
    return cov.toarray()


def localization_matrix(covariance, data_positions, shape, cov_type="lower"):
    """Return a localization matrix

    Build a localization matrix from a full covariance matrix based on specific
    data positions.

    Parameters
    ----------
    covariance : ndarray
        The lower triangular covariance matrix ``(nx*ny, nx*ny)``.
    data_positions : ndarray
        Positions in the grid for each data point (e.g., wells), zero-indexed,
        of size ``(nd, 2)``.
    shape : tuple
        Dimensions of the grid ``(nx, ny)``.
    cov_type : {'lower', 'upper', 'full'}; default: 'lower'
        Matrix type of the provided covariance matrix.

    Returns
    -------
    loc_matrix : ndarray
        Localization matrix of shape ``(nx, ny, nd)``.

    """
    # Convert 2D positions of data points to 1D indices suitable for accessing
    # the covariance matrix
    indices = np.array(
        data_positions[:, 1] * shape[0] + data_positions[:, 0],
    ).astype(int)

    # Extract the corresponding columns from the covariance matrix
    loc_mat = covariance[:, indices]
    if cov_type == "lower":
        loc_mat += np.tril(covariance, -1).T[:, indices]
    elif cov_type == "upper":
        loc_mat += np.triu(covariance, 1).T[:, indices]

    # Reshape and return
    return loc_mat.reshape((*shape, -1), order="F")


def rng(random=None):
    """Module-wide Random Number Generator.

    Instantiate a random number generator.


    Parameters
    ----------
    random : {None, int,  np.random.Generator}, default: None
        - If ``None`` it returns a :func:`numpy.random.default_rng()` instance
          instantiated on a module level.
        - If ``int``, it returns a newly created
          :func:`numpy.random.default_rng()` instance, instantiated with
          ``int`` as seed.
        - If it is already a :class:`numpy.random.Generator` instance, it
          simply returns it.


    Returns
    -------
    rng : random number generator
        A :class:`numpy.random.Generator` instance.

    """
    if isinstance(random, int):
        return np.random.default_rng(random)
    elif isinstance(random, np.random.Generator):
        return random
    else:
        if not hasattr(rng, "_rng"):
            rng._rng = np.random.default_rng()
        return rng._rng


class Report(ScoobyReport):
    """Print a Scooby report.

    For more info, consult https://github.com/banesullivan/scooby/.
    """

    def __init__(self, **kwargs):
        """Initiate a scooby.Report instance."""
        kwargs = {"ncol": 3, **kwargs}
        kwargs["core"] = ["dageo", "numpy", "scipy"]
        kwargs["optional"] = ["matplotlib", "IPython"]
        super().__init__(**kwargs)
