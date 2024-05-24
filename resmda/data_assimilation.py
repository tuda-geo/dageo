# Copyright 2024 Dieter Werthmüller, Gabriel Serrao Seabra
#
# This file is part of resmda.
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

from resmda.utils import rng

__all__ = ['esmda']


def __dir__():
    return __all__


def esmda(model_prior, forward, data_obs, sigma, alphas=4, data_prior=None,
          callback_post=None, return_post_data=False, random=None,
          enable_localization=False, localization_matrix=None):
    """ESMDA algorithm (Emerick and Reynolds, 2013) with optional localization.


    Parameters
    ----------
    model_prior : ndarray
        Prior models of dimension ``(ne, ...)``, where ``ne`` is the number of
        ensembles.
    forward : callable
        Forward model that takes an ndarray of the shape of the prior models
        ``(ne, ...)``, and returns a ndarray of the shape of the prior data
        ``(ne, nd)``; ``ne`` is the number of ensembles, ``nd`` the number of
        data.
    data_obs : ndarray
        Observed data of shape ``(nd)``.
    sigma : {float, ndarray}
        Standard deviation(s) of the observation noise.
    alphas : {int, ndarray}, default: 4
        Inflation factors for ES-MDA.
    data_prior : ndarray, default: None
        Prior data ensemble, of shape (ne, nd).
    callback_post : function, default: None
        Function to be executed after each ESMDA iteration.
    return_post_data : bool, default: False
        If true, returns data
    random : {None, int,  np.random.Generator}, default: None
        Seed or random generator for reproducibility.
    enable_localization : bool, default: False
        If True, apply localization to the Kalman gain matrix.
    localization_matrix : ???, default: None
        Localization matrix.


    Returns
    -------
    model_post : ndarray
        Posterior model ensemble.
    data_post : ndarray, only returned if ``return_post_data=True``
        Posterior simulated data ensemble.

    """
    # Get number of ensembles and time steps
    ne = model_prior.shape[0]
    nd = data_obs.size

    # Expand sigma if float
    if np.asarray(sigma).size == 1:
        sigma = np.zeros(nd) + sigma

    # Get alphas
    if isinstance(alphas, int):
        alphas = np.zeros(alphas) + alphas
    else:
        alphas = np.asarray(alphas)

    # Copy prior as start of post (output)
    model_post = model_prior.copy()

    # Loop over alphas
    for i, alpha in enumerate(alphas):
        print(f"ESMDA step {i+1}; α={alpha}")

        # == Step (a) of Emerick & Reynolds, 2013 ==
        # Run the ensemble from time zero.

        # Get data
        if i > 0 or data_prior is None:
            data_prior = forward(model_post)

        # == Step (b) of Emerick & Reynolds, 2013 ==
        # For each ensemble member, perturb the observation vector using
        # d_uc = d_obs + sqrt(α_i) * C_D^0.5 z_d; z_d ~ N(0, I_N_d)

        zd = rng(random).normal(size=(ne, nd))
        data_pert = data_obs + np.sqrt(alpha) * sigma * zd

        # == Step (c) of Emerick & Reynolds, 2013 ==
        # Update the ensemble using Eq. (3) with C_D replaced by α_i * C_D

        # Compute the (co-)variances
        # Note: The factor (ne-1) is part of the covariances CMD and CDD,
        # wikipedia.org/wiki/Covariance#Calculating_the_sample_covariance
        # but factored out of CMD(CDD+αCD)^-1 to be in αCD.
        cmodel = model_post - model_post.mean(axis=0)
        cdata = data_prior - data_prior.mean(axis=0)
        CMD = np.moveaxis(cmodel, 0, -1) @ cdata
        CDD = cdata.T @ cdata
        CD = np.diag(alpha * (ne - 1) * sigma**2)

        # Compute inverse of C
        # C is a real-symmetric positive-definite matrix.
        # If issues arise in real-world problems, try using
        # - a subspace inversions with Woodbury matrix identity, or
        # - Moore-Penrose via np.linalg.pinv, sp.linalg.pinv, spp.linalg.pinvh.
        Cinv = np.linalg.inv(CDD + CD)

        # Calculate the Kalman gain
        K = CMD@Cinv

        # TODO: Implement localization
        if enable_localization and localization_matrix is not None:
            localization_matrix = localization_matrix
            K = K.reshape(-1, nd)
            K = K * localization_matrix
            # Reshape back to original shape
            K = K.reshape(model_prior.shape[1], model_prior.shape[2], nd)

        # Update the ensemble parameters
        model_post += np.moveaxis(K @ (data_pert - data_prior).T, -1, 0)

        # Apply model parameter bounds.
        if callback_post:
            callback_post(model_post)

    # Return posterior model and corresponding data
    return model_post, forward(model_post)


def convert_positions_to_indices(positions, grid_dimensions):
    """Convert 2D grid positions to 1D indices assuming zero-indexed positions.

    Parameters
    ----------
    positions : ndarray
        Array of (x, y) positions in the grid.
    grid_dimensions : tuple
        Dimensions of the grid (nx, ny).

    Returns
    -------
    indices : ndarray
        Array of indices corresponding to the positions in a flattened array.

    """
    nx, ny = grid_dimensions
    # Ensure the positions are zero-indexed and correctly calculated for
    # row-major order
    return positions[:, 1] * nx + positions[:, 0]


def build_localization_matrix(cov_matrix, data_positions, grid_dimensions):
    """Build a localization matrix

    Build a localization matrix from a full covariance matrix based on specific
    data positions.

    Parameters
    ----------
    cov_matrix : ndarray
        The full (nx*ny) x (nx*ny) covariance matrix.
    data_positions : ndarray
        Positions in the grid for each data point (e.g., wells), zero-indexed.
    grid_dimensions : tuple
        Dimensions of the grid (nx, ny).

    Returns
    -------
    loc_matrix : ndarray
        Localization matrix of shape (nx*ny, number of data positions).

    """
    # Convert 2D positions of data points to 1D indices suitable for accessing
    # the covariance matrix
    indices = convert_positions_to_indices(
            data_positions, grid_dimensions).astype(int)
    # Extract the columns from the covariance matrix corresponding to each data
    # point's position
    return cov_matrix[:, indices]
