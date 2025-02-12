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
import warnings

import numpy as np

from dageo import utils

__all__ = ['esmda']


def __dir__():
    return __all__


def esmda(model_prior, forward, data_obs, sigma, alphas=4, data_prior=None,
          localization_matrix=None, callback_post=None, return_post_data=True,
          return_steps=False, random=None):
    """ESMDA algorithm ([EmRe13]_) with optional localization.

    Consult the section :ref:`esmda` in the manual for the theory and more
    information about ESMDA.

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
    alphas : {int, array-like}, default: 4
        Inflation factors for ESMDA.
    data_prior : ndarray, default: None
        Prior data ensemble, of shape ``(ne, nd)``.
    callback_post : function, default: None
        Function to be executed after each ESMDA iteration to the posterior
        model, ``callback_post(model_post)``.
    return_post_data : bool, default: True
        If true, returns also ``forward(model_post)``.
    return_steps : bool, default: False
        If true, returns model and data of all ESMDA steps. Setting
        ``return_steps`` to True enforces ``return_post_data=True``.
    random : {None, int,  np.random.Generator}, default: None
        Seed or random generator for reproducibility; see
        :func:`dageo.utils.rng`.
    localization_matrix : {ndarray, None}, default: None
        If provided, apply localization to the Kalman gain matrix, of shape
        ``(model-shape, nd)``.


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

    # Get random number generator
    rng = utils.rng(random)

    # Expand sigma if float
    if np.asarray(sigma).size == 1:
        sigma = np.zeros(nd) + sigma

    # Get alphas
    if isinstance(alphas, int):
        alphas = np.zeros(alphas) + alphas
    else:
        alphas = np.asarray(alphas)
    if abs(np.sum(1/alphas)-1) > 0.01:
        warnings.warn(
            f"The sum of 1/alphas should be 1; provided: {np.sum(1/alphas)}."
        )

    # Copy prior as start of post (output)
    model_post = model_prior.copy()

    # Loop over alphas
    for i, alpha in enumerate(alphas):
        print(f"ESMDA step {i+1: 3d}; α={alpha}")

        # == Step (a) of Emerick & Reynolds, 2013 ==
        # Run the ensemble from time zero.

        # Get data
        if i > 0 or data_prior is None:
            data_prior = forward(model_post)

        # == Step (b) of Emerick & Reynolds, 2013 ==
        # For each ensemble member, perturb the observation vector using
        # d_uc = d_obs + sqrt(α_i) * C_D^0.5 z_d; z_d ~ N(0, I_N_d)

        zd = rng.normal(size=(ne, nd))
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

        # Apply localization if provided
        if localization_matrix is not None:
            K *= localization_matrix

        # Update the ensemble parameters
        model_post += np.moveaxis(K @ (data_pert - data_prior).T, -1, 0)

        # Apply any provided post-checks
        if callback_post:
            callback_post(model_post)

        # If intermediate steps are wanted, store results
        if return_steps:
            # Initiate output if first iteration
            if i == 0:
                all_models = np.zeros((alphas.size+1, *model_post.shape))
                all_models[0, ...] = model_prior
                all_data = np.zeros((alphas.size+1, *data_prior.shape))
            all_models[i+1, ...] = model_post
            all_data[i, ...] = data_prior

    # Compute posterior data if wanted
    if return_post_data or return_steps:
        data_post = forward(model_post)
        if return_steps:
            all_data[-1, ...] = forward(model_post)

    # Return posterior model and corresponding data (if wanted)
    if return_steps:
        return all_models, all_data
    elif return_post_data:
        return model_post, data_post
    else:
        return model_post
