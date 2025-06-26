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

__all__ = ["esmda", "esmda_subspace"]


def __dir__():
    return __all__


def esmda(
    model_prior,
    forward,
    data_obs,
    sigma,
    alphas=4,
    data_prior=None,
    localization_matrix=None,
    callback_post=None,
    return_post_data=True,
    return_steps=False,
    random=None,
    verbose=True,
):
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
    verbose : bool, default: True
        Print progress messages.


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
    if abs(np.sum(1 / alphas) - 1) > 0.01:
        warnings.warn(
            f"The sum of 1/alphas should be 1; provided: {np.sum(1 / alphas)}."
        )

    # Copy prior as start of post (output)
    model_post = model_prior.copy()

    # Loop over alphas
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"ESMDA step {i + 1: 3d}; α={alpha}")

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
        K = CMD @ Cinv

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
                all_models = np.zeros((alphas.size + 1, *model_post.shape))
                all_models[0, ...] = model_prior
                all_data = np.zeros((alphas.size + 1, *data_prior.shape))
            all_models[i + 1, ...] = model_post
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


# -----------------------------------------------------------------------------
# Helper functions for esmda_subspace
# -----------------------------------------------------------------------------
def _default_alphas(n_iter: int) -> np.ndarray:
    """Equal weights αₖ = n_iter (sum of 1/αₖ = 1)."""
    return np.full(n_iter, n_iter, dtype=float)


def _anomalies(ensemble: np.ndarray) -> np.ndarray:
    """Compute ensemble anomalies normalized by sqrt(N-1)."""
    return (ensemble - ensemble.mean(axis=0, keepdims=True)) / np.sqrt(
        ensemble.shape[0] - 1.0
    )


def _perturb_obs(
    obs: np.ndarray, sigma: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """Perturb observations with scaled noise."""
    noise = rng.standard_normal(obs.shape) * sigma * np.sqrt(alpha)
    return obs + noise


def gaspari_cohn(dist: np.ndarray, radius: float) -> np.ndarray:
    """Gaspari-Cohn correlation function for localization.

    Parameters
    ----------
    dist : ndarray
        Distances between points.
    radius : float
        Localization radius.

    Returns
    -------
    rho : ndarray
        Correlation values (0 to 1).
    """
    q = np.abs(dist) / radius
    rho = np.zeros_like(q)

    # For q <= 1
    i = q <= 1.0
    qi = q[i]
    rho[i] = (
        1.0
        - 5.0 / 3.0 * qi**2
        + 5.0 / 8.0 * qi**3
        + 0.5 * qi**4
        - 0.25 * qi**5
    )

    # For 1 < q <= 2
    i = (q > 1.0) & (q <= 2.0)
    qi = q[i]
    rho[i] = (
        4.0 / 3.0
        - 5.0 * qi
        + 5.0 * qi**2
        - 5.0 / 3.0 * qi**3
        + 0.5 * qi**4
        - 1.0 / 12.0 * qi**5
    )

    return np.clip(rho, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Core update routines for esmda_subspace
# -----------------------------------------------------------------------------
def _update_global(
    X: np.ndarray,
    A_t: np.ndarray,
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    alpha: float,
) -> None:
    """Global update without localization (in-place)."""
    ne = X.shape[0]
    eye = np.eye(ne)

    # Check for numerical issues in input
    if np.any(~np.isfinite(Y_tilde)) or np.any(~np.isfinite(D_tilde)):
        # Skip update if data contains inf/nan
        return

    # Compute with numerical stability
    try:
        # Suppress warnings for this computation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            G = Y_tilde.T @ Y_tilde + alpha * eye  # (N×N)
            rhs = Y_tilde.T @ D_tilde  # (N×N)

            # Check condition number
            cond = np.linalg.cond(G)
            if cond > 1e12:
                # Matrix is ill-conditioned, add more regularization
                G += 1e-6 * eye

            # Solve the system
            W = np.linalg.solve(G, rhs)  # LU solve

            # Check result before updating
            increment = (A_t @ W).T
            if np.all(np.isfinite(increment)):
                X += increment
    except (np.linalg.LinAlgError, ValueError):
        # Skip update on numerical errors
        pass


def _update_local(
    X: np.ndarray,
    A_t: np.ndarray,
    Y_tilde: np.ndarray,
    D_tilde: np.ndarray,
    alpha: float,
    loc_fn,
) -> None:
    """Local update with R-inflation localization (in-place).

    This implements R-inflation localization where observation errors are
    inflated based on correlation distance. This is mathematically equivalent
    to Kalman gain localization.
    """
    ne = X.shape[0]
    n_state = X.shape[1]

    # For each state variable, apply localized update
    for i in range(n_state):
        # Get localization weights for state i vs all observations
        rho = loc_fn(i)  # (n_obs,)

        # Skip if no observations influence this state
        if np.max(rho) < 1e-10:
            continue

        # R-inflation: modify observation error based on correlation
        # When rho = 0, we want R -> infinity (no update)
        # When rho = 1, we want R unchanged
        # Since Y_tilde = Y_anom * sqrtR_inv, to inflate R by 1/rho,
        # we need to scale Y_tilde by sqrt(rho)

        rho_safe = np.maximum(rho, 1e-10)  # Avoid division by zero
        scale = np.sqrt(rho_safe)

        # Apply localization scaling
        Y_tilde_loc = Y_tilde * scale[:, None]  # (n_obs, ne)
        D_tilde_loc = D_tilde * scale[:, None]  # (n_obs, ne)

        # Check for numerical issues
        if np.any(~np.isfinite(Y_tilde_loc)) or np.any(
            ~np.isfinite(D_tilde_loc)
        ):
            continue

        try:
            # Suppress warnings for this computation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # Build and solve the localized system
                G_loc = Y_tilde_loc.T @ Y_tilde_loc + alpha * np.eye(ne)
                rhs_loc = Y_tilde_loc.T @ D_tilde_loc

                # Check condition number
                cond = np.linalg.cond(G_loc)
                if cond > 1e12:
                    # Matrix is ill-conditioned, add more regularization
                    G_loc += 1e-6 * np.eye(ne)

                W_loc = np.linalg.solve(G_loc, rhs_loc)

                # Compute the increment for state i
                # W_loc is (ne, ne) matrix that transforms ensemble members
                # Following the global update: X += (A_t @ W).T
                # For state i: X[:, i] += (A_t[i, :] @ W_loc)
                increment = A_t[i] @ W_loc  # (ne,)

                # Check result before updating
                if np.all(np.isfinite(increment)):
                    X[:, i] += increment
        except (np.linalg.LinAlgError, ValueError):
            # Skip this state variable on numerical errors
            continue


def esmda_subspace(
    model_prior,
    forward,
    data_obs,
    sigma,
    alphas=4,
    data_prior=None,
    localization_function=None,
    callback_post=None,
    return_post_data=True,
    return_steps=False,
    random=None,
    verbose=True,
):
    """ES-MDA using subspace formulation with LU-solve.

    Identical to the standard ESMDA but uses LU decomposition
    (numpy.linalg.solve) instead of Cholesky, following the
    approach in Geir Evensen's implementations. Uses function-based
    localization instead of matrix-based.

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
    localization_function : callable, optional
        Function that takes state index and returns correlation vector
        for all observations. If None, no localization is applied.
    verbose : bool, default: True
        Print progress messages.

    Returns
    -------
    model_post : ndarray
        Posterior model ensemble.
    data_post : ndarray, only returned if ``return_post_data=True``
        Posterior simulated data ensemble.
    """
    # Get dimensions
    model_shape = model_prior.shape
    ne = model_shape[0]
    nd = data_obs.size

    # Get random number generator
    rng = utils.rng(random)

    # Expand sigma if float
    if np.asarray(sigma).size == 1:
        sigma = np.zeros(nd) + sigma
    else:
        sigma = np.asarray(sigma)

    # Get alphas
    if isinstance(alphas, int):
        alphas = np.zeros(alphas) + alphas
    else:
        alphas = np.asarray(alphas)
    if abs(np.sum(1 / alphas) - 1) > 0.01:
        warnings.warn(
            f"The sum of 1/alphas should be 1; provided: {np.sum(1 / alphas)}."
        )

    # Copy prior as start of post (output)
    model_post = model_prior.copy()

    # For 2D models, flatten for efficient computation
    if model_prior.ndim > 2:
        original_shape = model_shape
        model_post = model_post.reshape(ne, -1)
    else:
        original_shape = None

    # Prepare for storing steps if requested
    if return_steps:
        all_models = np.zeros((alphas.size + 1, *model_shape))
        all_models[0, ...] = model_prior
        all_data = np.zeros((alphas.size + 1, ne, nd))

    # Prepare scaling
    sqrtR_inv = 1.0 / sigma

    # Loop over alphas
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"ESMDA step {i + 1: 3d}; α={alpha}")

        # Get data
        if i > 0 or data_prior is None:
            if original_shape is not None:
                # Reshape back for forward model
                data_prior = forward(model_post.reshape(original_shape))
            else:
                data_prior = forward(model_post)

        # Perturb observations
        obs_pert = _perturb_obs(data_obs, sigma, alpha, rng)

        # Compute anomalies
        A_t = _anomalies(model_post).T  # (n_state, N_e)
        Y_anom_t = _anomalies(data_prior).T  # (N_obs, N_e)

        # Check for numerical issues in data
        if np.any(~np.isfinite(data_prior)) or np.any(~np.isfinite(Y_anom_t)):
            warnings.warn("Numerical issues detected in data, skipping update")
            continue

        # Scale by observation error
        Y_tilde = Y_anom_t * sqrtR_inv[:, None]
        D_tilde = (obs_pert - data_prior).T * sqrtR_inv[:, None]

        # Update
        if localization_function is None:
            _update_global(model_post, A_t, Y_tilde, D_tilde, alpha)
        else:
            _update_local(
                model_post, A_t, Y_tilde, D_tilde, alpha, localization_function
            )

        # Apply callback
        if callback_post:
            if original_shape is not None:
                # Reshape for callback
                temp = model_post.reshape(original_shape)
                callback_post(temp)
                model_post = temp.reshape(ne, -1)
            else:
                callback_post(model_post)

        # Store steps if requested
        if return_steps:
            if original_shape is not None:
                all_models[i + 1, ...] = model_post.reshape(original_shape)
            else:
                all_models[i + 1, ...] = model_post
            all_data[i, ...] = data_prior

    # Reshape back if needed
    if original_shape is not None:
        model_post = model_post.reshape(original_shape)

    # Compute posterior data if wanted
    if return_post_data or return_steps:
        data_post = forward(model_post)
        if return_steps:
            all_data[-1, ...] = data_post

    # Return results
    if return_steps:
        return all_models, all_data
    elif return_post_data:
        return model_post, data_post
    else:
        return model_post
