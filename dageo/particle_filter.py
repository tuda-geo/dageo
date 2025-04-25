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
from dageo import utils

__all__ = ['particle_filter']


def __dir__():
    return __all__


def particle_filter(model_prior, forward, data_obs, sigma, n_steps=1,
                    resampling_threshold=0.5, callback_post=None,
                    return_post_data=True, return_weights=False,
                    return_steps=False, random=None):
    """Particle Filter algorithm for sequential data assimilation.

    The Particle Filter is a sequential Monte Carlo method for Bayesian state
    estimation of a dynamical system. It represents the posterior distribution
    using a set of weighted particles (samples). The algorithm consists of
    three main steps:
    1. Prediction: Propagate particles through the forward model
    2. Update: Calculate weights based on the likelihood of observations
    3. Resampling: Resample particles based on weights to avoid degeneracy

    Parameters
    ----------
    model_prior : ndarray
        Prior models of dimension ``(n_particles, ...)``, where ``n_particles`` is the number of
        particles.
    forward : callable
        Forward model that takes an ndarray of the shape of the prior models
        ``(n_particles, ...)``, and returns a ndarray of the shape of the prior data
        ``(n_particles, nd)``; ``n_particles`` is the number of particles, ``nd`` the number of
        data.
    data_obs : ndarray
        Observed data of shape ``(nd)``. If n_steps > 1, can be of shape
        ``(n_steps, nd)`` to provide different observations at each step.
    sigma : {float, ndarray}
        Standard deviation(s) of the observation noise.
    n_steps : int, default: 1
        Number of sequential assimilation steps.
    resampling_threshold : float, default: 0.5
        Threshold for effective sample size ratio (0 to 1) below which
        resampling is performed.
    callback_post : function, default: None
        Function to be executed after each particle filter iteration on the
        posterior model, ``callback_post(model_post)``.
    return_post_data : bool, default: True
        If true, returns also ``forward(model_post)``.
    return_weights : bool, default: False
        If true, returns the final weights of the particles.
    return_steps : bool, default: False
        If true, returns model and data of all steps. Setting ``return_steps``
        to True enforces ``return_post_data=True``.
    random : {None, int, np.random.Generator}, default: None
        Seed or random generator for reproducibility; see
        :func:`dageo.utils.rng`.

    Returns
    -------
    model_post : ndarray
        Posterior model ensemble.
    data_post : ndarray, only returned if ``return_post_data=True``
        Posterior simulated data ensemble.
    weights : ndarray, only returned if ``return_weights=True``
        Final weights of the particles.
    model_steps : ndarray, only returned if ``return_steps=True``
        Models at each step.
    data_steps : ndarray, only returned if ``return_steps=True``
        Data at each step.
    weights_steps : ndarray, only returned if ``return_steps and return_weights=True``
        Weights at each step.

    Notes
    -----
    The Particle Filter algorithm consists of the following steps:

    1. Initialize particles from the prior distribution
    2. For each sequential step:
        a. Predict: Propagate particles through the forward model
        b. Update: Calculate weights based on the likelihood of observations
        c. Resample: If effective sample size is below threshold, resample
           particles based on weights
    3. Return the posterior particles and optionally their weights
    """
    # Get number of particles and data points
    n_particles = model_prior.shape[0]
    nd = data_obs.shape[-1]

    # Get random number generator
    rng = utils.rng(random)

    # Expand sigma if float
    if isinstance(sigma, (int, float)) or (hasattr(sigma, 'size') and sigma.size == 1):
        sigma = np.zeros(nd) + sigma

    # Handle multiple observation sets or single one
    if data_obs.ndim == 1:
        data_obs = np.tile(data_obs[np.newaxis, :], (n_steps, 1))
    elif data_obs.ndim == 2 and n_steps > 1 and data_obs.shape[0] != n_steps:
        raise ValueError(f"Expected {n_steps} sets of observations, but got "
                         f"{data_obs.shape[0]}.")

    # Copy prior as start of post (output)
    model_post = model_prior.copy()
    weights = np.ones(n_particles) / n_particles  # Initial uniform weights

    # Store steps if required
    if return_steps:
        model_steps = np.zeros((n_steps + 1, *model_prior.shape))
        model_steps[0] = model_prior
        data_steps = np.zeros((n_steps, n_particles, nd))
        weights_steps = np.zeros((n_steps + 1, n_particles))
        weights_steps[0] = weights

    # Loop over steps
    for step in range(n_steps):
        print(f"Particle Filter step {step+1: 3d}")

        # Predict: Run forward model
        data_predicted = forward(model_post)

        # Update: Calculate likelihood and weights
        # Assume Gaussian likelihood: p(y|x) ∝ exp(-0.5 * (y - h(x))^2 / sigma^2)
        log_likelihood = np.zeros(n_particles)
        for i in range(n_particles):
            # Calculate log-likelihood for each particle
            residuals = data_predicted[i] - data_obs[step]
            log_likelihood[i] = -0.5 * np.sum((residuals / sigma)**2)

        # Avoid numerical issues with very small likelihoods
        log_likelihood -= np.max(log_likelihood)
        likelihood = np.exp(log_likelihood)

        # Update weights
        weights = weights * likelihood
        weights /= np.sum(weights)  # Normalize

        # Calculate effective sample size
        n_eff = 1.0 / np.sum(weights**2)
        n_eff_ratio = n_eff / n_particles

        print(f"  Effective sample size: {n_eff:.1f}/{n_particles} = {n_eff_ratio:.3f}")

        # Resample if effective sample size is below threshold
        if n_eff_ratio < resampling_threshold:
            print("  Resampling...")
            indices = rng.choice(n_particles, size=n_particles, p=weights)
            model_post = model_post[indices]
            # Reset weights after resampling
            weights = np.ones(n_particles) / n_particles
        
        # Apply any provided post-checks
        if callback_post:
            callback_post(model_post)

        # Store results for this step if required
        if return_steps:
            data_steps[step] = data_predicted
            model_steps[step + 1] = model_post
            weights_steps[step + 1] = weights

    # Compute posterior data if wanted
    if return_post_data or return_steps:
        data_post = forward(model_post)
        if return_steps:
            data_steps[-1] = data_post

    # Return based on requested outputs
    result = [model_post]
    
    if return_post_data or return_steps:
        result.append(data_post)
    
    if return_weights:
        result.append(weights)
    
    if return_steps:
        result.append(model_steps)
        result.append(data_steps)
        if return_weights:
            result.append(weights_steps)
    
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)