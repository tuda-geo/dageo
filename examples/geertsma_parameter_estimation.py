#%%
r"""
Geertsma Parameter Estimation with ESMDA
========================================

This example demonstrates how to use ESMDA for estimating Geertsma model 
parameters when the pressure field is known but reservoir properties are uncertain.

In many geomechanical applications, we know the pressure change in a reservoir
(from well measurements or production data), but are uncertain about the 
reservoir properties that control how this pressure change translates to surface
subsidence. In such cases, we can use ESMDA to estimate these parameters from
observed subsidence data.

Parameters that can be estimated include:
- Depth to reservoir
- Reservoir radius (for disc model)
- Uniaxial compaction coefficient
- Poisson's ratio
- Reservoir thickness

This example shows:
1. How to set up a parameter estimation problem with ESMDA
2. How to create a custom ensemble of Geertsma model parameters
3. How to analyze uncertainty reduction in estimated parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import dageo

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(4242)

# sphinx_gallery_thumbnail_number = 4

###############################################################################
# Problem Setup
# ------------
#
# We'll use the basic Geertsma disc model for this example. The scenario is:
# - The pressure change in the reservoir is known (e.g., from well data)
# - We have observed subsidence data from InSAR or leveling
# - We want to estimate key reservoir parameters that control subsidence

# Known pressure field (assume a uniform pressure drop of 5 MPa)
p0 = 20.0e6          # Initial pressure: 20 MPa
p_drop = 5.0e6       # Pressure drop: 5 MPa
p_field = np.ones((1, 1)) * (p0 - p_drop)  # Current pressure: 15 MPa

# True parameter values (these would be unknown in a real application)
true_depth = 2000.0          # Depth: 2000 m
true_radius = 1000.0         # Radius: 1000 m
true_thickness = 50.0        # Thickness: 50 m
true_cm = 1.0e-9             # Compaction coefficient: 1e-9 1/Pa
true_nu = 0.25               # Poisson's ratio: 0.25

# Number of ensembles and observation grid
ne = 100                     # Number of ensemble members
nobs = 21                    # Number of observation points
obs_range = 3000.0           # Observation extent (m)

# Create observation grid
X = np.linspace(-obs_range, obs_range, nobs)
Y = np.linspace(-obs_range, obs_range, nobs)
X_grid, Y_grid = np.meshgrid(X, Y)
obs_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

###############################################################################
# Create "True" Observations
# -----------------------
#
# Using the true parameters, generate synthetic observations of subsidence.

# Create the true Geertsma model
true_model = dageo.Geertsma(
    depth=true_depth,
    radius=true_radius,
    thickness=true_thickness,
    cm=true_cm,
    nu=true_nu,
    p0=p0,
    obs_points=obs_points
)

# Generate true subsidence
subsidence_true = true_model(p_field)

# Add observation noise
dstd = 0.0005  # 0.5 mm noise standard deviation
subsidence_obs = subsidence_true + rng.normal(0, dstd, size=subsidence_true.shape)

# Reshape for plotting
Z_true = subsidence_true.reshape(nobs, nobs)
Z_obs = subsidence_obs.reshape(nobs, nobs)

# Plot true and observed subsidence
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# True subsidence
im1 = axs[0].pcolormesh(X_grid, Y_grid, Z_true*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("True Subsidence")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
fig.colorbar(im1, ax=axs[0], label="Subsidence (mm)")

# Observed subsidence with noise
im2 = axs[1].pcolormesh(X_grid, Y_grid, Z_obs*1000, cmap='RdBu_r', shading='auto')
axs[1].set_title("Observed Subsidence (with noise)")
axs[1].set_xlabel("x (m)")
fig.colorbar(im2, ax=axs[1], label="Subsidence (mm)")

# Draw circle to represent the reservoir extent
for ax in axs:
    ax.add_patch(plt.Circle((0, 0), true_radius, fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')

###############################################################################
# Define Parameter Ranges
# --------------------
#
# For parameter estimation, we need to define reasonable prior ranges for each
# parameter. In practice, these would be based on geological knowledge and
# physical constraints.

# Parameter bounds
depth_bounds = (1500.0, 2500.0)      # Depth range: 1500-2500 m
radius_bounds = (500.0, 1500.0)       # Radius range: 500-1500 m
thickness_bounds = (20.0, 100.0)     # Thickness range: 20-100 m
cm_bounds = (0.5e-9, 2.0e-9)         # Compaction coefficient range: 0.5-2.0 e-9 1/Pa
nu_bounds = (0.15, 0.35)             # Poisson's ratio range: 0.15-0.35

###############################################################################
# Create Prior Parameter Ensemble
# ----------------------------
#
# Generate an ensemble of parameter sets by sampling from the prior distributions.
# We'll use uniform distributions for simplicity.

# Generate prior parameter ensemble
depth_prior = rng.uniform(depth_bounds[0], depth_bounds[1], size=ne)
radius_prior = rng.uniform(radius_bounds[0], radius_bounds[1], size=ne)
thickness_prior = rng.uniform(thickness_bounds[0], thickness_bounds[1], size=ne)
cm_prior = rng.uniform(cm_bounds[0], cm_bounds[1], size=ne)
nu_prior = rng.uniform(nu_bounds[0], nu_bounds[1], size=ne)

# Store parameters in a 2D array for easier handling
# Each row is an ensemble member, each column is a parameter
param_prior = np.column_stack((depth_prior, radius_prior, thickness_prior, cm_prior, nu_prior))
param_names = ['Depth (m)', 'Radius (m)', 'Thickness (m)', 'Comp. Coef. (1/Pa)', 'Poisson Ratio']

# Plot histograms of prior parameter distributions
fig, axs = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)

for i, (name, ax) in enumerate(zip(param_names, axs)):
    ax.hist(param_prior[:, i], bins=15, alpha=0.7, color='blue')
    ax.axvline(x=np.array([true_depth, true_radius, true_thickness, true_cm, true_nu])[i], 
              color='red', linestyle='--')
    ax.set_title(name)
    
    # Add text with true value
    if i < 3:  # First three parameters (depth, radius, thickness)
        true_val = np.array([true_depth, true_radius, true_thickness])[i]
        ax.text(0.05, 0.95, f'True: {true_val:.1f}', transform=ax.transAxes,
               verticalalignment='top', color='red')
    elif i == 3:  # Compaction coefficient
        ax.text(0.05, 0.95, f'True: {true_cm:.1e}', transform=ax.transAxes,
               verticalalignment='top', color='red')
    else:  # Poisson's ratio
        ax.text(0.05, 0.95, f'True: {true_nu:.2f}', transform=ax.transAxes,
               verticalalignment='top', color='red')

###############################################################################
# Define Forward Model for ESMDA
# ---------------------------
#
# Create a custom forward model function that takes parameter vectors and returns
# subsidence predictions. This function will:
# 1. Take a parameter ensemble as input
# 2. Create Geertsma models with these parameters
# 3. Calculate subsidence using the known pressure field
# 4. Return the resulting subsidence predictions

def forward_model(params):
    """Forward model for Geertsma parameter estimation.
    
    Takes parameter ensembles and returns subsidence predictions.
    
    Parameters
    ----------
    params : ndarray
        Parameter ensemble of shape (ne, 5), where each row contains
        [depth, radius, thickness, cm, nu]
        
    Returns
    -------
    subsidence : ndarray
        Predicted subsidence for each parameter set
    """
    ne = params.shape[0]  # Number of ensemble members
    nobs_points = obs_points.shape[0]  # Number of observation points
    subsidence = np.zeros((ne, nobs_points))
    
    # Loop through ensemble members
    for i in range(ne):
        # Extract parameters
        depth = params[i, 0]
        radius = params[i, 1]
        thickness = params[i, 2]
        cm = params[i, 3]
        nu = params[i, 4]
        
        # Create Geertsma model with these parameters
        model = dageo.Geertsma(
            depth=depth,
            radius=radius,
            thickness=thickness,
            cm=cm,
            nu=nu,
            p0=p0,
            obs_points=obs_points
        )
        
        # Calculate subsidence
        subsidence[i, :] = model(p_field)
    
    return subsidence

###############################################################################
# Calculate Prior Predictions
# -----------------------
#
# Run the forward model with the prior parameter ensemble to get subsidence
# predictions before assimilation.

# Calculate prior subsidence predictions
subsidence_prior = forward_model(param_prior)

# Plot a sample of prior subsidence predictions
fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
axs = axs.flatten()

# Plot observed (true) subsidence
im = axs[0].pcolormesh(X_grid, Y_grid, Z_obs*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("Observed Subsidence")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# Plot a few prior realizations
for i in range(1, 6):
    idx = i * 10  # Sample every 10th ensemble member
    Z_prior = subsidence_prior[idx].reshape(nobs, nobs)
    im = axs[i].pcolormesh(X_grid, Y_grid, Z_prior*1000, cmap='RdBu_r', shading='auto')
    axs[i].set_title(f"Prior Realization {i}")
    axs[i].set_xlabel("x (m)")
    if i == 1:
        axs[i].set_ylabel("y (m)")

# Add colorbar
fig.colorbar(im, ax=axs, label="Subsidence (mm)")

# Set aspect ratio for all subplots
for ax in axs:
    ax.set_aspect('equal')

###############################################################################
# Parameter Constraints
# ------------------
#
# When estimating parameters, it's important to enforce physical constraints.
# Define a callback function to keep parameters within their allowed ranges.

def restrict_parameters(params):
    """Restrict parameters to physical bounds."""
    # Clip each parameter to its bounds
    params[:, 0] = np.clip(params[:, 0], depth_bounds[0], depth_bounds[1])
    params[:, 1] = np.clip(params[:, 1], radius_bounds[0], radius_bounds[1])
    params[:, 2] = np.clip(params[:, 2], thickness_bounds[0], thickness_bounds[1])
    params[:, 3] = np.clip(params[:, 3], cm_bounds[0], cm_bounds[1])
    params[:, 4] = np.clip(params[:, 4], nu_bounds[0], nu_bounds[1])

###############################################################################
# Apply ESMDA
# ---------
#
# Use ESMDA to estimate the parameter values from the observed subsidence data.

# Run ESMDA
param_post, subsidence_post = dageo.esmda(
    model_prior=param_prior,
    forward=forward_model,
    data_obs=subsidence_obs,
    sigma=dstd,
    alphas=4,
    data_prior=subsidence_prior,
    callback_post=restrict_parameters,
    random=rng,
)

###############################################################################
# Analyze Results
# ------------
#
# Compare the prior and posterior parameter distributions with the true values.

# Plot histograms of prior and posterior parameter distributions
fig, axs = plt.subplots(2, 5, figsize=(15, 8), constrained_layout=True, sharex='col')

for i, name in enumerate(param_names):
    # Prior distribution
    axs[0, i].hist(param_prior[:, i], bins=15, alpha=0.7, color='blue')
    axs[0, i].axvline(x=np.array([true_depth, true_radius, true_thickness, true_cm, true_nu])[i], 
                    color='red', linestyle='--')
    axs[0, i].set_title(f"Prior: {name}")
    
    # Posterior distribution
    axs[1, i].hist(param_post[:, i], bins=15, alpha=0.7, color='green')
    axs[1, i].axvline(x=np.array([true_depth, true_radius, true_thickness, true_cm, true_nu])[i], 
                    color='red', linestyle='--')
    axs[1, i].set_title(f"Posterior: {name}")

# Calculate statistics
param_prior_mean = np.mean(param_prior, axis=0)
param_prior_std = np.std(param_prior, axis=0)
param_post_mean = np.mean(param_post, axis=0)
param_post_std = np.std(param_post, axis=0)
param_true = np.array([true_depth, true_radius, true_thickness, true_cm, true_nu])

# Calculate error and uncertainty reduction
rel_error_prior = np.abs(param_prior_mean - param_true) / param_true * 100
rel_error_post = np.abs(param_post_mean - param_true) / param_true * 100
error_reduction = (rel_error_prior - rel_error_post) / rel_error_prior * 100
uncertainty_reduction = (1 - param_post_std / param_prior_std) * 100

# Print results
print("Parameter Estimation Results:")
print("-" * 90)
print(f"{'Parameter':<15} {'True Value':<15} {'Prior Mean':<15} {'Post Mean':<15} "
      f"{'Error Red.(%)':<15} {'Uncert. Red.(%)':<15}")
print("-" * 90)

for i, name in enumerate(param_names):
    if i == 3:  # Compaction coefficient, use scientific notation
        print(f"{name:<15} {param_true[i]:.2e} {param_prior_mean[i]:.2e} "
              f"{param_post_mean[i]:.2e} {error_reduction[i]:>13.1f}% "
              f"{uncertainty_reduction[i]:>14.1f}%")
    else:  # Other parameters, use regular notation
        print(f"{name:<15} {param_true[i]:.2f} {param_prior_mean[i]:.2f} "
              f"{param_post_mean[i]:.2f} {error_reduction[i]:>13.1f}% "
              f"{uncertainty_reduction[i]:>14.1f}%")

print("-" * 90)

###############################################################################
# Cross-section Comparison
# ---------------------
#
# Compare the observed subsidence with the prior and posterior predictions.

# Calculate mean subsidence for prior and posterior
subsidence_prior_mean = np.mean(subsidence_prior, axis=0).reshape(nobs, nobs)
subsidence_post_mean = np.mean(subsidence_post, axis=0).reshape(nobs, nobs)

# Extract cross-section at y=0
mid_idx = nobs // 2
y_true_cross = Z_true[mid_idx, :] * 1000  # Convert to mm
y_obs_cross = Z_obs[mid_idx, :] * 1000
y_prior_mean_cross = subsidence_prior_mean[mid_idx, :] * 1000
y_post_mean_cross = subsidence_post_mean[mid_idx, :] * 1000

# Extract ensemble cross-sections
y_prior_cross = np.zeros((ne, nobs))
y_post_cross = np.zeros((ne, nobs))

for i in range(ne):
    y_prior = subsidence_prior[i].reshape(nobs, nobs)
    y_post = subsidence_post[i].reshape(nobs, nobs)
    y_prior_cross[i, :] = y_prior[mid_idx, :] * 1000
    y_post_cross[i, :] = y_post[mid_idx, :] * 1000

# Calculate std for cross-sections
y_prior_std = np.std(y_prior_cross, axis=0)
y_post_std = np.std(y_post_cross, axis=0)

# Plot cross-section comparison
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot prior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(X, y_prior_cross[i, :], color='lightblue', alpha=0.2)

# Plot posterior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(X, y_post_cross[i, :], color='lightgreen', alpha=0.2)

# Plot mean and confidence intervals
ax.fill_between(X, y_prior_mean_cross - y_prior_std, 
                y_prior_mean_cross + y_prior_std,
                color='blue', alpha=0.2, label='Prior ±σ')
ax.fill_between(X, y_post_mean_cross - y_post_std, 
                y_post_mean_cross + y_post_std,
                color='green', alpha=0.2, label='Posterior ±σ')

# Plot observed data
ax.plot(X, y_obs_cross, 'ro', label='Observations')
ax.plot(X, y_true_cross, 'k-', label='True')
ax.plot(X, y_prior_mean_cross, 'b-', label='Prior Mean')
ax.plot(X, y_post_mean_cross, 'g-', label='Posterior Mean')

ax.set_title("Cross-section of Subsidence at y=0")
ax.set_xlabel("x (m)")
ax.set_ylabel("Subsidence (mm)")
ax.legend()
ax.grid(True)

# Invert y-axis as subsidence is downward
ax.invert_yaxis()

###############################################################################
# 2D Subsidence Comparison
# ---------------------
#
# Compare the observed subsidence with the prior and posterior predictions.

# Plot 2D subsidence comparison
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# True/Observed subsidence
im1 = axs[0].pcolormesh(X_grid, Y_grid, Z_obs*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("Observed Subsidence")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# Prior mean subsidence
im2 = axs[1].pcolormesh(X_grid, Y_grid, subsidence_prior_mean*1000, cmap='RdBu_r', shading='auto')
axs[1].set_title("Prior Mean Subsidence")
axs[1].set_xlabel("x (m)")

# Posterior mean subsidence
im3 = axs[2].pcolormesh(X_grid, Y_grid, subsidence_post_mean*1000, cmap='RdBu_r', shading='auto')
axs[2].set_title("Posterior Mean Subsidence")
axs[2].set_xlabel("x (m)")

fig.colorbar(im1, ax=axs, label="Subsidence (mm)")

# Draw circle to represent the true reservoir extent
for ax in axs:
    ax.add_patch(plt.Circle((0, 0), true_radius, fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')


###############################################################################
# Calculate RMSE for Subsidence Predictions
# -------------------------------------
#
# Quantify the improvement in subsidence predictions.

# Calculate RMSE for subsidence
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Calculate RMSE for prior and posterior subsidence predictions
rmse_prior = rmse(subsidence_prior_mean, Z_true)
rmse_post = rmse(subsidence_post_mean, Z_true)

# Error reduction percentage
subsidence_error_reduction = (1 - rmse_post / rmse_prior) * 100

print("\nSubsidence Prediction Improvement:")
print(f"RMSE in subsidence: Prior = {rmse_prior*1000:.3f} mm, " +
      f"Posterior = {rmse_post*1000:.3f} mm")
print(f"Subsidence error reduction: {subsidence_error_reduction:.1f}%")

###############################################################################
# Parameter Correlation Analysis
# --------------------------
#
# Analyze the correlation between parameters in the posterior ensemble.
# This can reveal potential trade-offs or non-uniqueness in the solution.

# Calculate correlation matrix
corr_matrix = np.corrcoef(param_post, rowvar=False)

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label='Correlation')

# Add parameter names
ax.set_xticks(np.arange(len(param_names)))
ax.set_yticks(np.arange(len(param_names)))
ax.set_xticklabels(param_names)
ax.set_yticklabels(param_names)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add correlation values
for i in range(len(param_names)):
    for j in range(len(param_names)):
        text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                      ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")

ax.set_title("Parameter Correlation Matrix")

###############################################################################
# Conclusion
# --------
#
# This example demonstrated how to use ESMDA to estimate Geertsma model parameters 
# when the pressure field is known. The main findings are:
#
# 1. ESMDA can effectively estimate multiple reservoir parameters simultaneously.
# 2. Some parameters are better constrained than others, depending on their influence
#    on the subsidence pattern.
# 3. Parameter correlations reveal potential trade-offs in the solution.
#
# This approach can be valuable in geomechanical applications where:
# - Pressure changes are known from well data
# - Subsidence is measured at the surface
# - Reservoir properties are uncertain
#
# Extensions of this approach could include:
# - Using the GeertsmaFullGrid model for more complex reservoir geometries
# - Estimating spatially varying reservoir properties
# - Combining with pressure estimation in a joint inversion framework

dageo.Report()