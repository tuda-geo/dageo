r"""
Geertsma Subsidence ESMDA example
=================================

This example demonstrates how to use both Geertsma models (simple disc model and 
full-grid model) to simulate surface subsidence due to reservoir pressure 
depletion and apply ESMDA to estimate reservoir pressure changes from observed 
subsidence data.

The example showcases:
1. How to use both Geertsma model implementations as forward models
2. How to simulate subsidence data for different pressure scenarios
3. How to use ESMDA to invert for reservoir pressure changes
4. Comparing the performance of the two models for different reservoir scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import dageo

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(3456)

# sphinx_gallery_thumbnail_number = 6

###############################################################################
# Part 1: Basic Geertsma model with disc-shaped reservoir
# ======================================================
#
# First, we'll demonstrate the basic Geertsma model which assumes a uniform 
# disc-shaped reservoir with uniform pressure change.

# Model parameters for disc model
depth = 2000.0       # Reservoir depth: 2000 m
radius = 1000.0      # Reservoir radius: 1000 m
thickness = 50.0     # Reservoir thickness: 50 m
cm = 1.0e-9          # Compaction coefficient: 1e-9 1/Pa
nu = 0.25            # Poisson's ratio: 0.25
p0 = 20.0e6          # Initial pressure: 20 MPa

# ESMDA parameters
ne = 100             # Number of ensembles
dstd = 0.002         # Standard deviation of observation noise (2 mm)

# Observation grid
nobs = 21            # 21x21 observation grid
obs_range = 3000.0   # Observation extent: 3000 m

# Create observation points grid for visualization
X = np.linspace(-obs_range, obs_range, nobs)
Y = np.linspace(-obs_range, obs_range, nobs)
X_grid, Y_grid = np.meshgrid(X, Y)
obs_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

###############################################################################
# Initialize the basic Geertsma model
# ----------------------------------

# Initialize the disc-based Geertsma model
disc_model = dageo.Geertsma(
    depth=depth, radius=radius, thickness=thickness,
    cm=cm, nu=nu, p0=p0, obs_points=obs_points
)

###############################################################################
# Generate "true" pressure field and synthetic observations
# ------------------------------------------------------
#
# For the basic model, we'll use a uniform pressure change.

# Define "true" pressure - a uniform value for the disc model
p_true_disc = np.ones((1, 1)) * 15.0e6  # 15 MPa (pressure drop from initial 20 MPa)

# Calculate "true" subsidence
subsidence_true_disc = disc_model(p_true_disc)

# Add random noise to create synthetic observations
subsidence_obs_disc = subsidence_true_disc + rng.normal(0, dstd, size=subsidence_true_disc.shape)

# Reshape for plotting
Z_true_disc = subsidence_true_disc.reshape(nobs, nobs)
Z_obs_disc = subsidence_obs_disc.reshape(nobs, nobs)

# Plot true and observed subsidence
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# True subsidence
im1 = axs[0].pcolormesh(X_grid, Y_grid, Z_true_disc*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("True Subsidence - Disc Model")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
fig.colorbar(im1, ax=axs[0], label="Subsidence (mm)")

# Observed subsidence with noise
im2 = axs[1].pcolormesh(X_grid, Y_grid, Z_obs_disc*1000, cmap='RdBu_r', shading='auto')
axs[1].set_title("Observed Subsidence (with noise)")
axs[1].set_xlabel("x (m)")
fig.colorbar(im2, ax=axs[1], label="Subsidence (mm)")

# Draw circle to represent the reservoir extent
for ax in axs:
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')

###############################################################################
# Create prior ensemble for ESMDA
# -----------------------------
#
# For the disc model, our prior is a set of uniform pressure values.

# Define pressure bounds
p_min = 10.0e6  # 10 MPa
p_max = 25.0e6  # 25 MPa

# Generate prior ensemble - for the disc model, just random pressure values
p_prior_disc = rng.uniform(p_min, p_max, size=(ne, 1, 1))

# Run forward model on prior ensemble
subsidence_prior_disc = np.zeros((ne, nobs * nobs))
for i in range(ne):
    subsidence_prior_disc[i, :] = disc_model(p_prior_disc[i])

###############################################################################
# Apply ESMDA to estimate the pressure
# ----------------------------------

def restrict_pressure_disc(x):
    """Restrict possible pressures to the defined range."""
    np.clip(x, p_min, p_max, out=x)

# Run ESMDA
p_post_disc, subsidence_post_disc = dageo.esmda(
    model_prior=p_prior_disc,
    forward=disc_model,
    data_obs=subsidence_obs_disc,
    sigma=dstd,
    alphas=4,
    data_prior=subsidence_prior_disc,
    callback_post=restrict_pressure_disc,
    random=rng,
)

###############################################################################
# Analyze the results for the disc model
# ------------------------------------

# Plot cross-section of subsidence at y=0
mid_idx = nobs // 2
x_cross = X
y_true_cross = Z_true_disc[mid_idx, :] * 1000  # Convert to mm
y_obs_cross = Z_obs_disc[mid_idx, :] * 1000

# Extract cross-sections from prior and posterior ensembles
y_prior_cross_disc = np.zeros((ne, nobs))
y_post_cross_disc = np.zeros((ne, nobs))

for i in range(ne):
    y_prior = subsidence_prior_disc[i].reshape(nobs, nobs)
    y_post = subsidence_post_disc[i].reshape(nobs, nobs)
    y_prior_cross_disc[i, :] = y_prior[mid_idx, :] * 1000
    y_post_cross_disc[i, :] = y_post[mid_idx, :] * 1000

# Calculate mean and std
y_prior_mean_disc = y_prior_cross_disc.mean(axis=0)
y_prior_std_disc = y_prior_cross_disc.std(axis=0)
y_post_mean_disc = y_post_cross_disc.mean(axis=0)
y_post_std_disc = y_post_cross_disc.std(axis=0)

# Plot cross-section comparison
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot prior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(x_cross, y_prior_cross_disc[i, :], color='lightblue', alpha=0.2)

# Plot posterior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(x_cross, y_post_cross_disc[i, :], color='lightgreen', alpha=0.2)

# Plot mean and confidence intervals
ax.fill_between(x_cross, y_prior_mean_disc - y_prior_std_disc, 
                y_prior_mean_disc + y_prior_std_disc,
                color='blue', alpha=0.2, label='Prior ±σ')
ax.fill_between(x_cross, y_post_mean_disc - y_post_std_disc, 
                y_post_mean_disc + y_post_std_disc,
                color='green', alpha=0.2, label='Posterior ±σ')

# Plot observed data
ax.plot(x_cross, y_obs_cross, 'ro', label='Observations')
ax.plot(x_cross, y_true_cross, 'k-', label='True')
ax.plot(x_cross, y_prior_mean_disc, 'b-', label='Prior Mean')
ax.plot(x_cross, y_post_mean_disc, 'g-', label='Posterior Mean')

ax.set_title("Cross-section of Subsidence at y=0 - Disc Model")
ax.set_xlabel("x (m)")
ax.set_ylabel("Subsidence (mm)")
ax.legend()
ax.grid(True)

# Invert y-axis as subsidence is downward
ax.invert_yaxis()

# Display results
p_prior_mean_disc = p_prior_disc.mean() / 1e6  # Convert to MPa
p_prior_std_disc = p_prior_disc.std() / 1e6
p_post_mean_disc = p_post_disc.mean() / 1e6
p_post_std_disc = p_post_disc.std() / 1e6
p_true_disc_value = p_true_disc[0, 0] / 1e6

print("\nResults for Disc Model:")
print(f"True pressure: {p_true_disc_value:.2f} MPa")
print(f"Prior: {p_prior_mean_disc:.2f} ± {p_prior_std_disc:.2f} MPa")
print(f"Posterior: {p_post_mean_disc:.2f} ± {p_post_std_disc:.2f} MPa")
print(f"Uncertainty reduction: {(1 - p_post_std_disc/p_prior_std_disc)*100:.1f}%")
print(f"Error reduction: {(1 - abs(p_post_mean_disc - p_true_disc_value)/abs(p_prior_mean_disc - p_true_disc_value))*100:.1f}%")

###############################################################################
# Part 2: Full-grid Geertsma model with heterogeneous pressure field
# =================================================================
#
# Now, we'll demonstrate the more advanced GeertsmaFullGrid model which can
# handle spatially varying pressure changes in a gridded reservoir.

# Grid parameters
nx = 20
ny = 20
dx = 100.0
dy = 100.0

# Pressure parameters for the grid model
p_mean = 20.0e6      # 20 MPa mean pressure
p_min = 10.0e6       # 10 MPa minimum pressure
p_max = 30.0e6       # 30 MPa maximum pressure

###############################################################################
# Create heterogeneous pressure field
# --------------------------------
#
# We'll create a spatially varying pressure field using a Gaussian random field.

# Create a RandomPermeability instance to generate the fields
# We're repurposing this to create pressure fields
RP = dageo.RandomPermeability(
    nx, ny, p_mean, p_min, p_max, length=(500.0, 500.0), theta=0.0
)

# Generate the "true" pressure field and prior ensemble
p_true_grid = RP(1, random=rng)
p_prior_grid = RP(ne, random=rng)

# Plot the true pressure field
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
extent = [-nx*dx/2, nx*dx/2, -ny*dy/2, ny*dy/2]
im = ax.imshow(
    p_true_grid[0].T, origin='lower', cmap='viridis',
    extent=extent, vmin=p_min, vmax=p_max
)
ax.set_title("True Pressure Field - Grid Model")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
cbar = fig.colorbar(im, ax=ax, label="Pressure (Pa)")

###############################################################################
# Initialize the full-grid Geertsma model
# -------------------------------------

# Initialize the full-grid model
grid_model = dageo.GeertsmaFullGrid(
    nx=nx, ny=ny, depth=depth, dx=dx, dy=dy,
    thickness=thickness, cm=cm, nu=nu, p0=p_mean,
    obs_points=obs_points
)

###############################################################################
# Generate synthetic observations for the grid model
# ----------------------------------------------

# Calculate "true" subsidence from the true pressure field
subsidence_true_grid = grid_model(p_true_grid)

# Add random noise to create synthetic observations
subsidence_obs_grid = subsidence_true_grid + rng.normal(0, dstd, size=subsidence_true_grid.shape)

# Reshape for plotting
Z_true_grid = subsidence_true_grid.reshape(nobs, nobs)
Z_obs_grid = subsidence_obs_grid.reshape(nobs, nobs)

# Plot true and observed subsidence
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# True subsidence
im1 = axs[0].pcolormesh(X_grid, Y_grid, Z_true_grid*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("True Subsidence - Grid Model")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
fig.colorbar(im1, ax=axs[0], label="Subsidence (mm)")

# Observed subsidence with noise
im2 = axs[1].pcolormesh(X_grid, Y_grid, Z_obs_grid*1000, cmap='RdBu_r', shading='auto')
axs[1].set_title("Observed Subsidence (with noise)")
axs[1].set_xlabel("x (m)")
fig.colorbar(im2, ax=axs[1], label="Subsidence (mm)")

# Draw rectangle to represent the reservoir extent
for ax in axs:
    ax.add_patch(plt.Rectangle((-nx*dx/2, -ny*dy/2), nx*dx, ny*dy, 
                              fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')

###############################################################################
# Run forward model on prior ensemble for grid model
# ----------------------------------------------

# Calculate subsidence for the prior ensemble
subsidence_prior_grid = grid_model(p_prior_grid)

###############################################################################
# Apply ESMDA for the grid model
# ---------------------------

def restrict_pressure_grid(x):
    """Restrict possible pressures to the defined range."""
    np.clip(x, p_min, p_max, out=x)

# Run ESMDA
p_post_grid, subsidence_post_grid = dageo.esmda(
    model_prior=p_prior_grid,
    forward=grid_model,
    data_obs=subsidence_obs_grid,
    sigma=dstd,
    alphas=4,
    data_prior=subsidence_prior_grid,
    callback_post=restrict_pressure_grid,
    random=rng,
)

###############################################################################
# Compare prior and posterior pressure fields for grid model
# ------------------------------------------------------

# Calculate statistics
p_prior_mean_grid = p_prior_grid.mean(axis=0)
p_post_mean_grid = p_post_grid.mean(axis=0)

# Plot true, prior mean, and posterior mean pressure fields
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

im = axs[0].imshow(p_true_grid[0].T, origin='lower', cmap='viridis',
                   extent=extent, vmin=p_min, vmax=p_max)
axs[0].set_title("True Pressure Field")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

im = axs[1].imshow(p_prior_mean_grid.T, origin='lower', cmap='viridis',
                   extent=extent, vmin=p_min, vmax=p_max)
axs[1].set_title("Prior Mean")
axs[1].set_xlabel("x (m)")

im = axs[2].imshow(p_post_mean_grid.T, origin='lower', cmap='viridis',
                   extent=extent, vmin=p_min, vmax=p_max)
axs[2].set_title("Posterior Mean")
axs[2].set_xlabel("x (m)")

fig.colorbar(im, ax=axs, label="Pressure (Pa)")

###############################################################################
# Compare cross-sections of subsidence for grid model
# ------------------------------------------------

# Extract cross-sections at y=0
mid_idx = nobs // 2
y_true_cross_grid = Z_true_grid[mid_idx, :] * 1000  # Convert to mm
y_obs_cross_grid = Z_obs_grid[mid_idx, :] * 1000

# Extract cross-sections from prior and posterior ensembles
y_prior_cross_grid = np.zeros((ne, nobs))
y_post_cross_grid = np.zeros((ne, nobs))

for i in range(ne):
    y_prior = subsidence_prior_grid[i].reshape(nobs, nobs)
    y_post = subsidence_post_grid[i].reshape(nobs, nobs)
    y_prior_cross_grid[i, :] = y_prior[mid_idx, :] * 1000
    y_post_cross_grid[i, :] = y_post[mid_idx, :] * 1000

# Calculate mean and std
y_prior_mean_grid = y_prior_cross_grid.mean(axis=0)
y_prior_std_grid = y_prior_cross_grid.std(axis=0)
y_post_mean_grid = y_post_cross_grid.mean(axis=0)
y_post_std_grid = y_post_cross_grid.std(axis=0)

# Plot cross-section comparison
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot prior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(x_cross, y_prior_cross_grid[i, :], color='lightblue', alpha=0.2)

# Plot posterior ensemble realizations
for i in range(0, ne, 10):
    ax.plot(x_cross, y_post_cross_grid[i, :], color='lightgreen', alpha=0.2)

# Plot mean and confidence intervals
ax.fill_between(x_cross, y_prior_mean_grid - y_prior_std_grid, 
                y_prior_mean_grid + y_prior_std_grid,
                color='blue', alpha=0.2, label='Prior ±σ')
ax.fill_between(x_cross, y_post_mean_grid - y_post_std_grid, 
                y_post_mean_grid + y_post_std_grid,
                color='green', alpha=0.2, label='Posterior ±σ')

# Plot observed data
ax.plot(x_cross, y_obs_cross_grid, 'ro', label='Observations')
ax.plot(x_cross, y_true_cross_grid, 'k-', label='True')
ax.plot(x_cross, y_prior_mean_grid, 'b-', label='Prior Mean')
ax.plot(x_cross, y_post_mean_grid, 'g-', label='Posterior Mean')

ax.set_title("Cross-section of Subsidence at y=0 - Grid Model")
ax.set_xlabel("x (m)")
ax.set_ylabel("Subsidence (mm)")
ax.legend()
ax.grid(True)

# Invert y-axis as subsidence is downward
ax.invert_yaxis()

###############################################################################
# Calculate error and uncertainty reduction for grid model
# ----------------------------------------------------

# Calculate RMSE between true and predicted subsidence
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Calculate RMSE for prior and posterior pressure fields
rmse_prior_pressure = rmse(p_prior_mean_grid, p_true_grid[0])
rmse_post_pressure = rmse(p_post_mean_grid, p_true_grid[0])

# Calculate RMSE for prior and posterior subsidence predictions
rmse_prior_subsidence = rmse(
    subsidence_prior_grid.mean(axis=0).reshape(nobs, nobs), 
    Z_true_grid
)
rmse_post_subsidence = rmse(
    subsidence_post_grid.mean(axis=0).reshape(nobs, nobs), 
    Z_true_grid
)

# Error reduction percentage
pressure_error_reduction = (1 - rmse_post_pressure / rmse_prior_pressure) * 100
subsidence_error_reduction = (1 - rmse_post_subsidence / rmse_prior_subsidence) * 100

print("\nResults for Grid Model:")
print(f"RMSE in pressure field: Prior = {rmse_prior_pressure/1e6:.2f} MPa, " +
      f"Posterior = {rmse_post_pressure/1e6:.2f} MPa")
print(f"Pressure error reduction: {pressure_error_reduction:.1f}%")

print(f"RMSE in subsidence: Prior = {rmse_prior_subsidence*1000:.2f} mm, " +
      f"Posterior = {rmse_post_subsidence*1000:.2f} mm")
print(f"Subsidence error reduction: {subsidence_error_reduction:.1f}%")

###############################################################################
# Uncertainty reduction in grid model
# --------------------------------

# Calculate variance of prior and posterior ensemble
p_prior_var = p_prior_grid.var(axis=0)
p_post_var = p_post_grid.var(axis=0)

# Plot variance reduction
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Prior variance
vmax = np.max(p_prior_var)
im = axs[0].imshow(p_prior_var.T, origin='lower', cmap='plasma',
                   extent=extent, vmin=0, vmax=vmax)
axs[0].set_title("Prior Pressure Variance")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# Posterior variance
im = axs[1].imshow(p_post_var.T, origin='lower', cmap='plasma',
                   extent=extent, vmin=0, vmax=vmax)
axs[1].set_title("Posterior Pressure Variance")
axs[1].set_xlabel("x (m)")

# Variance reduction percentage
variance_reduction = (1 - p_post_var / p_prior_var) * 100
im = axs[2].imshow(variance_reduction.T, origin='lower', cmap='viridis',
                  extent=extent, vmin=0, vmax=100)
axs[2].set_title("Variance Reduction (%)")
axs[2].set_xlabel("x (m)")

# Add colorbars
cbar1 = fig.colorbar(im, ax=axs[0:2], label="Pressure Variance (Pa²)")
cbar2 = fig.colorbar(im, ax=axs[2], label="Variance Reduction (%)")

###############################################################################
# Part 3: Comparison of both models
# ===============================
#
# Here we'll compare the results from both models to highlight their 
# differences and suitability for different scenarios.

# Plot the subsidence profiles from both models
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

# Plot results from disc model
ax.plot(x_cross, y_true_cross, 'k-', label='True (Disc Model)')
ax.plot(x_cross, y_post_mean_disc, 'b-', label='Posterior Mean (Disc Model)')

# Plot results from grid model
ax.plot(x_cross, y_true_cross_grid, 'k--', label='True (Grid Model)')
ax.plot(x_cross, y_post_mean_grid, 'g-', label='Posterior Mean (Grid Model)')

# Plot observations
ax.plot(x_cross, y_obs_cross, 'ro', alpha=0.5, label='Disc Model Observations')
ax.plot(x_cross, y_obs_cross_grid, 'mo', alpha=0.5, label='Grid Model Observations')

ax.set_title("Comparison of Disc and Grid Models")
ax.set_xlabel("x (m)")
ax.set_ylabel("Subsidence (mm)")
ax.legend()
ax.grid(True)
ax.invert_yaxis()  # Subsidence is downward

###############################################################################
# Conclusions
# ----------
#
# This example demonstrated the following:
#
# 1. Using two different Geertsma models to calculate surface subsidence:
#    - Simple disc model for uniform, circular reservoirs
#    - Full grid model for complex, heterogeneous reservoirs
#
# 2. Applying ESMDA to both models to invert for reservoir pressure from 
#    surface subsidence measurements
#
# 3. Key differences between the models:
#    - The disc model is simpler but can only estimate a single uniform pressure
#    - The grid model can capture spatial variations in pressure but requires
#      more observations to constrain the solution
#
# 4. Performance considerations:
#    - The disc model is computationally more efficient
#    - The grid model offers more flexibility for complex reservoirs
#    - Both models can be effectively used with ESMDA
#
# When to use each model:
# - Use the disc model for simple, relatively homogeneous reservoirs or when
#   computational efficiency is paramount
# - Use the grid model for complex reservoirs with heterogeneous pressure
#   changes or when detailed spatial resolution is required
#
# The Geertsma models combined with ESMDA provide powerful tools for monitoring 
# and managing geomechanical effects in various subsurface operations, including
# oil and gas production, geothermal energy extraction, and CO2 storage.

dageo.Report()