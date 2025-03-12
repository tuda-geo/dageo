#%%
r"""
Particle Filter with Geertsma model
==================================

This example demonstrates how to use Particle Filter for data assimilation
with the Geertsma model to estimate reservoir pressure from subsidence observations.

The Particle Filter is a sequential Monte Carlo method that represents the
posterior distribution using a set of weighted particles (samples). It consists
of prediction, update, and resampling steps, and is particularly effective for
non-linear problems and non-Gaussian distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import dageo

# For reproducibility, we instantiate a random number generator with a fixed
# seed. For production, remove the seed!
rng = np.random.default_rng(7890)

# sphinx_gallery_thumbnail_number = 4

###############################################################################
# Problem Setup
# ------------
#
# We'll estimate reservoir pressure from subsidence observations using the
# basic Geertsma disc model.

# Known model parameters
depth = 2000.0       # Reservoir depth: 2000 m
radius = 1000.0      # Reservoir radius: 1000 m
thickness = 50.0     # Reservoir thickness: 50 m
cm = 1.0e-9          # Compaction coefficient: 1e-9 1/Pa
nu = 0.25            # Poisson's ratio: 0.25
p0 = 20.0e6          # Initial pressure: 20 MPa

# True pressure and observation parameters
p_true = np.ones((1, 1)) * 15.0e6  # 15 MPa (pressure drop from initial 20 MPa)
dstd = 0.001                        # Observation noise std: 1 mm

# Observation grid
nobs = 21                           # 21x21 observation grid
obs_range = 3000.0                  # Observation extent: 3000 m

# Create observation points
X = np.linspace(-obs_range, obs_range, nobs)
Y = np.linspace(-obs_range, obs_range, nobs)
X_grid, Y_grid = np.meshgrid(X, Y)
obs_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

###############################################################################
# Generate synthetic observations
# ----------------------------
#
# Create synthetic subsidence observations from the true pressure field.

# Initialize the Geertsma model
geertsma_model = dageo.Geertsma(
    depth=depth,
    radius=radius,
    thickness=thickness,
    cm=cm,
    nu=nu,
    p0=p0,
    obs_points=obs_points
)

# Generate true subsidence
subsidence_true = geertsma_model(p_true)

# Add random noise to create synthetic observations
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
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')

###############################################################################
# Create prior pressure ensemble
# --------------------------
#
# Generate a prior ensemble of pressure values. For the particle filter,
# we'll create a larger ensemble (particles) than we would typically use for ESMDA.

# Define pressure bounds
p_min = 10.0e6  # 10 MPa
p_max = 25.0e6  # 25 MPa

# Number of particles
n_particles = 5000

# Generate prior ensemble - uniformly distributed pressures
p_prior = rng.uniform(p_min, p_max, size=(n_particles, 1, 1))

# Define the forward model for the Particle Filter
def forward_model(pressures):
    """Forward model for the Particle Filter.
    
    Parameters
    ----------
    pressures : ndarray
        Pressure ensemble of shape (n_particles, 1, 1)
        
    Returns
    -------
    subsidence : ndarray
        Predicted subsidence for each pressure value
    """
    num_particles = pressures.shape[0]  # Number of particles
    nobs_points = obs_points.shape[0]  # Number of observation points
    subsidence = np.zeros((num_particles, nobs_points))
    
    # Calculate subsidence for each pressure
    for i in range(num_particles):
        subsidence[i, :] = geertsma_model(pressures[i])
    
    return subsidence

# Calculate prior subsidence predictions for later comparison
prior_subsidence = forward_model(p_prior)

###############################################################################
# Apply Particle Filter
# ------------------
#
# Apply the Particle Filter algorithm to estimate the pressure.
# Unlike ESMDA, the particle filter naturally handles sequential assimilation
# of observations. For demonstration, we'll run multiple steps even though
# we have only one observation dataset.

def restrict_pressure(x):
    """Restrict possible pressures to the defined range."""
    np.clip(x, p_min, p_max, out=x)

# Run multiple steps of the Particle Filter to demonstrate its sequential nature
# In a real-world application, these might be time-varying observations
n_steps = 3

# Run Particle Filter with multiple assimilation steps
p_post, subsidence_post, weights, p_steps, subsidence_steps, weights_steps = dageo.particle_filter(
    model_prior=p_prior,
    forward=forward_model,
    data_obs=subsidence_obs,  # Same observation for each step in this example
    sigma=dstd,
    n_steps=n_steps,
    resampling_threshold=0.5,  # Resample when effective sample size < 50%
    callback_post=restrict_pressure,
    return_weights=True,
    return_steps=True,
    random=rng,
)

###############################################################################
# Analyze results
# ------------
#
# Compare the prior and posterior pressure distributions.

# Calculate weighted statistics
p_prior_mean = np.mean(p_prior)
p_prior_std = np.std(p_prior)

# For posterior, use weights for statistics
p_post_mean = np.sum(p_post[:, 0, 0] * weights)
p_post_weighted_var = np.sum(weights * (p_post[:, 0, 0] - p_post_mean)**2)
p_post_std = np.sqrt(p_post_weighted_var)

# Convert to MPa for reporting
p_prior_mean_MPa = p_prior_mean / 1e6
p_prior_std_MPa = p_prior_std / 1e6
p_post_mean_MPa = p_post_mean / 1e6
p_post_std_MPa = p_post_std / 1e6
p_true_MPa = p_true[0, 0] / 1e6

# Calculate error and uncertainty reduction
rel_error_prior = abs(p_prior_mean - p_true[0, 0]) / p_true[0, 0] * 100
rel_error_post = abs(p_post_mean - p_true[0, 0]) / p_true[0, 0] * 100
error_reduction = (rel_error_prior - rel_error_post) / rel_error_prior * 100
uncertainty_reduction = (1 - p_post_std / p_prior_std) * 100

print("Pressure Estimation Results:")
print(f"True pressure: {p_true_MPa:.2f} MPa")
print(f"Prior: {p_prior_mean_MPa:.2f} ± {p_prior_std_MPa:.2f} MPa")
print(f"Posterior: {p_post_mean_MPa:.2f} ± {p_post_std_MPa:.2f} MPa")
print(f"Error reduction: {error_reduction:.1f}%")
print(f"Uncertainty reduction: {uncertainty_reduction:.1f}%")

###############################################################################
# Visualize the evolution of the pressure distribution
# ------------------------------------------------
#
# Plot how the pressure distribution evolves through the Particle Filter steps.

# Plot histograms of pressure distribution for each step
fig, axs = plt.subplots(1, n_steps+1, figsize=(15, 5), constrained_layout=True, sharey=True)

# Prior distribution
axs[0].hist(p_steps[0, :, 0, 0]/1e6, bins=30, alpha=0.7, color='blue', 
           weights=weights_steps[0], density=True)
axs[0].axvline(x=p_true_MPa, color='red', linestyle='--')
axs[0].set_title("Prior Distribution")
axs[0].set_xlabel("Pressure (MPa)")
axs[0].set_ylabel("Density")
axs[0].text(0.05, 0.95, f'True: {p_true_MPa:.1f} MPa', transform=axs[0].transAxes,
          verticalalignment='top', color='red')

# Distribution after each step
for i in range(n_steps):
    axs[i+1].hist(p_steps[i+1, :, 0, 0]/1e6, bins=30, alpha=0.7, color='green', 
                 weights=weights_steps[i+1], density=True)
    axs[i+1].axvline(x=p_true_MPa, color='red', linestyle='--')
    axs[i+1].set_title(f"After Step {i+1}")
    axs[i+1].set_xlabel("Pressure (MPa)")
    # Calculate weighted mean for this step
    step_mean = np.sum(p_steps[i+1, :, 0, 0] * weights_steps[i+1]) / 1e6
    axs[i+1].text(0.05, 0.95, f'Mean: {step_mean:.1f} MPa', transform=axs[i+1].transAxes,
              verticalalignment='top', color='green')

###############################################################################
# Compare subsidence predictions
# --------------------------
#
# Compare the observed subsidence with the posterior predictions.

# Calculate weighted mean subsidence
subsidence_post_mean = np.zeros(subsidence_post.shape[1])
for i in range(n_particles):
    subsidence_post_mean += weights[i] * subsidence_post[i]

# Reshape for plotting
Z_post_mean = subsidence_post_mean.reshape(nobs, nobs)

# Plot subsidence comparison
fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# Observed subsidence
im1 = axs[0].pcolormesh(X_grid, Y_grid, Z_obs*1000, cmap='RdBu_r', shading='auto')
axs[0].set_title("Observed Subsidence")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")

# True subsidence
im2 = axs[1].pcolormesh(X_grid, Y_grid, Z_true*1000, cmap='RdBu_r', shading='auto')
axs[1].set_title("True Subsidence")
axs[1].set_xlabel("x (m)")

# Posterior mean subsidence
im3 = axs[2].pcolormesh(X_grid, Y_grid, Z_post_mean*1000, cmap='RdBu_r', shading='auto')
axs[2].set_title("Posterior Mean Subsidence")
axs[2].set_xlabel("x (m)")

fig.colorbar(im1, ax=axs, label="Subsidence (mm)")

# Draw circle to represent the reservoir extent
for ax in axs:
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color='k', linestyle='--'))
    ax.set_aspect('equal')

###############################################################################
# Cross-section comparison
# ---------------------
#
# Compare cross-sections of the subsidence profiles.

# Extract cross-section at y=0
mid_idx = nobs // 2
x_cross = X
y_true_cross = Z_true[mid_idx, :] * 1000  # Convert to mm
y_obs_cross = Z_obs[mid_idx, :] * 1000
y_post_mean_cross = Z_post_mean[mid_idx, :] * 1000

# Extract sample of posterior realizations
n_samples = 30
sample_indices = rng.choice(n_particles, size=n_samples, p=weights)
y_post_samples = np.zeros((n_samples, nobs))
for i, idx in enumerate(sample_indices):
    y_post_samples[i, :] = subsidence_post[idx].reshape(nobs, nobs)[mid_idx, :] * 1000

# Calculate prior subsidence mean and std along the cross-section
y_prior_mean = np.zeros(nobs)
y_prior_std = np.zeros(nobs)

for i in range(nobs):
    subsidence_values = [prior_subsidence[j].reshape(nobs, nobs)[mid_idx, i] * 1000 
                         for j in range(n_particles)]
    y_prior_mean[i] = np.mean(subsidence_values)
    y_prior_std[i] = np.std(subsidence_values)

# Plot cross-section comparison
fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

# Plot prior uncertainty range
ax.fill_between(x_cross, y_prior_mean - y_prior_std, y_prior_mean + y_prior_std,
               color='blue', alpha=0.2, label='Prior ±σ')

# Plot sample of posterior realizations
for i in range(n_samples):
    ax.plot(x_cross, y_post_samples[i, :], color='lightgreen', alpha=0.2)

# Plot observed, true, and mean curves
ax.plot(x_cross, y_obs_cross, 'ro', markersize=6, label='Observations')
ax.plot(x_cross, y_true_cross, 'k-', linewidth=2, label='True')
ax.plot(x_cross, y_prior_mean, 'b-', linewidth=2, label='Prior Mean')
ax.plot(x_cross, y_post_mean_cross, 'g-', linewidth=2, label='Posterior Mean')

ax.set_title("Cross-section of Subsidence at y=0", fontsize=14)
ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("Subsidence (mm)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Invert y-axis as subsidence is downward
ax.invert_yaxis()

# Add annotations
rmse_prior = np.sqrt(np.mean((y_prior_mean - y_true_cross)**2))
rmse_post = np.sqrt(np.mean((y_post_mean_cross - y_true_cross)**2))
improvement = (1 - rmse_post/rmse_prior) * 100

ax.text(0.02, 0.02, 
       f"Prior RMSE: {rmse_prior:.2f} mm\nPosterior RMSE: {rmse_post:.2f} mm\nImprovement: {improvement:.1f}%", 
       transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

###############################################################################
# Compare Particle Weights
# ---------------------
#
# Visualize the particle weights at each step of the filter.

# Plot particle weights
fig, axs = plt.subplots(1, n_steps+1, figsize=(15, 5), constrained_layout=True, sharey=True)

# Number of particles to show in the visualization
max_particles_to_show = 50
particles_to_show = min(max_particles_to_show, n_particles)

# Uniform initial weights
# For initial weights, they should all be 1/n_particles
initial_weight = 1.0 / n_particles
axs[0].bar(np.arange(particles_to_show), np.ones(particles_to_show) * initial_weight, 
          color='blue', alpha=0.7)
axs[0].set_title("Initial Weights (Uniform)")
axs[0].set_xlabel("Particle Index")
axs[0].set_ylabel("Weight")
axs[0].set_ylim(0, max(0.1, initial_weight * 1.5))  # Set y limit to show bars clearly

# Weights after each step
for i in range(n_steps):
    # Get weights for this step
    step_weights = weights_steps[i+1, :particles_to_show]
    
    # Plot the weights
    axs[i+1].bar(np.arange(particles_to_show), step_weights, color='green', alpha=0.7)
    axs[i+1].set_title(f"Weights After Step {i+1}")
    axs[i+1].set_xlabel("Particle Index")
    
    # Calculate effective sample size
    n_eff = 1.0 / np.sum(weights_steps[i+1]**2)
    n_eff_ratio = n_eff / n_particles
    axs[i+1].text(0.05, 0.95, f'Effective N: {n_eff:.1f}\n({n_eff_ratio:.1%} of total)',
                transform=axs[i+1].transAxes, verticalalignment='top')
    
    # Adjust y-axis to show variation in weights
    max_weight = np.max(step_weights)
    if max_weight > 0:
        axs[i+1].set_ylim(0, min(0.1, max_weight * 1.5))

###############################################################################
# Comparing Multiple Runs
# -------------------
#
# Run the particle filter with different resampling thresholds and compare results.

# Define different resampling thresholds
thresholds = [0.2, 0.5, 0.8]
results = []

# Run particle filter with each threshold
for threshold in thresholds:
    print(f"\nRunning with resampling threshold: {threshold}")
    
    p_post, subsidence_post, weights = dageo.particle_filter(
        model_prior=p_prior,
        forward=forward_model,
        data_obs=subsidence_obs,
        sigma=dstd,
        n_steps=n_steps,
        resampling_threshold=threshold,
        callback_post=restrict_pressure,
        return_weights=True,
        random=rng,
    )
    
    # Calculate weighted mean and std
    p_mean = np.sum(p_post[:, 0, 0] * weights) / 1e6
    p_var = np.sum(weights * ((p_post[:, 0, 0] / 1e6) - p_mean)**2)
    p_std = np.sqrt(p_var)
    
    # Calculate effective sample size
    n_eff = 1.0 / np.sum(weights**2)
    n_eff_ratio = n_eff / n_particles
    
    results.append({
        'threshold': threshold,
        'mean': p_mean,
        'std': p_std,
        'n_eff': n_eff,
        'n_eff_ratio': n_eff_ratio
    })
    
    print(f"  Posterior Mean: {p_mean:.2f} ± {p_std:.2f} MPa")
    print(f"  Effective Sample Size: {n_eff:.1f} ({n_eff_ratio:.1%} of total)")

# Plot comparison of results
fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

# Create bar positions
bar_positions = np.arange(len(thresholds))
bar_width = 0.35

# Plot prior mean for reference
prior_bar = ax.bar(bar_positions - bar_width/2, [p_prior_mean_MPa] * len(thresholds), 
                  bar_width, alpha=0.5, color='blue', label='Prior Mean')

# Plot posterior means
posterior_bar = ax.bar(bar_positions + bar_width/2, [r['mean'] for r in results], 
                      bar_width, yerr=[r['std'] for r in results], 
                      label='Posterior Mean', color='green', alpha=0.7)

# Add horizontal line for true value
ax.axhline(y=p_true_MPa, color='red', linestyle='--', 
          label=f'True ({p_true_MPa:.1f} MPa)')

# Add prior error bars
ax.errorbar(bar_positions - bar_width/2, [p_prior_mean_MPa] * len(thresholds), 
           yerr=p_prior_std_MPa, fmt='none', color='blue', capsize=5)

# Add effective sample size as text above each bar
for i, result in enumerate(results):
    ax.text(i + bar_width/2, result['mean'] + result['std'] + 0.5, 
           f"N_eff: {result['n_eff']:.1f}\n({result['n_eff_ratio']:.1%})",
           ha='center', va='bottom')

# Add text for prior
ax.text(bar_positions[0] - bar_width/2, p_prior_mean_MPa + p_prior_std_MPa + 0.5,
       f"Prior: {p_prior_mean_MPa:.1f} ± {p_prior_std_MPa:.1f}",
       ha='center', va='bottom', color='blue')

# Formatting
ax.set_ylabel('Pressure (MPa)')
ax.set_title('Effect of Resampling Threshold on Pressure Estimation')
ax.set_xticks(bar_positions)
ax.set_xticklabels([f"Threshold: {t}" for t in thresholds])
ax.legend(loc='lower right')

# Set y-axis limits to show both prior and posterior clearly
y_min = min(p_true_MPa - 2, p_prior_mean_MPa - p_prior_std_MPa - 1)
y_max = max(p_true_MPa + 2, p_prior_mean_MPa + p_prior_std_MPa + 2)
ax.set_ylim(y_min, y_max)

# Add grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.3)

###############################################################################
# Conclusion
# --------
#
# This example demonstrated using the Particle Filter for data assimilation with
# the Geertsma model to estimate reservoir pressure from subsidence observations.
# Key findings include:
#
# 1. The Particle Filter effectively estimates reservoir pressure, similar to ESMDA,
#    but with different characteristics:
#    - It naturally handles non-Gaussian distributions
#    - It provides weighted particles rather than ensemble members
#    - It includes an explicit resampling step to avoid degeneracy
#
# 2. The resampling threshold is an important parameter that affects:
#    - The diversity of the posterior particles
#    - The effective sample size
#    - The accuracy and uncertainty of the estimates
#
# 3. The sequential nature of the Particle Filter makes it well-suited for
#    problems with time-evolving observations.
#
# The Particle Filter and ESMDA are complementary methods that can be chosen
# based on the specific characteristics of the problem and data.

dageo.Report()