#%%
"""
Particle Filter Example for the Basic Reservoir Model
========================================================

This example demonstrates how to use the Particle Filter algorithm for data
assimilation in a 2D reservoir model. We simulate pressure at a specific well
location based on a random ensemble of log-permeability fields. The true
pressure observation is generated from a "true" reservoir model, and the
Particle Filter updates the prior ensemble to obtain a posterior ensemble.

Note: The log permeability fields are generated using dageo.RandomPermeability,
and the Simulator from dageo is used to simulate pressure time series.
"""

import numpy as np
import matplotlib.pyplot as plt
import dageo

# Set seed for reproducibility
rng = np.random.default_rng(1234)

###############################################################################
# Model Setup
# ------------
nx = 30
ny = 25

# Permeability parameters (log permeability)
perm_mean = 3.0   # mean log permeability
perm_min = 0.5    # minimum log permeability
perm_max = 5.0    # maximum log permeability

# Time stepping parameters for the simulator
dt = np.full(5, 0.0002)              # 5 time steps
time = np.r_[0, np.cumsum(dt)]         # simulation time

# Observation location (well indices)
ox, oy = 1, 1

# Measurement noise standard deviation [bar]
dstd = 2.0

# Number of particles
n_particles = 1000                        # reduced number of particles for better diversity

# Create Random Permeability generator and generate true and prior fields
RP = dageo.RandomPermeability(nx, ny, perm_mean, perm_min, perm_max)
perm_true = RP(1, random=rng)            # true log permeability field (shape: (1, nx, ny))
perm_prior = RP(n_particles, random=rng)  # prior ensemble of log permeability fields (shape: (n_particles, nx, ny))

###############################################################################
# Reservoir Simulation Setup
# --------------------------
# Instantiate the Reservoir Simulator
RS = dageo.Simulator(nx, ny, wells=None)

# Define simulation function that returns the pressure time series at well (ox, oy)
def sim(x):
    """Simulate pressure at well (ox, oy) for a given log permeability field x.
    Parameters:
      x : ndarray of shape (nx, ny)
    Returns:
      pressure time series : ndarray of shape (nt,), where nt = number of time points
    """
    perm = np.exp(x)  # convert log permeability to permeability
    return RS(perm, dt=dt, data=(ox, oy))

# Generate the true pressure time series
true_pressure_series = sim(perm_true[0])

# Create synthetic observations for all time steps
data_obs = true_pressure_series + rng.normal(0, dstd, size=len(true_pressure_series))

###############################################################################
# Callback to Restrict Log Permeability
# --------------------------------------
# Ensure the log permeability values remain within [perm_min, perm_max]

def restrict_permeability(x):
    np.clip(x, perm_min, perm_max, out=x)

###############################################################################
# Run Particle Filter
# -------------------
# Run Particle Filter with multiple steps
n_steps = len(time) - 1  # Use all time steps

p_post, pressure_post, weights = dageo.particle_filter(
    model_prior=perm_prior,
    forward=sim,
    data_obs=data_obs,
    sigma=dstd,
    n_steps=n_steps,
    resampling_threshold=0.3,  
    callback_post=restrict_permeability,
    return_weights=True,
    random=rng
)

###############################################################################
# Analysis of Results
# -------------------
# Compute the predicted pressures for the prior and posterior ensembles
pred_prior = sim(perm_prior)
pred_post = sim(p_post)

# Compute statistics for predicted pressure
prior_mean = np.mean(pred_prior[:, -1])
prior_std = np.std(pred_prior[:, -1])
post_mean = np.sum(pred_post[:, -1] * weights)
post_var = np.sum(weights * (pred_post[:, -1] - post_mean)**2)
post_std = np.sqrt(post_var)

print("Pressure Estimation Results:")
print(f"True Pressure: {true_pressure_series[-1]:.2f} bar")
print(f"Prior Mean Pressure: {prior_mean:.2f} ± {prior_std:.2f} bar")
print(f"Posterior Mean Pressure: {post_mean:.2f} ± {post_std:.2f} bar")

###############################################################################
# Visualization
# -------------
# Plot histograms of predicted pressures from the prior and posterior ensembles
fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

ax[0].hist(pred_prior[:, -1], bins=30, alpha=0.7, color='blue', density=True)
ax[0].axvline(x=true_pressure_series[-1], color='red', linestyle='--', label="True Pressure")
ax[0].set_title("Prior Predicted Pressure")
ax[0].set_xlabel("Pressure (bar)")
ax[0].set_ylabel("Density")
ax[0].legend()

ax[1].hist(pred_post[:, -1], bins=30, alpha=0.7, color='green', density=True)
ax[1].axvline(x=true_pressure_series[-1], color='red', linestyle='--', label="True Pressure")
ax[1].set_title("Posterior Predicted Pressure")
ax[1].set_xlabel("Pressure (bar)")
ax[1].legend()

plt.show()

###############################################################################
# Additional Visualizations
# -------------

# Scatter plot: Mean Log Permeability vs Predicted Pressure
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))

# Compute mean log permeability for each ensemble
x_prior = np.mean(perm_prior, axis=(1, 2))
x_post = np.mean(p_post, axis=(1, 2))

y_prior = pred_prior[:, -1]
y_post = pred_post[:, -1]

ax2[0].scatter(x_prior, y_prior, alpha=0.5, color='blue', label='Prior')
ax2[0].set_title('Prior: Mean Log Permeability vs Predicted Pressure')
ax2[0].set_xlabel('Mean Log Permeability')
ax2[0].set_ylabel('Predicted Pressure (bar)')
ax2[0].legend()

ax2[1].scatter(x_post, y_post, alpha=0.5, color='green', label='Posterior')
ax2[1].set_title('Posterior: Mean Log Permeability vs Predicted Pressure')
ax2[1].set_xlabel('Mean Log Permeability')
ax2[1].set_ylabel('Predicted Pressure (bar)')
ax2[1].legend()

plt.show()

# Boxplot of Predicted Pressures for Prior and Posterior
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.boxplot([pred_prior[:, -1], pred_post[:, -1]], labels=['Prior', 'Posterior'])
ax3.set_title('Boxplot of Predicted Pressures')
ax3.set_ylabel('Pressure (bar)')
plt.show()

# Histogram of Particle Weights (Posterior)
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.hist(weights, bins=30, color='purple', alpha=0.7)
ax4.set_title('Histogram of Particle Weights (Posterior)')
ax4.set_xlabel('Weight')
ax4.set_ylabel('Count')
plt.show()

# Cumulative Distribution Function (CDF) of Predicted Pressures

def plot_cdf(data, label, ax):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    ax.plot(sorted_data, cdf, label=label)

fig5, ax5 = plt.subplots(figsize=(8,6))
plot_cdf(pred_prior[:, -1], 'Prior CDF', ax5)
plot_cdf(pred_post[:, -1], 'Posterior CDF', ax5)
ax5.set_title('CDF of Predicted Pressures')
ax5.set_xlabel('Pressure (bar)')
ax5.set_ylabel('Cumulative Probability')
ax5.legend()
plt.show()

# Add time series plot
fig7, ax7 = plt.subplots(figsize=(10, 6))
# Plot prior predictions
for i in range(n_particles):
    ax7.plot(time, pred_prior[i], color='blue', alpha=0.1)
# Plot posterior predictions
for i in range(n_particles):
    ax7.plot(time, pred_post[i], color='green', alpha=0.1)
# Plot true pressure and observations
ax7.plot(time, true_pressure_series, 'k-', label='True', linewidth=2)
ax7.plot(time, data_obs, 'ro', label='Observations')
ax7.set_title('Pressure Time Series')
ax7.set_xlabel('Time')
ax7.set_ylabel('Pressure (bar)')
ax7.legend()
plt.show()

###############################################################################
# Report
###############################################################################
dageo.Report() 