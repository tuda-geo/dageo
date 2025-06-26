# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from dageo.geertsma import TorchGeertsma, TorchGeertsmaFullGrid
from dageo.rml import rml

"""
Geertsma Parameter Estimation using RML with TorchGeertsma and
TorchGeertsmaFullGrid
=====================================================================================

This example demonstrates how to use the Randomized Maximum Likelihood (RML)
method to perform parameter estimation on differentiable versions of
Geertsma's model using PyTorch.

Two experiments are shown:
1. Single-Point Inversion: Using TorchGeertsma (which computes the
   subsidence at the reservoir center).
2. Full-Grid Inversion: Using TorchGeertsmaFullGrid (which computes spatial
   subsidence at a set of observation points).

In both cases, an ensemble of prior model guesses is provided and
independent inversions are performed.
Model parameters to be estimated:
    - Depth
    - Radius
    - Thickness
    - Compaction coefficient (cm)
    - Poisson's ratio (nu)

A vector prior_std is provided to properly scale the prior regularization term.
"""

# --------------------------
# Single-Point Inversion using TorchGeertsma
# --------------------------
print("=== Single-Point Inversion using TorchGeertsma ===")
torch.manual_seed(123)
np.random.seed(123)

# True model parameters
depth_true = 2000.0  # meters
radius_true = 1000.0  # meters
thickness_true = 50.0  # meters
cm_true = 1e-9  # 1/Pa
nu_true = 0.25  # unitless
p0 = 20e6  # Pa, initial pressure
p = 15e6  # Pa, current pressure (pressure drop)

# Create true model and compute observed subsidence at reservoir center
true_model = TorchGeertsma(
    depth_true, radius_true, thickness_true, cm_true, nu_true, p0, p
)
subsidence_true = true_model()  # tensor of shape (1,)
subsidence_true_val = subsidence_true.item()
print(f"True subsidence (m): {subsidence_true_val:.6e}")

# Add noise to generate observed data (scalar)
noise_std = 1e-6  # meters
noise = noise_std * np.random.randn()
Data_obs = subsidence_true_val + noise
print(f"Observed subsidence (m): {Data_obs:.6e}")


# Define differentiable forward model for single-point inversion
# (m is a tensor of shape (5,) representing [depth, radius, thickness, cm, nu])
def forward_model_single(m):
    depth = m[0]
    radius = m[1]
    thickness = m[2]
    cm = m[3]
    nu = m[4]
    model = TorchGeertsma(depth, radius, thickness, cm, nu, p0, p)
    return model()  # returns tensor of shape (1,)


# Create an ensemble of prior guesses (ensemble size = 10)
ensemble_size = 10
base_prior = np.array([2100.0, 900.0, 60.0, 1.2e-9, 0.28])
# Add some random noise to the base prior for each ensemble member
m_prior_ensemble = base_prior + np.random.randn(ensemble_size, 5) * np.array(
    [50.0, 50.0, 5.0, 1e-10, 0.02]
)

# Define a vector prior standard deviation to scale the regularization term
prior_std_single = np.array([100.0, 100.0, 5.0, 1e-10, 0.02])

# Run ensemble RML inversion (single-point)
m_est_single, obj_hist_single = rml(
    forward_model_single,
    Data_obs,
    noise_std,
    m_prior_ensemble,
    n_iter=1500,
    lr=1e-2,
    prior_std=prior_std_single,
    verbose=True,
    random_seed=123,
)

print("\nEstimated model parameters (Single-Point):")
param_names = [
    "Depth (m)",
    "Radius (m)",
    "Thickness (m)",
    "Compaction coef. (1/Pa)",
    "Poisson ratio",
]
for i, m_est in enumerate(m_est_single.cpu().numpy()):
    print(f"Member {i + 1}:")
    for name, val in zip(param_names, m_est):
        print(f"  {name}: {val:.6e}")

plt.figure(figsize=(8, 5))
if isinstance(obj_hist_single, list):
    # For ensemble, plot average objective history
    avg_obj = np.mean(np.array(obj_hist_single), axis=0)
    plt.plot(avg_obj, "b-")
    plt.title("Objective Function History (Average, Single-Point)")
else:
    plt.plot(obj_hist_single, "b-")
    plt.title("Objective Function History (Single-Point)")
plt.xlabel("Iteration")
plt.ylabel("Objective Loss")
plt.grid(True)
plt.show()

# --------------------------
# Full-Grid Inversion using TorchGeertsmaFullGrid
# --------------------------
print("\n=== Full-Grid Inversion using TorchGeertsmaFullGrid ===")
torch.manual_seed(456)
np.random.seed(456)

# Full-grid true model parameters (using the same true parameters)
nx, ny = 10, 10
dx, dy = 100.0, 100.0

# Create a true full-grid model and compute subsidence over a grid of
# observation points
true_full_model = TorchGeertsmaFullGrid(
    nx,
    ny,
    depth_true,
    dx,
    dy,
    thickness_true,
    cm_true,
    nu_true,
    p0,
    p,
    npoints=20,
)
subsidence_full_true = true_full_model()  # tensor of shape (n_obs,)
print(
    f"True full-grid subsidence: min={subsidence_full_true.min().item():.6e}, "
    f"max={subsidence_full_true.max().item():.6e}")

# Add noise to generate observed full-grid data
noise_std_full = 1e-6
noise_full = noise_std_full * np.random.randn(subsidence_full_true.shape[0])
Data_obs_full = subsidence_full_true.cpu().numpy() + noise_full


# Define differentiable forward model for full-grid inversion
# (m is a tensor of shape (5,) representing the model parameters)
def forward_model_full(m):
    depth = m[0]
    # radius = m[1]  # Not used in TorchGeertsmaFullGrid
    thickness = m[2]
    cm = m[3]
    nu = m[4]
    model = TorchGeertsmaFullGrid(
        nx, ny, depth, dx, dy, thickness, cm, nu, p0, p, npoints=20
    )
    return model()  # returns tensor of shape (n_obs,)


# Create an ensemble of prior guesses for full-grid inversion (ensemble
# size = 10)
m_prior_ensemble_full = base_prior + np.random.randn(
    ensemble_size, 5
) * np.array([50.0, 50.0, 5.0, 1e-10, 0.02])

# Define a vector prior standard deviation for full-grid inversion
prior_std_full = np.array([100.0, 100.0, 5.0, 1e-10, 0.02])

# Run ensemble RML inversion for full-grid
m_est_full, obj_hist_full = rml(
    forward_model_full,
    Data_obs_full,
    noise_std_full,
    m_prior_ensemble_full,
    n_iter=1500,
    lr=1e-2,
    prior_std=prior_std_full,
    verbose=True,
    random_seed=456,
)

print("\nEstimated model parameters (Full-Grid):")
for i, m_est in enumerate(m_est_full.cpu().numpy()):
    print(f"Member {i + 1}:")
    for name, val in zip(param_names, m_est):
        print(f"  {name}: {val:.6e}")

plt.figure(figsize=(8, 5))
if isinstance(obj_hist_full, list):
    avg_obj_full = np.mean(np.array(obj_hist_full), axis=0)
    plt.plot(avg_obj_full, "r-")
    plt.title("Objective Function History (Average, Full-Grid)")
else:
    plt.plot(obj_hist_full, "r-")
    plt.title("Objective Function History (Full-Grid)")
plt.xlabel("Iteration")
plt.ylabel("Objective Loss")
plt.grid(True)
plt.show()
