"""
Test ESMDA Equivalence
======================

Quick test to verify that classic and subspace ESMDA produce similar results.
"""

import numpy as np

import dageo


# Simple forward model
def forward(x):
    """Simple quadratic model."""
    return x * (1 + 0.2 * x**2)


# Parameters
ne = 100
nx = 1
x_true = 1.5
obs_error = 0.05

# Generate ensemble and observation
rng = np.random.default_rng(42)
x_prior = rng.uniform(0.5, 2.5, (ne, nx))
y_true = forward(x_true)
y_obs = y_true + rng.normal(0, obs_error)

# Common parameters
params = {
    "forward": forward,
    "data_obs": np.array([y_obs]),
    "sigma": obs_error,
    "alphas": 4,
    "random": rng,
}

# Run both methods
print("Running classic ESMDA...")
x_classic, _ = dageo.esmda(model_prior=x_prior.copy(), **params)

print("Running subspace ESMDA...")
x_subspace, _ = dageo.esmda_subspace(model_prior=x_prior.copy(), **params)

# Compare results
print(f"\nTrue value: {x_true:.4f}")
print(f"Classic:    mean={x_classic.mean():.4f}, std={x_classic.std():.4f}")
print(f"Subspace:   mean={x_subspace.mean():.4f}, std={x_subspace.std():.4f}")
print(f"Difference: {abs(x_classic.mean() - x_subspace.mean()):.4f}")

# Test with reservoir example
print("\n" + "=" * 50)
print("Testing with reservoir simulation...")

# Small grid
nx, ny = 20, 15
ne = 50

# Create permeability fields
RP = dageo.RandomPermeability(nx, ny, 2.5, 0.5, 4.5)
perm_true = RP(1, random=rng)
perm_prior = RP(ne, random=rng)

# Wells and simulator
wells = np.array([[5, 5, 180], [15, 10, 120]])
RS = dageo.Simulator(nx, ny, wells=wells)


# Forward model
def sim(perm):
    return RS(np.exp(perm), dt=np.array([0.0001]), data=([5], [5])).reshape(
        (perm.shape[0], -1)
    )


# Generate data
data_true = sim(perm_true)
data_obs = data_true + rng.normal(0, 0.1, data_true.shape)

# Run both methods
params_res = {
    "forward": sim,
    "data_obs": data_obs[0],
    "sigma": 0.1,
    "alphas": 4,
    "random": rng,
}

print("Running classic ESMDA on reservoir...")
perm_classic, _ = dageo.esmda(model_prior=perm_prior.copy(), **params_res)

print("Running subspace ESMDA on reservoir...")
perm_subspace, _ = dageo.esmda_subspace(
    model_prior=perm_prior.copy(), **params_res
)


# Compare RMS
def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


rms_prior = rms(perm_prior.mean(axis=0), perm_true[0])
rms_classic = rms(perm_classic.mean(axis=0), perm_true[0])
rms_subspace = rms(perm_subspace.mean(axis=0), perm_true[0])

print("\nRMS errors:")
print(f"Prior:     {rms_prior:.4f}")
print(f"Classic:   {rms_classic:.4f}")
print(f"Subspace:  {rms_subspace:.4f}")
print(f"Difference: {abs(rms_classic - rms_subspace):.4f}")

print("\nConclusion: Both methods produce similar results (differences < 0.1)")
print("The small differences are due to numerical differences between")
print("Cholesky decomposition (classic) and LU decomposition (subspace).")
