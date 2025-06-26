r"""
Classic vs Subspace ESMDA Comparison
====================================

This example compares the classic ESMDA implementation (using Cholesky
decomposition) with the new subspace formulation (using LU decomposition).

Both methods should produce similar results for the non-localized case,
demonstrating that they are mathematically equivalent despite different
numerical approaches.

This reproduces the examples from Geir Evensen's talk on *Properties of
Iterative Ensemble Smoothers and Strategies for Conditioning on Production
Data* at IPAM in May 2017.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import dageo

# For reproducibility - we'll create separate RNG instances for each method
# to ensure they get the same random numbers
seed = 2024

# sphinx_gallery_thumbnail_number = 3

###############################################################################
# Forward model
# -------------
# We use the same simple model as in the original basicESMDA example:
# y = x(1 + β x²), which is linear if β=0


def forward(x, beta):
    """Simple model: y = x (1 + β x²) (linear if beta=0)."""
    return np.atleast_1d(x * (1 + beta * x**2))


# Visualize the forward model
fig, axs = plt.subplots(
    1, 2, figsize=(8, 3), sharex=True, constrained_layout=True
)
fig.suptitle("Forward Model:  y = x (1 + β x²)")
px = np.linspace(-5, 5, 301)
for i, b in enumerate([0.0, 0.2]):
    axs[i].set_title(f"{['Linear model', 'Nonlinear model'][i]}: β = {b}")
    axs[i].plot(px, forward(px, b))
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")

###############################################################################
# Plotting functions
# ------------------


def pseudopdf(data, bins=200, density=True, **kwargs):
    """Return the pdf from a simple bin count."""
    x, y = np.histogram(data, bins=bins, density=density, **kwargs)
    return (y[:-1] + y[1:]) / 2, x


def plot_result(mpost, dpost, dobs, title, method_name, ax1, ax2):
    """Plot results in the style of the original basicESMDA example."""
    # Plot Likelihood
    rng_plot = np.random.default_rng(42)
    ax2.plot(
        *pseudopdf(rng_plot.normal(dobs, size=(10000,))),
        "C2",
        lw=2,
        label="Datum",
    )

    # Plot steps
    na = mpost.shape[0] - 1
    for i in range(na + 1):
        params = {
            "color": "C0" if i == na else "C3",  # Last blue, rest red
            "lw": 2 if i in [0, na] else 1,  # First/last thick
            "alpha": 1 if i in [0, na] else i / na,  # start faint
            "label": ["Initial", *((na - 2) * ("",)), "MDA steps", "MDA"][i],
        }
        ax1.plot(*pseudopdf(mpost[i, :, 0], range=(-3, 5)), **params)
        ax2.plot(*pseudopdf(dpost[i, :, 0], range=(-5, 8)), **params)

    # Axis and labels
    ax1.set_title(f"{method_name} - Model Parameter Domain")
    ax1.set_xlabel("x")
    ax1.set_xlim([-3, 5])
    ax1.legend()
    ax2.set_title(f"{method_name} - Data Domain")
    ax2.set_xlabel("y")
    ax2.set_xlim([-5, 8])
    ax2.legend()


###############################################################################
# Linear case
# -----------
#
# Prior model parameters and ESMDA parameters

# Point of our "observation"
xlocation = -1.0

# Ensemble size
ne = int(1e4)

# Data standard deviation
obs_std = 1.0

# Prior: Let's start with ones as our prior guess
# Create prior using a fixed seed
rng_prior = np.random.default_rng(seed)
mprior = rng_prior.normal(loc=1.0, scale=obs_std, size=(ne, 1))

###############################################################################
# Run ESMDA and plot


def lin_fwd(x):
    """Linear forward model."""
    return forward(x, beta=0.0)


# Sample an "observation"
l_dobs = lin_fwd(xlocation)

# Run classic ESMDA with its own RNG
print("Running classic ESMDA for linear case...")
rng_classic = np.random.default_rng(seed + 1)
lm_post_classic, ld_post_classic = dageo.esmda(
    model_prior=mprior.copy(),
    forward=lin_fwd,
    data_obs=l_dobs,
    sigma=obs_std,
    alphas=10,
    return_steps=True,
    random=rng_classic,
)

# Run subspace ESMDA with its own RNG
print("Running subspace ESMDA for linear case...")
rng_subspace = np.random.default_rng(seed + 1)
lm_post_subspace, ld_post_subspace = dageo.esmda_subspace(
    model_prior=mprior.copy(),
    forward=lin_fwd,
    data_obs=l_dobs,
    sigma=obs_std,
    alphas=10,
    return_steps=True,
    random=rng_subspace,
)

###############################################################################
# Plot linear case results

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(12, 8), constrained_layout=True
)
fig.suptitle("Linear Case Comparison", fontsize=16)

# Plot Classic ESMDA
plot_result(
    lm_post_classic,
    ld_post_classic,
    l_dobs,
    "Linear Case",
    "Classic ESMDA",
    ax1,
    ax2,
)
ax1.set_ylim([0, 0.6])

# Plot Subspace ESMDA
plot_result(
    lm_post_subspace,
    ld_post_subspace,
    l_dobs,
    "Linear Case",
    "Subspace ESMDA",
    ax3,
    ax4,
)
ax3.set_ylim([0, 0.6])

###############################################################################
# Nonlinear case
# --------------


def nonlin_fwd(x):
    """Nonlinear forward model."""
    return forward(x, beta=0.2)


# Sample a nonlinear observation
n_dobs = nonlin_fwd(xlocation)

# Run classic ESMDA with its own RNG
print("\nRunning classic ESMDA for nonlinear case...")
rng_classic_nl = np.random.default_rng(seed + 2)
nm_post_classic, nd_post_classic = dageo.esmda(
    model_prior=mprior.copy(),
    forward=nonlin_fwd,
    data_obs=n_dobs,
    sigma=obs_std,
    alphas=10,
    return_steps=True,
    random=rng_classic_nl,
)

# Run subspace ESMDA with its own RNG
print("Running subspace ESMDA for nonlinear case...")
rng_subspace_nl = np.random.default_rng(seed + 2)
nm_post_subspace, nd_post_subspace = dageo.esmda_subspace(
    model_prior=mprior.copy(),
    forward=nonlin_fwd,
    data_obs=n_dobs,
    sigma=obs_std,
    alphas=10,
    return_steps=True,
    random=rng_subspace_nl,
)

###############################################################################
# Plot nonlinear case results

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(12, 8), constrained_layout=True
)
fig.suptitle("Nonlinear Case Comparison", fontsize=16)

# Plot Classic ESMDA
plot_result(
    nm_post_classic,
    nd_post_classic,
    n_dobs,
    "Nonlinear Case",
    "Classic ESMDA",
    ax1,
    ax2,
)
ax1.set_ylim([0, 0.7])

# Plot Subspace ESMDA
plot_result(
    nm_post_subspace,
    nd_post_subspace,
    n_dobs,
    "Nonlinear Case",
    "Subspace ESMDA",
    ax3,
    ax4,
)
ax3.set_ylim([0, 0.7])

###############################################################################
# Direct comparison plots
# -----------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Direct Comparison: Classic vs Subspace ESMDA", fontsize=16)

# Linear case - Final ensemble
ax = axes[0, 0]
ax.hist(
    lm_post_classic[-1, :, 0],
    bins=50,
    alpha=0.5,
    density=True,
    label=f"Classic (mean={lm_post_classic[-1].mean():.3f})",
)
ax.hist(
    lm_post_subspace[-1, :, 0],
    bins=50,
    alpha=0.5,
    density=True,
    label=f"Subspace (mean={lm_post_subspace[-1].mean():.3f})",
)
ax.axvline(xlocation, color="red", linestyle="--", linewidth=2, label="Truth")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Density")
ax.set_title("Linear Case - Final Ensemble")
ax.legend()
ax.grid(True, alpha=0.3)

# Linear case - Scatter plot
ax = axes[0, 1]
# Sample for plotting (too many points otherwise)
rng = np.random.default_rng(seed)
idx = rng.choice(ne, size=min(1000, ne), replace=False)
ax.scatter(
    lm_post_classic[-1, idx, 0], lm_post_subspace[-1, idx, 0], alpha=0.5, s=1
)
ax.plot([-3, 3], [-3, 3], "r--", label="Perfect match")
ax.set_xlabel("Classic ESMDA")
ax.set_ylabel("Subspace ESMDA")
ax.set_title("Linear Case - Parameter Comparison")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect("equal")

# Nonlinear case - Final ensemble
ax = axes[1, 0]
ax.hist(
    nm_post_classic[-1, :, 0],
    bins=50,
    alpha=0.5,
    density=True,
    label=f"Classic (mean={nm_post_classic[-1].mean():.3f})",
)
ax.hist(
    nm_post_subspace[-1, :, 0],
    bins=50,
    alpha=0.5,
    density=True,
    label=f"Subspace (mean={nm_post_subspace[-1].mean():.3f})",
)
ax.axvline(xlocation, color="red", linestyle="--", linewidth=2, label="Truth")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Density")
ax.set_title("Nonlinear Case - Final Ensemble")
ax.legend()
ax.grid(True, alpha=0.3)

# Nonlinear case - Scatter plot
ax = axes[1, 1]
ax.scatter(
    nm_post_classic[-1, idx, 0], nm_post_subspace[-1, idx, 0], alpha=0.5, s=1
)
ax.plot([-3, 3], [-3, 3], "r--", label="Perfect match")
ax.set_xlabel("Classic ESMDA")
ax.set_ylabel("Subspace ESMDA")
ax.set_title("Nonlinear Case - Parameter Comparison")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect("equal")

###############################################################################
# Convergence analysis
# --------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig.suptitle("Convergence Analysis", fontsize=16)

# Linear case convergence
ax = axes[0]
ax.set_title("Linear Case - Mean Evolution")
steps = range(lm_post_classic.shape[0])
means_classic = [lm_post_classic[i].mean() for i in steps]
means_subspace = [lm_post_subspace[i].mean() for i in steps]
ax.plot(steps, means_classic, "o-", label="Classic ESMDA", linewidth=2)
ax.plot(steps, means_subspace, "s-", label="Subspace ESMDA", linewidth=2)
ax.axhline(xlocation, color="red", linestyle="--", label="Truth")
ax.set_xlabel("Iteration")
ax.set_ylabel("Ensemble mean")
ax.legend()
ax.grid(True, alpha=0.3)

# Nonlinear case convergence
ax = axes[1]
ax.set_title("Nonlinear Case - Mean Evolution")
means_classic = [nm_post_classic[i].mean() for i in steps]
means_subspace = [nm_post_subspace[i].mean() for i in steps]
ax.plot(steps, means_classic, "o-", label="Classic ESMDA", linewidth=2)
ax.plot(steps, means_subspace, "s-", label="Subspace ESMDA", linewidth=2)
ax.axhline(xlocation, color="red", linestyle="--", label="Truth")
ax.set_xlabel("Iteration")
ax.set_ylabel("Ensemble mean")
ax.legend()
ax.grid(True, alpha=0.3)

###############################################################################
# Summary statistics
# ------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nTrue parameter value: {xlocation}")
print(f"Prior: mean={mprior.mean():.3f}, std={mprior.std():.3f}")

print("\nLinear case:")
print(f"  Classic:  mean={lm_post_classic[-1].mean():.4f}, "
      f"std={lm_post_classic[-1].std():.4f}")
print(f"  Subspace: mean={lm_post_subspace[-1].mean():.4f}, "
      f"std={lm_post_subspace[-1].std():.4f}")
print(f"  Difference in mean: "
      f"{abs(lm_post_classic[-1].mean() - lm_post_subspace[-1].mean()):.2e}")

print("\nNonlinear case:")
print(f"  Classic:  mean={nm_post_classic[-1].mean():.4f}, "
      f"std={nm_post_classic[-1].std():.4f}")
print(f"  Subspace: mean={nm_post_subspace[-1].mean():.4f}, "
      f"std={nm_post_subspace[-1].std():.4f}")
print(f"  Difference in mean: "
      f"{abs(nm_post_classic[-1].mean() - nm_post_subspace[-1].mean()):.2e}")

###############################################################################
# Performance comparison
# ----------------------


print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

# Test with different ensemble sizes
ensemble_sizes = [50, 100, 200, 500, 1000]
times_classic = []
times_subspace = []

for ne_test in ensemble_sizes:
    x_test = rng.uniform(0.5, 2.5, (ne_test, 1))

    # Time classic ESMDA
    t0 = time.time()
    dageo.esmda(
        model_prior=x_test,
        forward=nonlin_fwd,
        data_obs=n_dobs,
        sigma=obs_std,
        alphas=4,
        random=rng,
    )
    times_classic.append(time.time() - t0)

    # Time subspace ESMDA
    t0 = time.time()
    dageo.esmda_subspace(
        model_prior=x_test,
        forward=nonlin_fwd,
        data_obs=n_dobs,
        sigma=obs_std,
        alphas=4,
        random=rng,
    )
    times_subspace.append(time.time() - t0)

    print(f"\nEnsemble size: {ne_test}")
    print(f"  Classic:  {times_classic[-1]:.3f} s")
    print(f"  Subspace: {times_subspace[-1]:.3f} s")
    print(f"  Ratio:    {times_subspace[-1] / times_classic[-1]:.2f}x")

# Plot performance
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.plot(
    ensemble_sizes, times_classic, "o-", label="Classic ESMDA", linewidth=2
)
ax.plot(
    ensemble_sizes, times_subspace, "s-", label="Subspace ESMDA", linewidth=2
)
ax.set_xlabel("Ensemble size")
ax.set_ylabel("Computation time (s)")
ax.set_title("Performance Comparison: Classic vs Subspace ESMDA")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()

###############################################################################
# Conclusions
# -----------
#
# 1. Both methods produce very similar results (differences typically < 1e-2)
# 2. The subspace formulation using LU decomposition is mathematically
#    equivalent to the classic Cholesky approach
# 3. Small numerical differences are expected due to different
#    decomposition methods
# 4. Performance is comparable for small problems
# 5. The subspace method provides a foundation for efficient localization
#    through R-inflation
#
# The slight differences observed are within numerical precision and both
# methods converge to the same solution.

dageo.Report()
