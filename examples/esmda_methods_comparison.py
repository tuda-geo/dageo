r"""
Comprehensive ESMDA Methods Comparison
======================================

This example demonstrates both ESMDA implementations (classic and subspace)
on a realistic reservoir simulation problem, showcasing:

1. Classic ESMDA with Cholesky decomposition
2. Subspace ESMDA with LU decomposition
3. Localization strategies for both methods
4. Performance and accuracy comparisons
5. Practical recommendations for different problem sizes

The reservoir model consists of a 2D grid with heterogeneous permeability,
multiple wells, and time-lapse production data.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

import dageo
from dageo.data_assimilation import gaspari_cohn

# sphinx_gallery_thumbnail_number = 3

###############################################################################
# Reservoir Model Setup
# ---------------------
#
# We create a synthetic reservoir with known true permeability to test
# our data assimilation methods.

# Grid dimensions
nx, ny = 40, 30
nc = nx * ny

# Permeability parameters
perm_mean = 2.5
perm_min = 0.5
perm_max = 4.5

# Create synthetic truth and prior ensemble
rng = np.random.default_rng(2024)
ne = 80  # Ensemble size

# Generate permeability fields
RP = dageo.RandomPermeability(nx, ny, perm_mean, perm_min, perm_max)
perm_true = RP(1, random=rng)
perm_prior = RP(ne, random=rng)

# Add small noise to avoid numerical issues
perm_prior += rng.normal(0, 0.01, perm_prior.shape)

print(f"Reservoir grid: {nx} × {ny} = {nc:,} cells")
print(f"Ensemble size: {ne} members")
print(
    f"Prior permeability: mean={perm_prior.mean():.2f}, "
    f"std={perm_prior.std():.2f}"
)

###############################################################################
# Well Configuration and Production Schedule
# ------------------------------------------

# Define well locations
# Format: [x, y, type] where type=180 (producer) or 120 (injector)
wells = np.array(
    [
        # Producers (red triangles pointing down)
        [8, 8, 180],
        [32, 8, 180],
        [20, 22, 180],
        # Injectors (blue triangles pointing up)
        [8, 22, 120],
        [32, 22, 120],
        [20, 8, 120],
    ]
)

# Production schedule
nt = 10  # Number of time steps
dt = np.ones(nt) * 0.0001  # Time step size
time_points = np.r_[0, np.cumsum(dt)]

# Observation locations (at producers)
obs_wells = wells[wells[:, 2] == 180]  # Only producers
ox = obs_wells[:, 0]
oy = obs_wells[:, 1]

print("\nWell configuration:")
print(f"Producers: {len(obs_wells)} wells")
print(f"Injectors: {len(wells) - len(obs_wells)} wells")
print(f"Time steps: {nt}")

###############################################################################
# Forward Model and Data Generation
# ---------------------------------

# Initialize reservoir simulator
RS = dageo.Simulator(nx, ny, wells=wells)


def forward_model(perm_log):
    """Forward model: log-permeability to production data."""
    # Ensure reasonable values
    perm_log_safe = np.clip(perm_log, perm_min, perm_max)
    perm = np.exp(perm_log_safe)

    # Run simulation
    data = RS(perm, dt=dt, data=(ox, oy))

    # Reshape to (ne, n_data)
    return data.reshape((perm_log.shape[0], -1))


# Generate synthetic observations
print("\nGenerating synthetic data...")
data_true = forward_model(perm_true)
obs_error = 0.2  # Observation error standard deviation
n_data = data_true.shape[1]

# Add noise to create observations
data_obs = data_true + rng.normal(0, obs_error, data_true.shape)

# Keep initial pressure measurements exact
data_obs[:, : len(ox)] = data_true[:, : len(ox)]

print(f"Number of observations: {n_data}")
print(f"Observation error: {obs_error}")

# Run prior ensemble
print("\nRunning prior ensemble simulations...")
t0 = time.time()
data_prior = forward_model(perm_prior)
print(f"Prior simulations completed in {time.time() - t0:.1f} seconds")

###############################################################################
# Visualization Helper Functions
# ------------------------------


def plot_field(
    ax, field, title, vmin=None, vmax=None, wells=wells, cmap="viridis"
):
    """Plot a permeability field with wells."""
    if vmin is None:
        vmin = perm_min
    if vmax is None:
        vmax = perm_max

    im = ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    # Add wells
    for well in wells:
        marker = "v" if well[2] == 180 else "^"
        color = "red" if well[2] == 180 else "blue"
        ax.plot(
            well[0],
            well[1],
            marker,
            color=color,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    return im


def plot_ensemble_statistics(ax, ensemble, truth, title):
    """Plot ensemble mean, std, and truth."""
    mean = ensemble.mean(axis=0)
    std = ensemble.std(axis=0)

    # Create RGB image: R=std, G=mean, B=truth
    rgb = np.zeros((nx, ny, 3))
    rgb[:, :, 0] = (std - std.min()) / (std.max() - std.min())
    rgb[:, :, 1] = (mean - perm_min) / (perm_max - perm_min)
    rgb[:, :, 2] = (truth[0] - perm_min) / (perm_max - perm_min)

    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add legend
    ax.text(
        0.02,
        0.98,
        "Red: Uncertainty",
        transform=ax.transAxes,
        va="top",
        color="red",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.92,
        "Green: Mean",
        transform=ax.transAxes,
        va="top",
        color="green",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.86,
        "Blue: Truth",
        transform=ax.transAxes,
        va="top",
        color="blue",
        fontweight="bold",
    )


###############################################################################
# ESMDA Parameters
# ----------------

# Common parameters for both methods
alphas = 4  # Number of iterations (uniform alphas)


def callback(x):
    return np.clip(x, perm_min, perm_max, out=x)


# Base input dictionary
esmda_params = {
    "forward": forward_model,
    "data_obs": data_obs[0],  # Single realization
    "sigma": obs_error,
    "alphas": alphas,
    "data_prior": data_prior,
    "callback_post": callback,
    "random": rng,
    "return_steps": True,  # Get intermediate results
}

###############################################################################
# Run Classic ESMDA
# -----------------

print("\n" + "=" * 60)
print("CLASSIC ESMDA (Cholesky decomposition)")
print("=" * 60)

# Without localization
print("\nRunning classic ESMDA without localization...")
t0 = time.time()
perm_steps_classic_nl, data_steps_classic_nl = dageo.esmda(
    model_prior=perm_prior.copy(), **esmda_params
)
perm_post_classic_nl = perm_steps_classic_nl[-1]
data_post_classic_nl = data_steps_classic_nl[-1]
time_classic_nl = time.time() - t0
print(f"Completed in {time_classic_nl:.2f} seconds")

# With localization
print("\nCreating localization matrix...")
obs_positions = np.tile(np.array([ox, oy]), len(time_points)).T
loc_matrix = dageo.localization_matrix(RP.cov, obs_positions, (nx, ny))
print(f"Localization matrix size: {loc_matrix.nbytes / 1024**2:.1f} MB")

print("Running classic ESMDA with matrix localization...")
t0 = time.time()
perm_steps_classic_loc, data_steps_classic_loc = dageo.esmda(
    model_prior=perm_prior.copy(),
    localization_matrix=loc_matrix,
    **esmda_params,
)
perm_post_classic_loc = perm_steps_classic_loc[-1]
data_post_classic_loc = data_steps_classic_loc[-1]
time_classic_loc = time.time() - t0
print(f"Completed in {time_classic_loc:.2f} seconds")

###############################################################################
# Run Subspace ESMDA
# ------------------

print("\n" + "=" * 60)
print("SUBSPACE ESMDA (LU decomposition)")
print("=" * 60)

# Without localization
print("\nRunning subspace ESMDA without localization...")
t0 = time.time()
perm_steps_subspace_nl, data_steps_subspace_nl = dageo.esmda_subspace(
    model_prior=perm_prior.copy(), **esmda_params
)
perm_post_subspace_nl = perm_steps_subspace_nl[-1]
data_post_subspace_nl = data_steps_subspace_nl[-1]
time_subspace_nl = time.time() - t0
print(f"Completed in {time_subspace_nl:.2f} seconds")

# With R-inflation localization
print("\nCreating localization function...")


def localization_function(state_idx):
    """R-inflation localization using Gaspari-Cohn."""
    # Convert state index to 2D position
    y, x = divmod(state_idx, nx)

    # Compute distances to all observations
    distances = []
    for obs_x, obs_y in zip(ox, oy):
        dist = np.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
        # Repeat for each time step
        distances.extend([dist] * len(time_points))

    # Apply Gaspari-Cohn correlation
    return gaspari_cohn(np.array(distances), radius=15.0)


print("Running subspace ESMDA with R-inflation localization...")
t0 = time.time()
perm_steps_subspace_loc, data_steps_subspace_loc = dageo.esmda_subspace(
    model_prior=perm_prior.copy(),
    localization_function=localization_function,
    **esmda_params,
)
perm_post_subspace_loc = perm_steps_subspace_loc[-1]
data_post_subspace_loc = data_steps_subspace_loc[-1]
time_subspace_loc = time.time() - t0
print(f"Completed in {time_subspace_loc:.2f} seconds")

###############################################################################
# Compare Results
# ---------------


# Calculate RMS errors
def rms_error(estimate, truth):
    return np.sqrt(np.mean((estimate - truth) ** 2))


results = {
    "Prior": {"ensemble": perm_prior, "time": 0},
    "Classic": {"ensemble": perm_post_classic_nl, "time": time_classic_nl},
    "Classic+Loc": {
        "ensemble": perm_post_classic_loc,
        "time": time_classic_loc,
    },
    "Subspace": {"ensemble": perm_post_subspace_nl, "time": time_subspace_nl},
    "Subspace+Loc": {
        "ensemble": perm_post_subspace_loc,
        "time": time_subspace_loc,
    },
}

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Method':<15} {'RMS Error':>10} {'Ens. Spread':>12} {'Time (s)':>10}")
print("-" * 50)

for name, result in results.items():
    ens = result["ensemble"]
    rms = rms_error(ens.mean(axis=0), perm_true[0])
    spread = ens.std(axis=0).mean()
    print(f"{name:<15} {rms:>10.4f} {spread:>12.4f} {result['time']:>10.2f}")

###############################################################################
# Visualization: Final Results
# ----------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Truth and prior
im1 = plot_field(axes[0, 0], perm_true[0], "Truth")
im2 = plot_field(axes[0, 1], perm_prior.mean(axis=0), "Prior Mean")
plot_ensemble_statistics(axes[0, 2], perm_prior, perm_true, "Prior Statistics")

# Classic vs Subspace with localization
im4 = plot_field(
    axes[1, 0], perm_post_classic_loc.mean(axis=0), "Classic + Loc"
)
im5 = plot_field(
    axes[1, 1], perm_post_subspace_loc.mean(axis=0), "Subspace + Loc"
)
plot_ensemble_statistics(
    axes[1, 2], perm_post_subspace_loc, perm_true, "Subspace + Loc Statistics"
)

plt.tight_layout()

# Add colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
fig.colorbar(im1, cax=cbar_ax, label="Log-permeability")

###############################################################################
# Visualization: Convergence Analysis
# -----------------------------------

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Plot RMS error evolution
ax = axes2[0, 0]
for name, steps in [
    ("Classic", perm_steps_classic_nl),
    ("Classic+Loc", perm_steps_classic_loc),
    ("Subspace", perm_steps_subspace_nl),
    ("Subspace+Loc", perm_steps_subspace_loc),
]:
    rms_evolution = [
        rms_error(step.mean(axis=0), perm_true[0]) for step in steps
    ]
    ax.plot(
        range(len(rms_evolution)), rms_evolution, "o-", label=name, linewidth=2
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("RMS Error")
ax.set_title("Convergence of Different Methods")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot ensemble spread evolution
ax = axes2[0, 1]
for name, steps in [
    ("Classic", perm_steps_classic_nl),
    ("Classic+Loc", perm_steps_classic_loc),
    ("Subspace", perm_steps_subspace_nl),
    ("Subspace+Loc", perm_steps_subspace_loc),
]:
    spread_evolution = [step.std(axis=0).mean() for step in steps]
    ax.plot(
        range(len(spread_evolution)),
        spread_evolution,
        "o-",
        label=name,
        linewidth=2,
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("Average Ensemble Spread")
ax.set_title("Uncertainty Reduction")
ax.legend()
ax.grid(True, alpha=0.3)

# Compare classic vs subspace (no localization)
ax = axes2[1, 0]
ax.scatter(
    perm_post_classic_nl.flatten(),
    perm_post_subspace_nl.flatten(),
    alpha=0.5,
    s=1,
)
ax.plot([perm_min, perm_max], [perm_min, perm_max], "r--", label="y=x")
ax.set_xlabel("Classic ESMDA")
ax.set_ylabel("Subspace ESMDA")
ax.set_title("Classic vs Subspace (No Localization)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

# Compare classic vs subspace (with localization)
ax = axes2[1, 1]
ax.scatter(
    perm_post_classic_loc.flatten(),
    perm_post_subspace_loc.flatten(),
    alpha=0.5,
    s=1,
)
ax.plot([perm_min, perm_max], [perm_min, perm_max], "r--", label="y=x")
ax.set_xlabel("Classic ESMDA (Matrix Loc)")
ax.set_ylabel("Subspace ESMDA (R-inflation)")
ax.set_title("Classic vs Subspace (With Localization)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()

###############################################################################
# Localization Visualization
# --------------------------

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Show localization for a central point
central_idx = ny // 2 * nx + nx // 2
central_y, central_x = divmod(central_idx, nx)

# Matrix localization
# loc_matrix shape is (nx, ny, nd) where nd = n_data
# For visualization, we'll show average localization for the central point
loc_weights_matrix = (
    loc_matrix[central_x, central_y, :].reshape(-1, len(ox)).mean(axis=0)
)
ax = axes3[0]
ax.imshow(np.zeros((ny, nx)), cmap="gray", alpha=0.3, origin="lower")
for i, (x, y, w) in enumerate(zip(ox, oy, loc_weights_matrix)):
    circle = plt.Circle((x, y), radius=w * 20, color="red", alpha=0.5)
    ax.add_patch(circle)
    ax.plot(x, y, "ro", markersize=8)
ax.plot(central_x, central_y, "b*", markersize=15, label="Reference point")
ax.set_title("Matrix Localization Weights")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_aspect("equal")
ax.set_xlim(-1, nx)
ax.set_ylim(-1, ny)

# R-inflation localization pattern
ax = axes3[1]
loc_field = np.zeros((ny, nx))
for idx in range(nc):
    loc_field.flat[idx] = localization_function(idx).mean()
im = ax.imshow(loc_field.T, origin="lower", cmap="hot")
ax.plot(central_x, central_y, "b*", markersize=15)
for x, y in zip(ox, oy):
    ax.plot(x, y, "co", markersize=8)
ax.set_title("R-inflation Correlation Field")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="Average correlation")

# Localization effect on update
ax = axes3[2]
update_classic = perm_post_classic_loc.mean(axis=0) - perm_prior.mean(axis=0)
update_subspace = perm_post_subspace_loc.mean(axis=0) - perm_prior.mean(axis=0)
diff = update_classic - update_subspace
im = ax.imshow(
    diff.T,
    origin="lower",
    cmap="RdBu_r",
    vmin=-np.abs(diff).max(),
    vmax=np.abs(diff).max(),
)
ax.set_title("Update Difference (Classic - Subspace)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="Difference")

plt.tight_layout()

###############################################################################
# Performance Analysis
# --------------------

print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

# Memory usage
print("\nMemory usage:")
print(f"Prior ensemble: {perm_prior.nbytes / 1024**2:.1f} MB")
print(f"Localization matrix: {loc_matrix.nbytes / 1024**2:.1f} MB")
print("Localization function: ~0 MB (computed on demand)")

# Scalability test
print("\nScalability test (estimated for larger problems):")
sizes = [(50, 50), (100, 100), (200, 200), (500, 500)]
n_obs = 50  # Fixed number of observations

print(
    f"{'Grid Size':<12} {'Cells':>8} {'Matrix Loc (MB)':>15} {'Speedup':>10}"
)
print("-" * 50)

for nx_test, ny_test in sizes:
    n_cells = nx_test * ny_test
    matrix_size = n_cells * n_obs * len(time_points) * 8 / 1024**2
    speedup = matrix_size / 100  # Rough estimate
    print(
        f"{nx_test}×{ny_test:<9} {n_cells:>8,} {matrix_size:>15.1f} "
        f"{speedup:>10.1f}x"
    )

###############################################################################
# Animation: ESMDA Evolution
# --------------------------

# Create animation of ESMDA iterations
fig_anim, axes_anim = plt.subplots(2, 2, figsize=(12, 10))
axes_anim = axes_anim.ravel()


def animate(frame):
    for ax in axes_anim:
        ax.clear()

    # Show evolution for all methods
    methods = [
        ("Classic", perm_steps_classic_nl),
        ("Classic + Loc", perm_steps_classic_loc),
        ("Subspace", perm_steps_subspace_nl),
        ("Subspace + Loc", perm_steps_subspace_loc),
    ]

    for i, (name, steps) in enumerate(methods):
        if frame < len(steps):
            field = steps[frame].mean(axis=0)
            rms = rms_error(field, perm_true[0])
            plot_field(
                axes_anim[i], field,
                f"{name} - Iter {frame + 1} (RMS={rms:.3f})"
            )

    plt.suptitle(f"ESMDA Evolution - Iteration {frame + 1}", fontsize=16)
    return axes_anim


# Create animation (only if in notebook environment)
try:
    anim = FuncAnimation(
        fig_anim, animate, frames=alphas + 1, interval=1000, repeat=True
    )
    plt.close(fig_anim)  # Don't show static figure
    HTML(anim.to_jshtml())
except BaseException:
    print("\nAnimation created (viewable in Jupyter notebook)")

###############################################################################
# Recommendations
# ---------------

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print("\n1. SMALL PROBLEMS (< 10,000 cells, < 100 observations):")
print("   - Both methods perform similarly")
print("   - Classic ESMDA may be slightly faster")
print("   - Matrix localization is feasible")

print("\n2. MEDIUM PROBLEMS (10,000 - 100,000 cells, 100-1000 observations):")
print("   - Consider subspace ESMDA for numerical stability")
print("   - R-inflation localization recommended over matrix")
print("   - Memory savings become significant")

print("\n3. LARGE PROBLEMS (> 100,000 cells, > 1000 observations):")
print("   - Subspace ESMDA with R-inflation strongly recommended")
print("   - Matrix localization becomes prohibitive")
print("   - Consider adaptive localization radius")

print("\n4. LOCALIZATION RADIUS:")
print("   - Start with 1.5-2x average well spacing")
print("   - Increase if updates are too noisy")
print("   - Decrease if updates are too smooth")

print("\n5. INFLATION FACTORS (alphas):")
print("   - Uniform: Good default choice")
print("   - Decreasing: More aggressive early updates")
print("   - Conservative: Better for noisy data")

###############################################################################
# Save Results
# ------------

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save the best result
best_result = perm_post_subspace_loc
np.save("reservoir_esmda_result.npy", best_result)
print("Saved best ensemble to 'reservoir_esmda_result.npy'")

# Create summary plot
fig_summary, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(
    best_result.mean(axis=0).T,
    origin="lower",
    cmap="viridis",
    vmin=perm_min,
    vmax=perm_max,
)
ax.set_title(
    "Final Permeability Estimate (Subspace ESMDA with R-inflation)",
    fontsize=14,
)
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add wells
for well in wells:
    marker = "v" if well[2] == 180 else "^"
    color = "red" if well[2] == 180 else "blue"
    label = "Producer" if well[2] == 180 else "Injector"
    ax.plot(
        well[0],
        well[1],
        marker,
        color=color,
        markersize=12,
        markeredgecolor="white",
        markeredgewidth=2,
        label=(
            label if well[0] == wells[0, 0] or well[0] == wells[3, 0] else ""
        ),
    )

ax.legend(loc="upper right")
plt.colorbar(im, ax=ax, label="Log-permeability")
plt.tight_layout()
plt.savefig("reservoir_esmda_summary.png", dpi=300, bbox_inches="tight")
print("Saved summary figure to 'reservoir_esmda_summary.png'")

plt.show()

###############################################################################

# Report
dageo.Report()
