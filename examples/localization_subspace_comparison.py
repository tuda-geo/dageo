r"""
Localization: Matrix vs Function-based Comparison
=================================================

Compare matrix-based localization (classic ESMDA) with function-based
localization (subspace ESMDA) using a reservoir simulation example.

This example demonstrates:
1. Matrix-based localization using pre-computed covariance
2. Function-based localization with Gaspari-Cohn correlation (R-inflation)
3. Different inflation strategies via alphas
4. Memory and performance comparisons
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np

import dageo
from dageo.data_assimilation import gaspari_cohn

# For reproducibility
rng = np.random.default_rng(2020)

# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Model parameters
# ----------------

# Grid extension
nx = 25
ny = 20
nc = nx * ny

# Permeabilities
perm_mean = 3.0
perm_min = 0.5
perm_max = 5.0

# ESMDA parameters
ne = 50  # Number of ensembles
dt = np.zeros(5) + 0.0001  # Time steps
time_array = np.r_[0, np.cumsum(dt)]

# Assumed standard deviation of our data
dstd = 0.5

# Measurement points
ox = (5, 12, 20)
oy = (5, 10, 15)

# Number of data points
nd = time_array.size * len(ox)

# Wells
wells = np.array(
    [
        [ox[0], oy[0], 180],
        [7, 3, 120],
        [ox[1], oy[1], 180],
        [15, 8, 120],
        [ox[2], oy[2], 180],
        [18, 17, 120],
    ]
)

###############################################################################
# Create permeability maps
# ------------------------

# Get the model and ne prior models
RP = dageo.RandomPermeability(nx, ny, perm_mean, perm_min, perm_max)
perm_true = RP(1, random=rng)
perm_prior = RP(ne, random=rng)

# Add some noise to avoid numerical issues
perm_prior += rng.normal(0, 0.01, perm_prior.shape)

###############################################################################
# Run the prior models
# --------------------

# Instantiate reservoir simulator
RS = dageo.Simulator(nx, ny, wells=wells)


def sim(x):
    """Custom fct to use exp(x), and specific dt & location."""
    # Clip to avoid extreme values
    x_clipped = np.clip(x, perm_min, perm_max)
    return RS(np.exp(x_clipped), dt=dt, data=(ox, oy)).reshape(
        (x.shape[0], -1)
    )


# Simulate data for the prior and true fields
print("Running forward simulations...")
data_prior = sim(perm_prior)
data_true = sim(perm_true)
data_obs = rng.normal(data_true, dstd)
# Fix pressure observations
data_obs[0, : len(ox)] = data_true[0, : len(ox)]

###############################################################################
# Matrix-based Localization
# -------------------------

# Vector of nd length with the well x and y position for each nd data point
nd_positions = np.tile(np.array([ox, oy]), time_array.size).T

# Create localization matrix
print("Creating localization matrix...")
loc_mat = dageo.localization_matrix(RP.cov, nd_positions, (nx, ny))

print(f"Localization matrix shape: {loc_mat.shape}")
print(f"Localization matrix memory: {loc_mat.nbytes / 1024**2:.2f} MB")

###############################################################################
# Function-based Localization
# ---------------------------


# Create localization function using Gaspari-Cohn
def create_localization_fn(nx, ny, obs_positions, radius=10.0):
    """Create a function-based localization using Gaspari-Cohn."""

    def loc_fn(state_idx):
        # Convert 1D state index to 2D position
        state_y, state_x = divmod(state_idx, nx)

        # Compute distances to all observation points
        distances = np.zeros(len(obs_positions))
        for i, (obs_x, obs_y) in enumerate(obs_positions):
            dist = np.sqrt((state_x - obs_x) ** 2 + (state_y - obs_y) ** 2)
            distances[i] = dist

        # Apply Gaspari-Cohn correlation
        return gaspari_cohn(distances, radius)

    return loc_fn


# Create the localization function
loc_fn = create_localization_fn(nx, ny, nd_positions, radius=10.0)

# Test the function
test_correlations = loc_fn(0)  # Test for first state variable
print("\nFunction-based localization test:")
print(f"Correlations shape: {test_correlations.shape}")
print(f"Non-zero correlations: {np.sum(test_correlations > 0)}")

###############################################################################
# ESMDA with different approaches
# --------------------------------


def restrict_permeability(x):
    """Restrict possible permeabilities."""
    np.clip(x, perm_min, perm_max, out=x)


# Common input parameters
base_inp = {
    "forward": sim,
    "data_obs": data_obs,
    "sigma": dstd,
    "alphas": 4,
    "data_prior": data_prior,
    "callback_post": restrict_permeability,
    "random": rng,
}

print("\nRunning ESMDA variants...")

# 1. No localization (classic)
print("1. Classic ESMDA without localization...")
t0 = time.time()
nl_perm_post_classic, nl_data_post_classic = dageo.esmda(
    model_prior=perm_prior.copy(), **base_inp
)
time_no_loc_classic = time.time() - t0
print(f"   Time: {time_no_loc_classic:.3f} s")

# 2. Matrix localization (classic)
print("2. Classic ESMDA with matrix localization...")
t0 = time.time()
ml_perm_post_classic, ml_data_post_classic = dageo.esmda(
    model_prior=perm_prior.copy(), localization_matrix=loc_mat, **base_inp
)
time_matrix_loc = time.time() - t0
print(f"   Time: {time_matrix_loc:.3f} s")

# 3. No localization (subspace)
print("3. Subspace ESMDA without localization...")
t0 = time.time()
nl_perm_post_subspace, nl_data_post_subspace = dageo.esmda_subspace(
    model_prior=perm_prior.copy(), **base_inp
)
time_no_loc_subspace = time.time() - t0
print(f"   Time: {time_no_loc_subspace:.3f} s")

# 4. Function localization (subspace)
print("4. Subspace ESMDA with R-inflation localization...")
t0 = time.time()
fl_perm_post_subspace, fl_data_post_subspace = dageo.esmda_subspace(
    model_prior=perm_prior.copy(), localization_function=loc_fn, **base_inp
)
time_func_loc = time.time() - t0
print(f"   Time: {time_func_loc:.3f} s")

print("\nExecution time summary:")
print(f"Classic (no loc):     {time_no_loc_classic:.3f} s")
print(
    f"Classic (matrix loc): {time_matrix_loc:.3f} s "
    f"(overhead: {time_matrix_loc - time_no_loc_classic:.3f} s)"
)
print(f"Subspace (no loc):    {time_no_loc_subspace:.3f} s")
print(
    f"Subspace (func loc):  {time_func_loc:.3f} s "
    f"(overhead: {time_func_loc - time_no_loc_subspace:.3f} s)"
)

###############################################################################
# Compare results
# ---------------


# Compute RMS errors
def compute_rms(perm_post, perm_true):
    return np.sqrt(np.mean((perm_post - perm_true) ** 2))


rms_prior = compute_rms(perm_prior.mean(axis=0), perm_true[0])
rms_no_loc_classic = compute_rms(
    nl_perm_post_classic.mean(axis=0), perm_true[0]
)
rms_matrix_loc = compute_rms(ml_perm_post_classic.mean(axis=0), perm_true[0])
rms_no_loc_subspace = compute_rms(
    nl_perm_post_subspace.mean(axis=0), perm_true[0]
)
rms_func_loc = compute_rms(fl_perm_post_subspace.mean(axis=0), perm_true[0])

print("\nRMS Errors:")
print(f"Prior:                {rms_prior:.4f}")
print(f"Classic (no loc):     {rms_no_loc_classic:.4f}")
print(f"Classic (matrix loc): {rms_matrix_loc:.4f}")
print(f"Subspace (no loc):    {rms_no_loc_subspace:.4f}")
print(f"Subspace (func loc):  {rms_func_loc:.4f}")

###############################################################################
# Visualization
# -------------

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)


# Helper function for plotting
def plot_field(ax, field, title, wells=wells):
    im = ax.imshow(
        field.T, origin="lower", cmap="viridis", vmin=perm_min, vmax=perm_max
    )
    ax.set_title(title)
    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    # Plot wells
    for well in wells:
        ax.plot(
            well[0], well[1], ["C3v", "C1^"][int(well[2] == 120)], markersize=8
        )
    return im


# Plot results
im1 = plot_field(axes[0, 0], perm_true[0], "Truth")
im2 = plot_field(
    axes[0, 1], perm_prior.mean(axis=0), f"Prior Mean (RMS={rms_prior:.3f})"
)
im3 = plot_field(
    axes[0, 2],
    nl_perm_post_classic.mean(axis=0),
    f"Classic No Loc (RMS={rms_no_loc_classic:.3f})",
)
im4 = plot_field(
    axes[1, 0],
    ml_perm_post_classic.mean(axis=0),
    f"Classic Matrix Loc (RMS={rms_matrix_loc:.3f})",
)
im5 = plot_field(
    axes[1, 1],
    nl_perm_post_subspace.mean(axis=0),
    f"Subspace No Loc (RMS={rms_no_loc_subspace:.3f})",
)
im6 = plot_field(
    axes[1, 2],
    fl_perm_post_subspace.mean(axis=0),
    f"Subspace R-inflation (RMS={rms_func_loc:.3f})",
)

# Add colorbar
fig.colorbar(im1, ax=axes, label="log-permeability")

###############################################################################
# Localization visualization
# --------------------------

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Matrix localization visualization
loc_weight = loc_mat.sum(axis=2) / time_array.size
im1 = ax1.imshow(loc_weight.T, origin="lower", cmap="plasma")
ax1.contour(
    loc_weight.T,
    levels=[
        0.2,
    ],
    colors="w",
)
ax1.set_title("Matrix-based Localization Weights")
ax1.set_xlabel("x-direction")
ax1.set_ylabel("y-direction")
for well in wells:
    ax1.plot(well[0], well[1], ["C3v", "C1^"][int(well[2] == 120)])
fig2.colorbar(im1, ax=ax1, label="Average Localization Weight")

# Function-based localization for central gridpoint
central_idx = ny // 2 * nx + nx // 2
correlations = loc_fn(central_idx).reshape(-1, len(ox))
avg_corr = correlations.mean(axis=0)

# Create a 2D visualization
func_loc_vis = np.zeros((ny, nx))
for i in range(nc):
    y, x = divmod(i, nx)
    corr = loc_fn(i)
    func_loc_vis[y, x] = corr.mean()  # Average correlation

im2 = ax2.imshow(func_loc_vis, origin="lower", cmap="plasma")
ax2.contour(
    func_loc_vis,
    levels=[
        0.2,
    ],
    colors="w",
)
ax2.set_title("R-inflation Localization (avg correlation)")
ax2.set_xlabel("x-direction")
ax2.set_ylabel("y-direction")
ax2.plot(nx // 2, ny // 2, "wx", markersize=10, label="Reference point")
for well in wells:
    ax2.plot(well[0], well[1], ["C3v", "C1^"][int(well[2] == 120)])
ax2.legend()
fig2.colorbar(im2, ax=ax2, label="Average Correlation")

###############################################################################
# Different inflation strategies
# ------------------------------

print("\n" + "=" * 60)
print("Testing different inflation strategies")
print("=" * 60)

# Different alpha configurations
alphas_configs = {
    "uniform": [4, 4, 4, 4],
    "decreasing": [10, 5, 3, 2],
    "aggressive": [2, 2, 2, 2],
    "conservative": [8, 8, 8, 8],
}

results_inflation = {}

for name, alphas in alphas_configs.items():
    # Check sum
    alpha_sum = np.sum(1 / np.array(alphas))
    print(f"\n{name}: alphas={alphas}, sum(1/α)={alpha_sum:.3f}")

    # Run with function localization
    t0 = time.time()
    perm_post, _ = dageo.esmda_subspace(
        model_prior=perm_prior.copy(),
        forward=sim,
        data_obs=data_obs,
        sigma=dstd,
        alphas=alphas,
        localization_function=loc_fn,
        callback_post=restrict_permeability,
        random=rng,
    )
    exec_time = time.time() - t0

    rms = compute_rms(perm_post.mean(axis=0), perm_true[0])
    results_inflation[name] = {
        "rms": rms,
        "time": exec_time,
        "ensemble": perm_post,
    }
    print(f"  RMS error: {rms:.4f}, Time: {exec_time:.3f} s")

###############################################################################
# Plot inflation comparison
# -------------------------

fig3, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

for i, (name, result) in enumerate(results_inflation.items()):
    plot_field(
        axes[i],
        result["ensemble"].mean(axis=0),
        f'{name.capitalize()} (RMS={result["rms"]:.3f})',
    )

fig3.suptitle("Effect of Different Inflation Strategies", fontsize=16)

###############################################################################
# Summary statistics
# ------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\nLocalization comparison:")
print("- Matrix-based: Pre-computes full covariance localization matrix")
print(
    "- R-inflation: Computes correlations on-demand, "
    "inflates observation errors"
)
print("\nMemory usage:")
print(
    f"Matrix approach: {loc_mat.nbytes / 1024**2:.2f} MB "
    f"for localization matrix"
)
print("R-inflation approach: ~0 MB (computed on-demand)")

print("\nPerformance:")
print(
    f"Matrix localization overhead: "
    f"{time_matrix_loc - time_no_loc_classic:.3f} s"
)
print(f"R-inflation overhead: {time_func_loc - time_no_loc_subspace:.3f} s")

print("\nAccuracy comparison:")
print(f"Classic with matrix localization:    RMS = {rms_matrix_loc:.4f}")
print(f"Subspace with R-inflation:          RMS = {rms_func_loc:.4f}")
print(
    f"Difference:                         "
    f"{abs(rms_matrix_loc - rms_func_loc):.4f}"
)

print("\nBest inflation strategy (by RMS):")
best_inflation = min(results_inflation.items(), key=lambda x: x[1]["rms"])
print(f"{best_inflation[0]}: RMS = {best_inflation[1]['rms']:.4f}")

# Test memory efficiency for larger problems
print("\n" + "=" * 60)
print("MEMORY EFFICIENCY DEMONSTRATION")
print("=" * 60)

# Simulate a larger problem
nx_large = 100
ny_large = 100
nc_large = nx_large * ny_large
nd_large = 50

print("\nLarge problem dimensions:")
print(f"Grid: {nx_large} x {ny_large} = {nc_large:,} cells")
print(f"Observations: {nd_large}")

# Matrix approach memory estimate
matrix_mem = nc_large * nd_large * 8 / 1024**2  # 8 bytes per float64
print(f"\nMatrix localization would require: {matrix_mem:.1f} MB")

# Function approach memory
print("R-inflation localization requires: ~0 MB (on-demand computation)")
print(f"Memory savings: {matrix_mem:.1f} MB ({matrix_mem / 1024:.2f} GB)")

print(
    "\nFor very large problems (e.g., 1000x1000 grid with 1000 observations):"
)
huge_mem = 1000 * 1000 * 1000 * 8 / 1024**3
print(f"Matrix approach would need: {huge_mem:.1f} GB")
print("R-inflation approach needs: ~0 GB")

###############################################################################
# Conclusions
# -----------
print("\n" + "=" * 60)
print("CONCLUSIONS")
print("=" * 60)
print(
    "1. R-inflation localization produces comparable results to "
    "matrix localization"
)
print("2. Memory usage is dramatically reduced for large problems")
print(
    "3. Computational overhead is reasonable for the memory savings "
    "achieved"
)
print("4. Different inflation strategies can be used to tune the algorithm")
print(
    "5. The subspace formulation enables efficient localization for "
    "large-scale problems"
)

plt.show()

###############################################################################

dageo.Report()
