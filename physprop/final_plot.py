import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -- plot setup
import scienceplots
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
from itertools import cycle
import joblib

plt.style.use(['science'])

n_colors = int(5)
mycolors = [to_hex(viridis(i / (n_colors - 1))) for i in range(n_colors)]
color_cycle = cycle(mycolors)  # Create an infinite color iterator
markers = ['o', 's']
marker_cycle = cycle(markers)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=mycolors)
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 2

# --- Parameters ---
target_folder = "./data/datasets/d567b/ptargets"
regression_csv = "regression_results.csv"
feature_rmse = {
    0: 0.0121,
    1: 0.0119,
    2: 0.0055,
    3: 0.0134,
    4: 0.0218,
    5: 0.0303,
    6: 0.0293,
    7: 0.0361,
}

# --- Load target data (ground truth) ---
desired_fname = '70'
filenames = sorted([
    f for f in os.listdir(target_folder)
    if desired_fname in f and f.endswith('.npy')
])
Y = np.stack([np.load(os.path.join(target_folder, f)) for f in filenames])  # (N, 8)

n_frames, n_targets = Y.shape
Y_scaled = np.zeros_like(Y)

# -- camilofs (load scalers from phead_search_r3)
scalers = []
for i in range(n_targets):
    scaler = joblib.load(f'saved_models/d567b/scalers/scaler_target_{i}.pkl')
    Y_scaled[:, i:i+1] = scaler.transform(Y[:, i:i+1])
    scalers.append(scaler)

# --- Load regression results ---
df = pd.read_csv(regression_csv)

# --- Preprocess regression data ---
# We'll store a matrix (n_frames x n_targets) for predictions and uncertainties
pred_matrix = np.zeros((n_frames, n_targets))
unc_matrix = np.zeros((n_frames, n_targets))

for _, row in df.iterrows():
    frame = int(row["frame"])
    feature = int(row["feature"])
    pred = row["regression"]
    unc = row["regression_unc"]

    pred_matrix[frame, feature] = pred
    unc_matrix[frame, feature] = unc + feature_rmse[feature]  # add RMSE as instructed

# --- Invert MinMax scaling ---
preds_unscaled = np.zeros_like(pred_matrix)
uncs_unscaled = np.zeros_like(unc_matrix)

for i in range(n_targets):
    # Reshape to use inverse_transform
    pred_scaled = pred_matrix[:, i:i+1]
    unc_scaled = unc_matrix[:, i:i+1]
    
    # Invert scaling
    preds_unscaled[:, i:i+1] = scalers[i].inverse_transform(pred_scaled)
    
    # Estimate uncertainty bounds
    preds_plus_unc = pred_scaled + unc_scaled
    upper_bound_unscaled = scalers[i].inverse_transform(preds_plus_unc)
    
    # Calculate the uncertainty in the original scale
    uncs_unscaled[:, i] = (upper_bound_unscaled - preds_unscaled[:, i:i+1]).flatten()
    
   
# Scale for better plot
Y[:, 3] *= 5e3
Y[:, 4] *= 1e5
Y[:, 6] *= 1e5

preds_unscaled[:, 3] *= 5e3
preds_unscaled[:, 4] *= 1e5
preds_unscaled[:, 6] *= 1e5

uncs_unscaled[:, 3] *= 5e3
uncs_unscaled[:, 4] *= 1e5
uncs_unscaled[:, 6] *= 1e5

# Remaping x
dpa = (np.arange(n_frames) * 2 * 200) / 686000

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Calculate box data for the left plot
box_data = np.zeros((n_frames, 5))  # [min, Q1, median, Q3, max] per frame

for t in range(n_frames):
    median = Y[t, 0]
    max_val = Y[t, 2]
    sigma = np.sqrt(Y[t, 1])  # std = √variance
    
    box_data[t, 0] = median - (max_val - median)  # min (symmetry assumption)
    box_data[t, 1] = median - sigma               # Q1 (approximation)
    box_data[t, 2] = median                       # median
    box_data[t, 3] = median + sigma               # Q3 (approximation)
    box_data[t, 4] = max_val                      # max

# Left: Ground-truth
for i in range(n_targets):
    if i in {5, 7}:  # Skip unwanted features
        continue
    
    if i == 0:  # Boxplot for median/var/max
        ax1.boxplot(
            box_data.T,  # Transpose for correct shape (5 stats × n_frames)
            positions=dpa,            # x-axis positions
            widths=0.0002,            # box width
            patch_artist=True,
            boxprops=dict(facecolor=mycolors[4], alpha=0.5),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linestyle='-', linewidth=1, color='silver'),
            showfliers=False
        )
    elif i not in {1, 2}: # Other features
        mycolor = mycolors[i-4]
        if i == 3:
            mycolor = mycolors[i-2]
        ax1.plot(dpa, Y[:, i], color=mycolor, label=f"Feature {i}")

ax1.set_title("(b)")
ax1.set_xlabel("Dose (dpa)")
ax1.set_ylabel("C15 size (atoms)")
ax1.set_xticks([0.005, 0.01, 0.02, 0.03, 0.035])
ax1.set_xticklabels([0.005, 0.01, 0.02, 0.03, 0.035], rotation=0)
# ax1.legend(loc="upper right")
# ax1.grid(True)
ax1.set_xlim(0.002, 0.035)
ax1.set_ylim(0, 40)

# Calculate box data for the right plot
box_data = np.zeros((n_frames, 5))  # [min, Q1, median, Q3, max] per frame

for t in range(n_frames):
    median = preds_unscaled[t, 0]
    max_val = preds_unscaled[t, 2]
    sigma = np.sqrt(preds_unscaled[t, 1])  # std = √variance
    
    box_data[t, 0] = median - (max_val - median)  # min (symmetry assumption)
    box_data[t, 1] = median - sigma               # Q1 (approximation)
    box_data[t, 2] = median                       # median
    box_data[t, 3] = median + sigma               # Q3 (approximation)
    box_data[t, 4] = max_val                      # max


# Right: Predictions with uncertainties
# Left: Ground-truth
for i in range(n_targets):
    if i in {5, 7}:  # Skip unwanted features
        continue
    
    if i == 0:  # Boxplot for median/var/max
        ax2.boxplot(
            box_data.T,  # Transpose for correct shape (5 stats × n_frames)
            positions=dpa,            # x-axis positions
            widths=0.0002,            # box width
            patch_artist=True,
            boxprops=dict(facecolor=mycolors[4], alpha=0.5),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linestyle='-', linewidth=1, color='silver'),
            showfliers=False
        )

    elif i not in {1, 2}:
        marker = next(marker_cycle)
        mycolor = mycolors[i-4]
        if i == 3:
            mycolor = mycolors[i-2]
        ax2.errorbar(
            dpa,
            preds_unscaled[:, i],
            yerr=uncs_unscaled[:, i],
            label=f"Feature {i}",
            fmt=marker,               
            markerfacecolor='none',
            markeredgecolor=mycolor,
            ecolor=mycolor,
            linestyle='none', 
            capsize=2
        )

ax2.set_title("(c)")
ax2.set_xlabel("Dose (dpa)")
ax2.set_ylabel("C15 size (atoms)")
ax2.set_xticks([0.005, 0.01, 0.02, 0.03, 0.035])
ax2.set_xticklabels([0.005, 0.01, 0.02, 0.03, 0.035], rotation=0)
ax2.axvline(x=0.00466, color='red', linestyle=':', alpha=0.5)
# ax1.legend(loc="upper right")
# ax1.grid(True)
ax2.set_xlim(0.002, 0.035)
ax2.set_ylim(0, 40)

plt.tight_layout()
# plt.show()
plt.savefig('fig7_varp_physprop_beginning', dpi=300)
