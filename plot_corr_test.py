import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- plot setup
import scienceplots
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
from itertools import cycle

plt.style.use(['science'])

n_colors = int(17)
mycolors = [to_hex(viridis(i / (n_colors - 1))) for i in range(n_colors)]
color_cycle = cycle(mycolors)  # Create an infinite color iterator
markers = ['o', 's']
marker_cycle = cycle(markers)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=mycolors)
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 2

y_true_folder = "data/datasets/d567b/rtargets"

# -- full model example (for fine-tuned models)
y_pred_folder = "results/d567b/predictions_efn_b0_2hl/"
test_list_path = "test_filenames.txt"

# -- heads example
# y_pred_folder = "results/d567b/heads/convnext_in/"
# test_list_path = "test_filenames_heads.txt"

# --- Load test stems from test_filenames.txt ---
with open(test_list_path, "r") as f:
    test_stems = set(line.strip() for line in f)

# --- Collect all matching .npy filenames ---
common_files = sorted(set(os.listdir(y_true_folder)) & set(os.listdir(y_pred_folder)))
assert common_files, "No common .npy files found!"

# -- filter out noise and keep only test stems
common_files = [
    f for f in common_files
    if int(f.split('_')[0]) <= 73 and os.path.splitext(f)[0] in test_stems
]

# --- Containers for all values ---
all_true = []
all_pred = []

# --- Load vectors ---
for fname in common_files:
    y_true = np.load(os.path.join(y_true_folder, fname))
    y_pred = np.load(os.path.join(y_pred_folder, fname))
    
    all_true.append(y_true.ravel())  # flatten in case of 1D/2D
    all_pred.append(y_pred.ravel())

# Stack into single long vectors
all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

print(f"Loaded {len(all_true):,} values for correlation plot.")

rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
print(f"Test RMSE: {rmse:.6f}")

# --- 2D Histogram Plot ---
plt.figure(figsize=(8, 6))
plt.hist2d(all_true, all_pred, bins=110, cmap='YlGnBu', norm='log', vmax=1e4)

# -- equal limits to ensure 45 deg 
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("(d)")
plt.colorbar(label='log(Count)')
plt.tight_layout()
# plt.show()
plt.savefig('fig4d_efb0_ft.png', dpi=300)
