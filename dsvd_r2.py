import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt

from tqdm import tqdm
from natsort import natsorted

base_path = "data/datasets/d567/"
target_path = os.path.join(base_path, "rtargets_temp")

# Natural sort of .npy files
npy_files = natsorted([
    os.path.join(target_path, fname)
    for fname in os.listdir(target_path)
    if fname.endswith(".npy")
])

all_targets = [np.load(f) for f in npy_files]
y_train = np.vstack(all_targets)

print("Collected targets shape:", y_train.shape)

# Center the data (subtract mean)
y_centered = y_train - np.mean(y_train, axis=0)

# -- part 1 (SVD analysis)
# Compute covariance matrix (n_features × n_features)
cov_matrix = np.cov(y_centered, rowvar=False)

# Perform SVD
U, s, Vh = svd(cov_matrix)  # s: singular values (sorted in descending order)

# Print singular values (diagnose rank)
print("Singular values:", s)

# Effective rank (number of significant singular values)
rank = np.sum(s > 1e-6)  # Adjust threshold based on magnitude
print("Effective rank of y_features:", rank, "/", y_train.shape[1])

plt.plot(s, 'o-')
plt.yscale('log')  # Log scale for better visibility
plt.xlabel("SV Index")
plt.ylabel("SVD magnitude (log)")
plt.xlim(0, 450)
plt.title("SV spectrum of random features")
plt.grid()
plt.show()

# save original indexes
# Assuming you already have SVD results: U, s, Vh
n_components = len(s)  # Number of features

# For the top-k SVD components (e.g., k=5), find original features with highest weights
top_k_components = 4
original_feature_importance = []

for i in range(top_k_components):
    component_weights = Vh[i].copy()  # Weights for i-th SVD component
    top_original_features = np.argsort(np.abs(component_weights))[::-1][:10]  # Top 10 features
    original_feature_importance.append((i, s[i], top_original_features))

# Print results
for comp_idx, singular_val, features in original_feature_importance:
    print(f"SVD Component {comp_idx} (σ={singular_val:.2f}): Original features = {features}")

top_k_per_component = 4  # Adjust based on your tolerance for redundancy
selected_features = set()

for i in range(rank):  # Top components
    component_weights = Vh[i]
    top_features = np.argsort(np.abs(component_weights))[::-1][:top_k_per_component]
    selected_features.update(top_features)


print(f"Selected {len(selected_features)} features out of {y_train.shape[1]}.")
print("Min index:", min(selected_features))
print("Max index:", max(selected_features))

# Export
import json

my_list = selected_features
native_list = [int(x) for x in my_list]

with open(f'{target_path}/selected_features.json', 'w') as f:
    json.dump(native_list, f)  # Order doesn't matter for feature identity

