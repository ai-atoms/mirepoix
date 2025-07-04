import os
import numpy as np
import matplotlib.pyplot as plt

# Settings
feature_indices = [7, 10, 11, 15, 17, 22, 27, 31, 34, 37, 40, 46, 49, 56, 57, 62]
root_dir = 'data/datasets/dsf60_d9'
target_folder = f'{root_dir}/r2targets/'

# Load all .npy files from the folder
all_data = []
for fname in sorted(os.listdir(target_folder)):
    if fname.endswith('.npy'):
        path = os.path.join(target_folder, fname)
        data = np.load(path)
        all_data.append(data)

# Stack all files into a single array (assuming same shape per sample)
all_data = np.stack(all_data, axis=0)  # shape: (num_samples, num_features)

# Select only the requested features
selected_data = all_data[:, feature_indices]  # shape: (num_samples, len(feature_indices))

# Plot each feature in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for i, feat_idx in enumerate(feature_indices):
    ax = axes[i]
    ax.plot(selected_data[:, i])
    ax.set_title(f'Feature {feat_idx}', fontsize=10)
    ax.set_xlim([0, 32])  # Limit x-axis
    ax.tick_params(axis='both', labelsize=8)


# Hide any unused subplots (if len(feature_indices) < 16)
for j in range(len(feature_indices), 16):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
