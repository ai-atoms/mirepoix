import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
import scienceplots

plt.style.use(['science'])
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 2

# Settings
root_dir = 'data/datasets/d567b'
target_folder = f'{root_dir}/r3targets/'

# Load all .npy files from the folder
all_data = []
for fname in sorted(os.listdir(target_folder)):
    if '70_' in fname:
        if fname.endswith('.npy'):
            path = os.path.join(target_folder, fname)
            data = np.load(path)
            all_data.append(data)

# Stack all files into a single array (assuming same shape per sample)
data = np.stack(all_data, axis=0)  # shape: (num_samples, num_features)
print (data.shape)

# Define groups
group1 = [0, 4, 5, 7, 8, 9, 10, 11, 13, 14]  # 10 variables
group1 = sorted(group1, key=lambda x: data[-1, x], reverse=True)

group2 = [1, 2, 3, 6, 12, 15]  # 6 variables (Note: 15 was in both groups, so I removed it)
group2 = sorted(group2, key=lambda x: data[-1, x], reverse=True)

# Create a 2x1 subplot layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Custom colors for each group (Viridis colormap)
n1 = len(group1)  # Number of variables in group1
n2 = len(group2)  # Number of variables in group2

# Generate Viridis colors for each group
colors_group1 = [to_hex(viridis(i / n1)) for i in range(n1)]
colors_group2 = [to_hex(viridis(i / n2)) for i in range(n2)]

# Plot group1 (top subplot)
for i, var_idx in enumerate(group1):
    ax1.plot(data[:, var_idx], color=colors_group1[i], lw=1.5, label=f'f{var_idx}')

ax1.set_title('(a)')
# ax1.grid(True)
ax1.legend(loc='lower center', ncol=3, fontsize=12) #, bbox_to_anchor=(1.15, 1))  # Place legend outside
ax1.set_xlim(0, 150)
ax1.set_ylim(0.0, 0.8)

# Plot group2 (bottom subplot)
for i, var_idx in enumerate(group2):
    ax2.plot(data[:, var_idx], color=colors_group2[i], lw=1.5, label=f'f{var_idx}')

ax2.set_title('(b)')
# ax2.grid(True)
ax2.legend(loc='upper center', ncol=2, fontsize=12) # , bbox_to_anchor=(1.15, 1))  # Place legend outside
ax2.set_xlim(0, 150)
ax2.set_ylim(0.6, 1.0)

plt.tight_layout()  # Adjust spacing
plt.show()
