import os
import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt

from tqdm import tqdm
from natsort import natsorted

base_path = 'results/feat_gen/rs_169006142'
target = 'targets_15360'
target_path = os.path.join(base_path, target)

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

# Export
np.save(f'{base_path}/{target}.npy', s)