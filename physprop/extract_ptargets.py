import pandas as pd
import numpy as np
import ast
import os

# --- Load and preprocess CSV ---
df = pd.read_csv('data/datasets/d567b/pcomp_d567b.csv', encoding="utf-7")

# Fix C15 format: convert to comma-separated list strings, then wrap with brackets
df['C15'] = df['C15'].str.replace(r'[\[\]\s]+', ',', regex=True).str.strip(',').apply(lambda x: '[' + x + ']')

# Output folder
output_dir = 'data/datasets/d567b/ptargets'
os.makedirs(output_dir, exist_ok=True)

# --- Export .npy files ---
for _, row in df.iterrows():
    vec = np.zeros(8, dtype=np.float32)

    # Parse and insert C15 values (truncate if longer, pad if shorter)
    c15 = sorted([float(x) for x in ast.literal_eval(row['C15'])])
    if len(c15) == 0:
        vec[0] = 0.0  # mean
        vec[1] = 0.0  # variance
        vec[2] = 0.0  # max
    else:
        c15 = np.array(c15, dtype=np.float32)
        vec[0] = c15.mean()
        vec[1] = c15.var()
        vec[2] = c15.max()

    # Add 5 additional features
    vec[3] = row['nvpa']
    vec[4] = row['d_111']
    vec[5] = row['n_111']
    vec[6] = row['d_100']
    vec[7] = row['n_100']

    # Save using the key as filename
    np.save(os.path.join(output_dir, f"{row['key']}.npy"), vec)
