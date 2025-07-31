import os
import numpy as np
import pandas as pd
from natsort import natsorted

def reduce_feature_space_from_txt(input_folder, selected_txt_path, output_dir="temp/reduced"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load selected features from TXT file
    with open(selected_txt_path, "r") as f:
        selected_features = [int(line.strip().replace("feature_", "")) for line in f if line.strip()]

    # 2. Sort selected feature indices
    selected_features_sorted = sorted(selected_features)

    # 3. Get and sort all .npy files
    npy_files = natsorted([
        fname for fname in os.listdir(input_folder) if fname.endswith(".npy")
    ])

    reduced_data = []
    keys = []

    # 4. Process each .npy file
    for fname in npy_files:
        key = os.path.splitext(fname)[0]
        features = np.load(os.path.join(input_folder, fname))

        # Select only the desired features
        reduced_features = features[selected_features_sorted]

        # Save reduced vector as .npy
        np.save(os.path.join(output_dir, f"{key}.npy"), reduced_features)

        # Optional: collect data for summary CSV
        keys.append(key)
        reduced_data.append(reduced_features)

    # 5. Export CSV summary
    column_names = [f"feature_{i}" for i in selected_features_sorted]
    reduced_df = pd.DataFrame(reduced_data, columns=column_names)
    reduced_df.insert(0, "key", keys)
    reduced_df.to_csv(os.path.join(output_dir, "reduced_features.csv"), index=False)

    return selected_features_sorted

reduce_feature_space_from_txt(
    input_folder="data/datasets/d567/rtargets",
    selected_txt_path="data/datasets/d567/top16_hd.txt",
    output_dir="data/datasets/d567/r3targets"
)
