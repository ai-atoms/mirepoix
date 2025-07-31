import os
import json
import numpy as np
import pandas as pd
from natsort import natsorted


def reduce_feature_space_from_npy(input_folder, selected_json_path, output_dir="temp/reduced"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load selected features from the JSON
    with open(selected_json_path) as f:
        selected_features = json.load(f)

    # 2. Convert to sorted list of integers
    selected_features_sorted = sorted([int(x) for x in selected_features])

    # 3. Get and sort .npy files
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

        # Save reduced features as .npy
        np.save(os.path.join(output_dir, f"{key}.npy"), reduced_features)

        # Append to list for CSV export
        keys.append(key)
        reduced_data.append(reduced_features)

    # 5. Export combined CSV
    selected_columns = [f"feature_{i}" for i in selected_features_sorted]
    reduced_df = pd.DataFrame(reduced_data, columns=selected_columns)
    reduced_df.insert(0, "key", keys)
    reduced_df.to_csv(os.path.join(output_dir, "reduced_features.csv"), index=False)

    return selected_features_sorted


def remap_feature_names_in_csv(csv_path, original_json_path):
    # Load the original names
    with open(original_json_path) as f:
        original_names = json.load(f)

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Build new column names
    new_columns = []
    for col in df.columns:
        if col.startswith("feature_"):
            index = int(col.replace("feature_", ""))
            if index < len(original_names):
                new_name = f"feature_{original_names[index]}"
            else:
                new_name = col  # fallback if out of range
            new_columns.append(new_name)
        else:
            new_columns.append(col)  # keep other columns like 'key'

    # Assign new column names and save
    df.columns = new_columns
    df.to_csv(csv_path, index=False)

    return new_columns


reduce_feature_space_from_npy(
    input_folder="data/datasets/d567/rtargets",
    selected_json_path="data/datasets/d567/r2_features/selected_features.json",
    output_dir="data/datasets/d567/r2targets"
)

'''
# in case you need to recover the original indexes
csv_path = "data/datasets/d567/r2targets/reduced_features.csv"
original_json = "data/datasets/d567/r2_features/original.json"

renamed_columns = remap_feature_names_in_csv(csv_path, original_json)
print("Renamed columns:", renamed_columns)
'''