import os
import json
import numpy as np
import pandas as pd
from random import sample


def check_feature_ranges(csv_path, lower_bound, upper_bound):
    """
    Checks which features in a CSV file have values within the specified range.
    
    Parameters:
    csv_path (str): Path to the CSV file
    lower_bound (float): Lower bound to check against
    upper_bound (float): Upper bound to check against
    
    Returns:
    dict: Dictionary with feature names as keys and boolean indicating if any value falls in the range
    """
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Initialize results dictionary
    results = {}
    
    # Iterate through each feature
    for feature in df.columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
            
        # Check if any value in the feature falls within the range
        in_range = ((df[feature] > lower_bound) & (df[feature] < upper_bound)).any()
        results[feature] = in_range
    
    return results


def filter_feature_file(input_file_path, output_file_path, selected_features):
    """
    Filters a feature definition file to only include selected features.
    
    Args:
        input_file_path: Path to the input text file with all features
        output_file_path: Path where the filtered file will be written
        selected_features: Dictionary of {'feature_X': bool} indicating which features to keep
    """
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Check if line starts with "Feature"
            if line.startswith("Feature"):
                # Extract the feature number
                parts = line.split(":")
                feature_key = f"feature_{parts[0].split()[1]}"
                
                # Check if this feature is marked True in our selection dict
                if selected_features.get(feature_key, False):
                    outfile.write(line)
            else:
                # Write non-feature lines (like "Format" header) if needed
                # If you want to skip them, remove this else clause
                outfile.write(line)


def reduce_feature_space(csv_file_path, selected_json_path, output_dir="temp/reduced"):
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    keys = df.iloc[:, 0]  # First column contains keys
    features = df.iloc[:, 1:]  # Remaining columns are features
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load selected features from the json
    with open(selected_json_path) as f:
        selected_features = json.load(f)
    
    # 2. Convert to sorted list of integers and sort them
    selected_features_sorted = sorted([int(x) for x in selected_features])
    
    # 3. Create column names for the selected features
    selected_columns = [f'feature_{i}' for i in selected_features_sorted]
    
    # 4. Filter the dataframe to only keep selected columns
    reduced_features = features[selected_columns]
    
    # 5. Create combined dataframe with keys and reduced features
    reduced_df = pd.concat([keys, reduced_features], axis=1)
    
    # 6. Export combined features CSV
    output_csv_path = os.path.join(output_dir, 'reduced_features.csv')
    reduced_df.to_csv(output_csv_path, index=False)
    
    # 7. Export each row as NPY file (after sorting)
    for idx, row in reduced_df.iterrows():
        key = row.iloc[0]  # Get the key
        feature_values = row.iloc[1:].values.astype(np.float32)  # Get feature values
        
        # Save as NPY file
        output_npy_path = os.path.join(output_dir, f'{key}.npy')
        np.save(output_npy_path, feature_values)
    
    return selected_features_sorted


# Parameters
root_dir = 'data/datasets/d567'
csv_file_path = f"{root_dir}/features/combined_features.csv" 
json_file_path = f"{root_dir}/features/combined.json"

selected_features = reduce_feature_space(
    csv_file_path=csv_file_path,
    selected_json_path=json_file_path,
    output_dir=f"{root_dir}/reduced_targets"
)