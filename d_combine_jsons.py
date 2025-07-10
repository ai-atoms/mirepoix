import json
import numpy as np
from pathlib import Path

def combine_feature_jsons(json_paths, output_path="combined_features.json", target_size=1536):
    """
    Combine multiple feature JSON files into one sorted, unique list padded to target_size.
    
    Args:
        json_paths (list): List of paths to JSON files containing feature indices
        output_path (str): Path to save the combined JSON file
        target_size (int): Desired number of features (default: 1536)
    Returns:
        list: Sorted list of unique combined features padded to target_size
    """
    unique_features = set()
    
    # 1. Load all JSON files and collect unique features
    for json_path in json_paths:
        with open(json_path) as f:
            features = json.load(f)
            if isinstance(features, list):
                unique_features.update(features)
            else:
                unique_features.add(features)
    
    # Convert to sorted list of integers
    combined_features = sorted(int(x) for x in unique_features)
    print ('Original size (without padding):')
    print (len(combined_features))
    
    # 2. Pad to target size if needed
    if len(combined_features) < target_size:
        min_val = min(combined_features)
        max_val = max(combined_features)
        
        # Generate additional unique features within the range
        while len(combined_features) < target_size:
            new_feature = np.random.randint(min_val, max_val + 1)
            if new_feature not in combined_features:
                combined_features.append(new_feature)
        
        # Re-sort after adding new features
        combined_features = sorted(combined_features)
    
    # 3. Save the combined features
    with open(output_path, 'w') as f:
        json.dump(combined_features, f, indent=2)
    
    return combined_features

# Example usage:
if __name__ == "__main__":
    json_files = [
        "data/datasets/d567/features/selected_f50.json",
        "data/datasets/d567/features/selected_f60.json",
        "data/datasets/d567/features/selected_f70.json",
    ]
    
    combined = combine_feature_jsons(json_files, 
                output_path='data/datasets/d567/features/combined.json',
                target_size=640)
    print(f"Generated {len(combined)} features (target: 640)")

