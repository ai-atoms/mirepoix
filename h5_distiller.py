# Updated June 2025 by @camilofs
import os, shutil
import numpy as np

import pickle
import pandas as pd
import csv, h5py
from h5py import Group, File

from typing import List, Dict, TypedDict, Optional
from difflib import SequenceMatcher
from ase import Atoms
from ase.io import read

# camilofs imports
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # For progress bar
import random


class H5Manager :
    # From @lapointe
    def __init__(self, path_h5 : os.PathLike[str], mode : str = "writing") :
        self.path_h5 = path_h5
        self.mode = mode 
        self.h5_file : File = None

        if mode == "writing" :
            print('writing')
            self.init_h5()
        elif mode == "reading" : 
            self.read_h5()

    def close_h5(self) -> None : 
        self.h5_file.close()
        return 

    def init_h5(self) -> None : 
        """Init .h5 file for writing"""
        self.h5_file = h5py.File(self.path_h5, 'w')
        return 
    
    def read_h5(self) -> None : 
        """Init .h5 file reading"""
        self.h5_file = h5py.File(self.path_h5, "r")
        return

    def add_or_update_dfct_group(self, key_group : str,
                            data_config : Atoms,
                            extra_keys : List[str] = []) -> None :
        """Update the .h5 file 
        
        Parameters
        ----------

        hdf5_group : ```Group```
            ```Group``` associated to a given configuration

        data_config : ```Atoms```
            ```Atoms``` configuration to update in .h5 file

        extra_keys : List[str] 
            List of extra data key to update in .h5 file

        """
        positions = data_config.get_positions()
        cell = data_config.get_cell()[:]
        milady_descriptor = data_config.get_array('milady-descriptors')
        species = data_config.get_chemical_formula()

        if key_group not in self.h5_file:
            # Create group for configuration
            config_group = self.h5_file.create_group(key_group)
            config_group.create_dataset('positions', data=positions, compression="gzip", compression_opts=9)
            config_group.create_dataset('cell', data=cell, compression="gzip", compression_opts=9)
            config_group.create_dataset('milady-descriptors', data=milady_descriptor, compression="gzip", compression_opts=9)
            config_group.create_dataset('species', data=species)

            if len(extra_keys) > 0 : 
                for key in extra_keys : 
                    extra_data = data_config.get_array(key)
                    config_group.create_dataset(key, data=extra_data, compression="gzip", compression_opts=9)

        else : 
            config_group = self.h5_file[key_group]
            config_group.create_dataset('positions', data=positions, compression="gzip", compression_opts=9)
            config_group.create_dataset('cell', data=cell, compression="gzip", compression_opts=9)
            config_group.create_dataset('milady-descriptors', data=milady_descriptor, compression="gzip", compression_opts=9)
            config_group.create_dataset('species', data=species)

            if len(extra_keys) > 0 : 
                for key in extra_keys : 
                    extra_data = data_config.get_array(key)
                    config_group.create_dataset(key, data=extra_data, compression="gzip", compression_opts=9)

        return 
    
    def extract_atoms_object(self, key_group : str,
                             milady_desc : bool = True, 
                             extra_keys : List[str] = []) -> Atoms : 
        """Fill ```Atoms``` object based on .h5 data
        
        Parameters
        ----------

        key_group : str
            Key of the group to used to fill ```Atoms``` object

        milady_desc : bool 
            Milady descriptor is included in ```Atoms``` arrays

        extra_keys : List[str]
            Extra arrays to add in ```Atoms``` object

        Returns
        -------

        ```Atoms```
            ```Atoms``` object containing all arrays
        """


        h5_config = self.h5_file[key_group]
        cell = h5_config['cell'][()]
        positions = h5_config['positions'][()]
        species = h5_config['species'][()].decode("utf-8")

        ats = Atoms(symbols=species,
                    positions=positions,
                    cell=cell)
        
        if milady_desc : 
            milady_descriptor = h5_config['milady-descriptors'][()]
            ats.set_array('milady-descriptors', milady_descriptor, dtype=float)
        
        for key in extra_keys : 
            extra_data = h5_config[key][()]
            ats.set_array(key, extra_data, dtype=float)

        return ats


# -- inputs
path_h5 = 'data/fpa70/descriptor/fpa70_dist.h5'


# -- camilofs utility additions
def inspect_distance_bounds(h5_obj, distance_key):
    """
    Compute and print the global min and max of all distance arrays in the HDF5 file.
    """
    global_min = float('inf')  # Initialize with infinity
    global_max = -float('inf') # Initialize with negative infinity
    
    # Iterate through all entries
    for key, val in tqdm(h5_obj.h5_file.items(), desc="Processing entries"):
        try:
            distance_array = val[distance_key][:]
            
            # Update global min and max
            current_min = np.min(distance_array)
            current_max = np.max(distance_array)
            
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max
                
        except Exception as e:
            print(f"Error processing entry {key}: {str(e)}")
            continue
    
    # Print final results
    print(f"\nStats for: {distance_key}")
    print(f"  Minimum value: {global_min:.4f}")
    print(f"  Maximum value: {global_max:.4f}")


def export_distance_sample(h5_obj, distance_key, output_dir='temp/inspection', sample_size=1000):
    """
    Samples one entry of the given distance key and exports it to CSV for inspection.
    Handles both 1D (N,) and 2D (N,M) arrays, with special reporting for (N,2) cases.
    
    Args:
        h5_obj: HDF5 object containing the data
        distance_key: The distance key to inspect
        output_dir: Output directory for CSV file
        sample_size: Number of elements to sample (use None for full array)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find first valid entry with the distance key
    sample_entry = None
    sample_key = None
    sample_array = None
    
    for key, val in tqdm(h5_obj.h5_file.items(), desc="Finding first valid entry"):
        try:
            if distance_key in val:
                sample_entry = val
                sample_key = key
                sample_array = val[distance_key][:]
                break
        except Exception as e:
            print(f"Error checking entry {key}: {str(e)}")
            continue
    
    if sample_array is None:
        print(f"\nERROR: Distance key '{distance_key}' not found in any entry")
        return
    
    # Handle different array dimensionalities
    if sample_array.ndim == 1:
        # Standard 1D case (N,)
        if sample_size is not None and len(sample_array) > sample_size:
            step = len(sample_array) // sample_size
            sampled_data = sample_array[::step]
        else:
            sampled_data = sample_array
        
        df = pd.DataFrame({
            'index': np.arange(len(sampled_data)),
            'value': sampled_data
        })
        
    elif sample_array.ndim == 2:
        # 2D case (N,M)
        if sample_array.shape[1] == 2:
            # Special handling for (N,2) case
            print("\nFound 2D array with shape (N,2) - exporting both columns")
            
            if sample_size is not None and len(sample_array) > sample_size:
                step = len(sample_array) // sample_size
                sampled_data = sample_array[::step, :]
            else:
                sampled_data = sample_array
            
            df = pd.DataFrame({
                'index': np.arange(len(sampled_data)),
                'column_0': sampled_data[:, 0],
                'column_1': sampled_data[:, 1]
            })
        else:
            # General 2D case (N,M)
            print(f"\nFound 2D array with shape {sample_array.shape}")
            
            if sample_size is not None and len(sample_array) > sample_size:
                step = len(sample_array) // sample_size
                sampled_data = sample_array[::step, :]
            else:
                sampled_data = sample_array
            
            # Create multi-column DataFrame
            df = pd.DataFrame(sampled_data)
            df.columns = [f'col_{i}' for i in range(sampled_data.shape[1])]
            df.insert(0, 'index', np.arange(len(sampled_data)))
            
    else:
        print(f"\nERROR: Unsupported array dimensionality {sample_array.ndim}")
        return
    
    # Add shape information as metadata
    shape_info = f"Original shape: {sample_array.shape}, dtype: {sample_array.dtype}"
    df.attrs['shape_info'] = shape_info
    
    # Export to CSV
    output_path = os.path.join(output_dir, f"{distance_key}_sample.csv")
    df.to_csv(output_path, index=False)
    
    # Print comprehensive diagnostics
    print(f"\n=== Inspection Report for: {distance_key} ===")
    print(f"Entry key: {sample_key}")
    print(f"Array shape: {sample_array.shape}")
    print(f"Array dtype: {sample_array.dtype}")
    print(f"Sample size: {len(df)} rows")
    
    if sample_array.ndim == 2:
        print("\nColumn-wise statistics:")
        for i in range(sample_array.shape[1]):
            col = sample_array[:, i]
            print(f"  Column {i}: min={np.min(col):.4f}, max={np.max(col):.4f}, "
                  f"mean={np.mean(col):.4f}, NaN={np.isnan(col).sum()}")
    
    print(f"\nOutput saved to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    return output_path


def gen_feature_vector(
    distance_dict: Dict[str, np.ndarray],
    distance_keys: List[str],
    n_features: int = 1280,
    condition_file: str = "conditions.txt",
    random_seed: Optional[int] = None
    ) -> np.ndarray:
    '''
    Generates a feature vector with per-feature value ranges.
    Special handling for first distance key if it is a (N, 2) array.
    '''
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Define max ranges for each internal key
    feature_max_range = {
        'd1': 1920,
        'd1b': 1920,
        'd2': 2048,
        'd3': 768,
        'd4': 2816,
        'd5': 768,
        'd6': 3584,
    }

    if len(distance_keys) == 0:
        raise ValueError("No distance keys provided")
    
    internal_keys = []
    key_mapping = {}
    
    first_key = distance_keys[0]
    first_array = distance_dict.get(first_key, np.zeros(0))
    
    if first_array.ndim == 2 and first_array.shape[1] == 2:
        internal_keys.extend(['d1', 'd1b'])
        key_mapping['d1'] = (first_key, 0)
        key_mapping['d1b'] = (first_key, 1)
    else:
        internal_keys.append('d1')
        key_mapping['d1'] = (first_key, None)
    
    for i in range(1, min(6, len(distance_keys))):
        key = f'd{i+1}'
        internal_keys.append(key)
        key_mapping[key] = (distance_keys[i], None)
    
    operations = ['AND', 'OR', 'NOT', 'NAND', 'NOR', 'XOR', 'XNOR']
    comparisons = ['<', '>', '<=', '>=']
    
    array_length = len(first_array)
    feature_vector = np.zeros(n_features)
    conditions = []
    
    for i in range(n_features):
        d1 = random.choice(internal_keys)
        d2 = random.choice(internal_keys)
        comp1 = random.choice(comparisons)
        comp2 = random.choice(comparisons)
        
        # Sample val1 and val2 using feature-specific ranges
        val1 = random.uniform(0, feature_max_range[d1])
        val2 = random.uniform(0, feature_max_range[d2])
        
        op = random.choice(operations)
        
        ext_key1, col1 = key_mapping[d1]
        arr1 = distance_dict.get(ext_key1, np.zeros(array_length))
        if col1 is not None:
            arr1 = arr1[:, col1]
        
        ext_key2, col2 = key_mapping[d2]
        arr2 = distance_dict.get(ext_key2, np.zeros(array_length))
        if col2 is not None:
            arr2 = arr2[:, col2]
        
        # Format condition string
        if op == 'NOT':
            condition_str = f"NOT ({d1} {comp1} {val1:.2f})"
        else:
            condition_str = f"({d1} {comp1} {val1:.2f}) {op} ({d2} {comp2} {val2:.2f})"
        conditions.append(condition_str)
        
        # Evaluate condition
        cond1 = eval(f"arr1 {comp1} val1")
        if op != 'NOT':
            cond2 = eval(f"arr2 {comp2} val2")
        
        if op == 'AND':
            result = cond1 & cond2
        elif op == 'OR':
            result = cond1 | cond2
        elif op == 'NOT':
            result = ~cond1
        elif op == 'NAND':
            result = ~(cond1 & cond2)
        elif op == 'NOR':
            result = ~(cond1 | cond2)
        elif op == 'XOR':
            result = cond1 ^ cond2
        elif op == 'XNOR':
            result = ~(cond1 ^ cond2)
        
        feature_vector[i] = np.sum(result) / array_length

    # Write out the logic to a file
    with open(condition_file, 'w') as f:
        f.write(f"# Special mapping: {key_mapping}\n")
        for i, condition in enumerate(conditions):
            f.write(f"Feature {i}: {condition}\n")
    
    return feature_vector


def create_combined_vector(h5_obj, distance_keys, csv=True, output_dir='temp/outputs', 
                          n_features=1280, random_seed=42):
    """
    Processes H5 entries to:
    1. Generate feature vectors using random conditions
    2. Save individual vectors as .npy files
    3. Optionally create combined CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    condition_file=str(f'conditions_{random_seed}.txt')
    condition_output_path = os.path.join(output_dir, condition_file)
    
    # First pass: Process and save individual .npy files
    for key, val in tqdm(h5_obj.h5_file.items(), desc="Processing entries"):
        try:
            # Convert h5 group to distance dictionary format
            distance_dict = {dk: val[dk][:] for dk in distance_keys if dk in val}
            
            # Generate feature vector with random conditions
            features = gen_feature_vector(
                distance_dict,
                distance_keys,
                n_features=n_features,
                condition_file=condition_output_path,
                random_seed=random_seed
            )
            
            # Save individual .npy file
            np.save(os.path.join(output_dir, f'{key}.npy'), features)
            
            # Prepare CSV row if needed
            if csv:
                row = [key] + features.tolist()
                results.append(row)
            
        except Exception as e:
            print(f"Skipping {key} - Error: {str(e)}")
            continue
    
    # Second pass: Save CSV if requested
    if csv and results:
        headers = ["key"] + [f"feature_{i}" for i in range(n_features)]
        csv_path = os.path.join(output_dir, 'combined_features.csv')
        
        # Write CSV with headers
        with open(csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            for row in results:
                f.write(','.join(map(str, row)) + '\n')
    
    return distance_keys, results if csv else None



# -- start (sanity check)
h5_obj = H5Manager(path_h5, mode = 'reading')

first_key = next(iter(h5_obj.h5_file.keys()))
first_obj = h5_obj.h5_file[first_key]
dist_keys = [k for k in first_obj.keys() if 'dist' in k]
print(f'Available keys are {dist_keys}')

'''
# -- testing utility functions
selected = dist_keys[5]
print(f'\nInspecting {selected}')
inspect_distance_bounds(
    h5_obj=h5_obj,
    distance_key=str(selected))

# Extract the specific distance array
distance_array = first_obj[selected][:]
print(f"Shape of {selected}:", distance_array.shape)


# sample (checking)
export_distance_sample(
    h5_obj=h5_obj,
    distance_key=str(selected))
'''

create_combined_vector(h5_obj, dist_keys, n_features=1024, random_seed=169006) # (n_features,)

# -- end