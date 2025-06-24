import os 
from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
from ovito.pipeline import *

import numpy as np
import pandas as pd
from natsort import natsorted

data_root = 'data/fpa70/selected/cfg2'
output_file = 'ovito_results_f70.csv'

# -- import
file_list = []
for file in os.listdir(data_root):
    file_list.append(data_root + '/' + file)

file_list = natsorted(file_list)

pipeline = import_file(file_list)
print(f'Working with {pipeline.source.num_frames} frames...')


# -- Wigner-Seitz analysis
ws_mod = WignerSeitzAnalysisModifier(
    affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference,
    reference_frame = 0
)

pipeline.modifiers.append(ws_mod)

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['frame', 'num_vacancies'])

# Analyze all frames (WS)
for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)
    
    # Initialize record
    record = {'frame': frame, 'num_vacancies': 0}
    
    # Check if analysis was successful
    if 'Occupancy' not in data.particles:
        print(f"Frame {frame}: Wigner-Seitz analysis failed")
        results_df = results_df.append(record, ignore_index=True)
        continue
    
    # Get occupancy numbers
    occupancies = data.particles['Occupancy']
    
    # Count vacancies and get positions
    record['num_vacancies'] = np.count_nonzero(occupancies == 0)
    
    # Append to DataFrame
    results_df = results_df._append(record, ignore_index=True)
    print(f"Frame {frame}: {record['num_vacancies']} vacancies")

print (results_df.tail())


# -- DXA (for BCC structure)
pipeline = import_file(file_list) # reset the pipeline
print(f'Working with {pipeline.source.num_frames} frames...')

dxa_mod = DislocationAnalysisModifier(
    input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC,
)
pipeline.modifiers.append(dxa_mod)

# Initialize DataFrame
columns = ['frame', 'd_111', 'n_111', 
           'd_100', 'n_100']
dxa_results = pd.DataFrame(columns=columns)

for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)
    cell_volume = data.attributes['DislocationAnalysis.cell_volume']

    # Initialize counts and lengths
    count_1_2_111 = 0
    count_100 = 0

    # Get total lengths from attributes
    try:
        length_1_2_111 = data.attributes['DislocationAnalysis.length.1/2<111>']
        length_100 = data.attributes['DislocationAnalysis.length.<100>']
    except KeyError:
        print(f"Frame {frame}: DXA attributes missing")

    # Count dislocation types
    if hasattr(data, 'dislocations') and hasattr(data.dislocations, 'segments'):
        for segment in data.dislocations.segments:
            burg_vec = segment.true_burgers_vector
            # Check for <100> type (contains 1 or -1)
            if np.any(np.abs(burg_vec) == 1.0):
                count_100 += 1
            else:
                count_1_2_111 += 1

    # Store results
    dxa_results = dxa_results._append({
        'frame': frame,
        'd_111': length_1_2_111 / cell_volume,
        'n_111': count_1_2_111,
        'd_100': length_100 / cell_volume,
        'n_100': count_100
    }, ignore_index=True)

    print(f"Frame {frame} analyzed - 1/2<111>: {count_1_2_111}, <100>: {count_100}")


# -- Final export
combined_results = pd.merge(results_df, dxa_results, on='frame', how='outer')

# Save to CSV (using semicolon as delimiter)
combined_results.to_csv(output_file, index=False, sep=';')

# Optional: Print confirmation and show first few rows
print("Results successfully saved to csv")
print("\nFirst few rows of combined results:")
print(combined_results.head())
