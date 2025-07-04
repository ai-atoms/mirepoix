import os
from glob import glob

from utils import process_lammps_dump

# Define your type mapping (ASE's auto-assigned symbols to your desired elements)
type_mapping = {
    'H': 'Fe'} #, # type 1 becomes Aluminum
    # 'He': 'Co', # type 2 becomes Cobalt
    # 'Li': 'Ni', # type 3 becomes Nickel
    # Add more as needed

# Input and output directories
input_dir = 'data/fpa60/selected'
output_dir = 'data/fpa60/selected_cfg'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all .dump files in the input directory
for dump_file in glob(os.path.join(input_dir, '*.xyz')):
    # Create output filename (same name but with .cfg extension)
    base_name = os.path.basename(dump_file)
    output_file = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.cfg')

    # Process the file
    process_lammps_dump(
        input_file=dump_file,
        output_file=output_file,
        type_map=type_mapping
    )
    print(f"Processed: {base_name} → {os.path.basename(output_file)}")
