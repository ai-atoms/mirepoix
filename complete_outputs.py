import os
import subprocess
import time

data_dir = './data'
output_dir = './outputs'
ctf_cs = '2e-6' # select the desired script
ctf_print = str(int(float(ctf_cs)*1e6))

# Get all .cfg files from the data directory
data_files = [f for f in os.listdir(data_dir) if f.endswith('.cfg')]

# Get all .png files from the outputs directory
output_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
output_files_names = {os.path.splitext(f)[0] for f in output_files}

# Iterate over each data file and check if corresponding output exists
for data_file in data_files:
    base_name = os.path.splitext(data_file)[0]
    name_with_ctf = '{}_{}'.format(base_name, ctf_print)

    if name_with_ctf not in output_files_names:
        print(f"Running calculation for {data_file}...")
        subprocess.run(['python3', './scripts/bf_gpu_896gpts.py', base_name, ctf_cs]) # adjust resolution
        time.sleep(30) # adjust cooldown time
