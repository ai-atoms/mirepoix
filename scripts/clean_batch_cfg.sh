#!/bin/bash

# Set directories
target_folder="data/fpa70/selected/cfg"
output_folder="data/fpa70/selected/cfg2"

# Create output directory if it doesn't exist
mkdir -p "$output_folder"

# Process each .cfg file in the target folder
for input_file in "$target_folder"/*.cfg; do
    # Get just the filename without path
    filename=$(basename "$input_file")

    # Remove 'md_config_' prefix if present (keeping the rest including .cfg)
    output_file="${filename#md_config_}"

    # Process the file with sed
    sed -e 's/entry_count = 6/.NO_VELOCITY.\nentry_count = 3/' \
        -e 's/  0.000000e+00 0.000000e+00 0.000000e+00/ /g' \
        "$input_file" > "$output_folder/$output_file"

    echo "Processed $filename → $output_file"
done

echo "All files processed and saved to $output_folder"
