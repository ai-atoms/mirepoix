#!/bin/bash

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

target_dir="$1"

# Check if the directory exists
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist."
    exit 1
fi

# Loop through all files matching the pattern md_config_*.cfg in the target directory
for file in "$target_dir"/md_config_*.cfg; do
    # Check if any files match the pattern (otherwise the loop will run once with the literal pattern)
    if [ -f "$file" ]; then
        # Extract the NNNNNN part using basename and parameter expansion
        new_name=$(basename "$file" | sed 's/^md_config_//')
        # Rename the file
        mv -v "$file" "$target_dir/$new_name"
    fi
done

echo "Renaming complete."