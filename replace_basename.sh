#!/bin/bash

# Target folder
folder="data/datasets/d567/rtargets_temp"

# Change to target directory
cd "$folder" || { echo "Folder not found!"; exit 1; }

# Loop through matching files
for file in 00_000_*.npy; do
    # Check if file exists (in case the pattern doesn't match anything)
    [ -e "$file" ] || continue

    # New filename with 70_ prefix
    new_name="${file/00_/50_}"

    # Rename the file
    mv "$file" "$new_name"
    echo "Renamed: $file -> $new_name"
done
