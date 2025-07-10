#!/bin/bash

# Target folder
folder="data/fpa70/selected/images_ovito"

# Change to the target directory
cd "$folder" || { echo "Folder not found!"; exit 1; }

# Range and constants
start=0
end=150
base=4000
step=4000

# Loop through the range
for ((i=start; i<=end; i++)); do
    i_padded=$(printf "%05d" "$i")
    old_name="fpa7${i_padded}.png"

    new_index=$((base + i * step))
    new_name="${new_index}_8.png"

    if [[ -f "$old_name" ]]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name -> $new_name"
    else
        echo "File not found: $old_name"
    fi
done
