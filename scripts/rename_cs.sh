#!/bin/bash

# Check if directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Navigate to the directory
cd "$1" || exit

# Loop through all PNG files and rename them
for file in *_*.png; do
    # Extract base name (e.g., "00_000_000002_8" → "00_000_000002")
    new_name=$(echo "$file" | sed 's/_[0-9]\+\.png$/.png/')
    
    # Rename the file (only if new_name is different)
    if [ "$file" != "$new_name" ]; then
        mv -v "$file" "$new_name"
    fi
done

echo "Renaming complete!"
