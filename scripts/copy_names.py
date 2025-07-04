import os
import re
from pathlib import Path

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

def rename_files_based_on_reference(source_dir, reference_dir, output_dir=None):
    """
    Renames files in source_dir based on the names in reference_dir, using natural sorting.
    
    Args:
        source_dir: Path to directory with files to be renamed
        reference_dir: Path to directory with reference file names
        output_dir: Optional path to output directory (if None, renames in place)
    """
    # Get lists of files with natural sorting
    source_files = sorted(Path(source_dir).iterdir(), key=natural_sort_key)
    reference_files = sorted(Path(reference_dir).iterdir(), key=natural_sort_key)
    
    # Check if we have matching file counts
    if len(source_files) != len(reference_files):
        print(f"Warning: Different file counts - source: {len(source_files)}, reference: {len(reference_files)}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = source_dir
    else:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Rename files
    for i, (src, ref) in enumerate(zip(source_files, reference_files)):
        if src.is_file():
            new_name = ref.stem + src.suffix  # Keep original extension
            new_path = Path(output_dir) / new_name
            
            # Handle name collisions
            counter = 1
            while new_path.exists():
                new_name = f"{ref.stem}_{counter}{src.suffix}"
                new_path = Path(output_dir) / new_name
                counter += 1
            
            # Rename (or copy) the file
            if output_dir == source_dir:
                src.rename(new_path)
                print(f"Renamed: {src.name} -> {new_name}")
            else:
                import shutil
                shutil.copy2(src, new_path)
                print(f"Copied: {src.name} -> {new_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rename files in one folder based on names in another folder"
    )
    parser.add_argument("source_dir", help="Directory containing files to rename")
    parser.add_argument("reference_dir", help="Directory containing reference names")
    parser.add_argument("--output_dir", help="Optional output directory (default is to rename in place)")
    
    args = parser.parse_args()
    
    rename_files_based_on_reference(args.source_dir, args.reference_dir, args.output_dir)
