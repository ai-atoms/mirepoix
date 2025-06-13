import os
from ase.io import read, write
from ase.atoms import Atoms

def load_lammps_with_real_symbols(input_path, custom_map):
    """Load LAMMPS dump file and remap ASE's automatic element assignments"""
    atoms = read(input_path, format='lammps-dump-text')

    # Verify all automatic symbols are in our mapping
    default_fake_symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    for fake_symbol in default_fake_symbols:
        if fake_symbol not in custom_map:
            raise ValueError(f"Missing mapping for fake symbol '{fake_symbol}'")

    # Replace with real symbols
    real_symbols = [custom_map[sym] for sym in atoms.get_chemical_symbols()]
    atoms.set_chemical_symbols(real_symbols)
    return atoms

def process_lammps_dump(input_file, output_file, type_map):
    """
    Process LAMMPS dump file: wrap atoms, remove all velocity info, remap types, save as CFG

    Parameters:
    - input_file: Path to LAMMPS dump file
    - output_file: Path for output CFG file
    - type_map: Dictionary mapping ASE's auto-assigned symbols to desired elements
                (e.g., {'H': 'Al', 'He': 'Co'} for type 1=Al, type 2=Co)
    """
    # Load and remap atom types
    atoms = load_lammps_with_real_symbols(input_file, type_map)

    # Wrap atoms back into the unit cell
    atoms.wrap()

    # Remove all velocity/momentum information
    atoms.set_momenta(None)  # Explicitly remove momenta

    # Write to CFG file (without velocities)
    write(output_file, atoms, format='cfg')
    print(f"Successfully processed and saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    # ASE's automatic assignment: type1=H, type2=He, type3=Li, etc.
    # Map these to your desired elements
    type_mapping = {
        'H': 'Al'}   # type 1 becomes Aluminum
        # 'He': 'X',  # type 2 becomes Cobalt
        # 'Li': 'X',  # type 3 becomes Nickel
        # Add more mappings as needed for your system

    # Process the files
    process_lammps_dump(
        input_file='md_config_252000.xyz',
        output_file='252000.cfg',
        type_map=type_mapping,
    )
