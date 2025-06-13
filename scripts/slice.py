from ase import Atoms
from ase.geometry import wrap_positions
import numpy as np


def split_box_2x2_with_overlap(atoms: Atoms, lattice_param: float):
    """Split a periodic MD box into 2x2 overlapping XY quadrants for image generation.

    Parameters:
        atoms (ase.Atoms): Input system.
        lattice_param (float): Base lattice parameter to define overlap (added with small tolerance).

    Returns:
        List[ase.Atoms]: Four overlapping sub-boxes in XY plane.
    """
    overlap = lattice_param + 1e-2
    cell = atoms.get_cell()
    a, b, c = cell[0], cell[1], cell[2]
    full_pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Box lengths in x and y
    Lx = np.linalg.norm(a)
    Ly = np.linalg.norm(b)

    # Midpoints
    x_m = Lx / 2
    y_m = Ly / 2

    # Define 4 boxes as (x_min, x_max, y_min, y_max)
    boxes = [
        (0 - overlap, x_m + overlap, 0 - overlap, y_m + overlap),        # Q0 = (1,1)
        (x_m - overlap, Lx + overlap, 0 - overlap, y_m + overlap),       # Q1 = (2,1)
        (0 - overlap, x_m + overlap, y_m - overlap, Ly + overlap),       # Q2 = (1,2)
        (x_m - overlap, Lx + overlap, y_m - overlap, Ly + overlap),      # Q3 = (2,2)
    ]

    atoms_quadrants = []

    # Wrap positions first
    wrapped_pos = wrap_positions(full_pos, cell)

    for x_min, x_max, y_min, y_max in boxes:
        selected_pos = []
        selected_symbols = []

        for i, pos in enumerate(wrapped_pos):
            x, y, z = pos

            # Apply periodic shift to match overlap logic
            if x < x_min:
                x += Lx
            elif x > x_max:
                x -= Lx
            if y < y_min:
                y += Ly
            elif y > y_max:
                y -= Ly

            if x_min <= x <= x_max and y_min <= y <= y_max:
                selected_pos.append([x - x_min, y - y_min, z])  # Local position
                selected_symbols.append(symbols[i])

        # Define new cell with same z but adjusted x, y
        new_cell = np.array([
            [x_max - x_min, 0, 0],
            [0, y_max - y_min, 0],
            [0, 0, c[2]]
        ])

        sub_atoms = Atoms(selected_symbols, positions=selected_pos, cell=new_cell, pbc=True)
        atoms_quadrants.append(sub_atoms)

    return atoms_quadrants


# -- start
from ase.build import bulk

# Build a periodic structure (e.g., FCC Cu)
atoms = bulk('Cu', cubic=True).repeat((16, 16, 4))
N = len(atoms)

# Randomly choose 20% of atom indices
np.random.seed(42)  # for reproducibility
ni_indices = np.random.choice(N, size=N // 5, replace=False)

# Assign species: Cu everywhere, then Ni in selected positions
symbols = ['Cu'] * N
for i in ni_indices:
    symbols[i] = 'Ni'

atoms.set_chemical_symbols(symbols)
atoms.write('quadrant_4.xyz')

# Define lattice parameter for Cu (in Ångström)
lattice_param = atoms.get_cell()[0, 0]
lattice_param = lattice_param / 16
print (f'a = {lattice_param} Angstrom')

# Generate overlapping quadrants
quadrants = split_box_2x2_with_overlap(atoms, lattice_param)

# Save each for inspection
for i, q in enumerate(quadrants):
    q.write(f'quadrant_{i}.xyz')

