"""Export per-configuration distortion-state distributions (7, N) from the descriptor HDF5.

Channel order (see CHANNELS): the two columns of the GMM distance, followed by the
Mahalanobis (6-SIA and SIA, inner and skin) and MCD distances.
Only the distance arrays are exported; atomic positions and raw descriptors are not.
"""
import argparse
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

CHANNELS = ['d1', 'd1b', 'd2', 'd3', 'd4', 'd5', 'd6']
SOURCES = [
    ('gmm-distance-Refs_c15_bulk', 0),
    ('gmm-distance-Refs_c15_bulk', 1),
    ('maha-distance-Refs_6sia_c15_inner', None),
    ('maha-distance-Refs_6sia_c15_skin', None),
    ('maha-distance-Refs_sia_c15_inner', None),
    ('maha-distance-Refs_sia_c15_skin', None),
    ('mcd-distance-Auto_bulk_BCC', None),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h5', default='data/fpa70/descriptor/fpa70_dist.h5')
    ap.add_argument('--out', default='data/fpa70a/descriptor/dsv')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.h5, 'r') as f:
        for key in tqdm(list(f.keys()), desc='Exporting'):
            grp = f[key]
            rows = [grp[name][:] if col is None else grp[name][:, col] for name, col in SOURCES]
            np.save(os.path.join(args.out, f'{key}.npy'), np.stack(rows))

    with open(os.path.join(args.out, 'channels.json'), 'w') as fh:
        json.dump({c: f'{n}[{col}]' if col is not None else n
                   for c, (n, col) in zip(CHANNELS, SOURCES)}, fh, indent=2)


if __name__ == '__main__':
    main()
