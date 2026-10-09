"""Generate the random-logic feature vectors from distortion-state distributions.

Each feature is the fraction of atoms satisfying a random logical condition on one or
two of the 7 distortion-state channels (e.g. "(d2 < 512.30) XOR (d5 >= 100.10)").
The conditions depend only on the seed, so they are identical for every configuration.
Smaller feature sets (targets_N) are the first N features of the full vector.
"""
import argparse
import glob
import json
import operator as op
import os
import random
from multiprocessing import Pool

import numpy as np
from natsort import natsorted

CHANNELS = ['d1', 'd1b', 'd2', 'd3', 'd4', 'd5', 'd6']
MAX_RANGE = {'d1': 1920, 'd1b': 1920, 'd2': 2048, 'd3': 768, 'd4': 2816, 'd5': 768, 'd6': 3584}
OPERATIONS = ['AND', 'OR', 'NOT', 'NAND', 'NOR', 'XOR', 'XNOR']
COMPARISONS = {'<': op.lt, '>': op.gt, '<=': op.le, '>=': op.ge}
LOGIC = {
    'AND': lambda x, y: x & y,
    'OR': lambda x, y: x | y,
    'NOT': lambda x, y: ~x,
    'NAND': lambda x, y: ~(x & y),
    'NOR': lambda x, y: ~(x | y),
    'XOR': lambda x, y: x ^ y,
    'XNOR': lambda x, y: ~(x ^ y),
}


def make_conditions(n_features, seed):
    random.seed(seed)
    conditions = []
    for _ in range(n_features):
        d1 = random.choice(CHANNELS)
        d2 = random.choice(CHANNELS)
        c1 = random.choice(list(COMPARISONS))
        c2 = random.choice(list(COMPARISONS))
        v1 = random.uniform(0, MAX_RANGE[d1])
        v2 = random.uniform(0, MAX_RANGE[d2])
        conditions.append((d1, c1, v1, d2, c2, v2, random.choice(OPERATIONS)))
    return conditions


def describe(cond):
    d1, c1, v1, d2, c2, v2, o = cond
    if o == 'NOT':
        return f'NOT ({d1} {c1} {v1:.2f})'
    return f'({d1} {c1} {v1:.2f}) {o} ({d2} {c2} {v2:.2f})'


def compute_features(dsv, conditions):
    index = {c: i for i, c in enumerate(CHANNELS)}
    out = np.zeros(len(conditions))
    for i, (d1, c1, v1, d2, c2, v2, o) in enumerate(conditions):
        x = COMPARISONS[c1](dsv[index[d1]], v1)
        y = COMPARISONS[c2](dsv[index[d2]], v2) if o != 'NOT' else None
        out[i] = LOGIC[o](x, y).sum() / dsv.shape[1]
    return out


def _process(args):
    path, conditions = args
    return os.path.splitext(os.path.basename(path))[0], compute_features(np.load(path), conditions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dsv', default='data/fpa70a/descriptor/dsv', help='folder with <key>.npy of shape (7, N)')
    ap.add_argument('--out', default='data/fpa70a/features')
    ap.add_argument('--seed', type=int, default=169006142)
    ap.add_argument('--n-features', type=int, default=15360)
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[256, 512, 1024, 2048, 4096, 8192, 11520, 12800, 15360],
                    help='sizes of the targets_N subsets (first N features)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    conditions = make_conditions(args.n_features, args.seed)
    root = os.path.join(args.out, f'rs_{args.seed}')
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f'conditions_{args.seed}.txt'), 'w') as fh:
        fh.write('\n'.join(f'Feature {i}: {describe(c)}' for i, c in enumerate(conditions)) + '\n')
    with open(os.path.join(root, 'channels.json'), 'w') as fh:
        json.dump(CHANNELS, fh)

    sizes = sorted(s for s in args.sizes if s <= args.n_features)
    for s in sizes:
        os.makedirs(os.path.join(root, f'targets_{s}'), exist_ok=True)

    files = natsorted(glob.glob(os.path.join(args.dsv, '*.npy')))
    with Pool(args.workers) as pool:
        for key, feats in pool.imap(_process, [(f, conditions) for f in files]):
            for s in sizes:
                np.save(os.path.join(root, f'targets_{s}', f'{key}.npy'), feats[:s])
            print(f'{key} done', flush=True)


if __name__ == '__main__':
    main()
