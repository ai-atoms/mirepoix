"""Distil the random-logic features into compact DSV vectors (u_640, u_64, u_16).

  select  SVD-based feature selection. The covariance of the stacked feature vectors is
          decomposed; every component whose singular value exceeds --threshold
          contributes its --k largest-|loading| features. The union is written as a JSON list.
  reduce  Keep the listed feature indices of every <key>.npy in a folder.
  combine Union of several index lists, padded with random indices up to --size.

Chain used for the paper (indices always refer to the input vector of each step):
  select  targets_15360 (per box)           --k 16 --threshold 1e-6  -> selected_f<box>.json
  (union of the per-box selections, padded with random indices up to 640 -> combined.json)
  reduce  targets_15360 + u640.json                                  -> u_640
  select  u_640 (all boxes stacked)         --k 4  --threshold 1e-6  -> positions in u_640 (= u64.json)
  reduce  targets_15360 + u64.json                                   -> u_64
  reduce  targets_15360 + u16.json (16 of the 64, large amplitude)   -> u_16
Index files in data/npj_components/ hold indices into the original 15360 features, together with
the feature conditions as listed in the paper tables.
"""
import argparse
import json
import os

import numpy as np
from natsort import natsorted


def load_stack(folders, prefix=None):
    files = []
    for folder in folders:
        files += [os.path.join(folder, f) for f in natsorted(os.listdir(folder))
                  if f.endswith('.npy') and (prefix is None or f.startswith(prefix))]
    return np.vstack([np.load(f) for f in files])


def read_indices(path):
    if path.endswith('.json'):
        with open(path) as fh:
            return sorted(int(i['index'] if isinstance(i, dict) else i) for i in json.load(fh))
    with open(path) as fh:  # text file with one "feature_<i>" (or "<i>") per line
        return sorted(int(line.strip().replace('feature_', '')) for line in fh if line.strip())


def select(args):
    y = load_stack(args.targets, args.prefix)
    print('Stacked targets:', y.shape)
    # SVD of the data matrix; covariance singular values are s**2 / (n - 1)
    _, s, vh = np.linalg.svd(y - y.mean(axis=0), full_matrices=False)
    s_cov = s ** 2 / (y.shape[0] - 1)
    rank = int((s_cov > args.threshold).sum())
    selected = set()
    for i in range(rank):
        selected.update(np.argsort(np.abs(vh[i]))[::-1][:args.k].tolist())
    print(f'Rank (SV > {args.threshold:g}): {rank}; selected {len(selected)} of {y.shape[1]} features')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(sorted(int(i) for i in selected), fh)


def reduce(args):
    idx = read_indices(args.indices)
    os.makedirs(args.out, exist_ok=True)
    files = [f for f in natsorted(os.listdir(args.targets)) if f.endswith('.npy')]
    for f in files:
        np.save(os.path.join(args.out, f), np.load(os.path.join(args.targets, f))[idx].astype(np.float32))
    print(f'Reduced {len(files)} vectors to {len(idx)} features -> {args.out}')


def combine(args):
    union = set()
    for path in args.indices:
        union.update(read_indices(path))
    rng = np.random.default_rng(args.seed)
    pool = np.setdiff1d(np.arange(args.n_features), sorted(union))
    n_pad = max(0, args.size - len(union))
    union.update(rng.choice(pool, n_pad, replace=False).tolist())
    print(f'Union of {len(args.indices)} lists + {n_pad} random indices = {len(union)} features')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(sorted(int(i) for i in union), fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('select', help='SVD-based feature selection')
    p.add_argument('--targets', nargs='+', required=True, help='folder(s) with <key>.npy vectors, stacked')
    p.add_argument('--prefix', help='only use files whose name starts with this prefix')
    p.add_argument('--k', type=int, default=16, help='features kept per component')
    p.add_argument('--threshold', type=float, default=1e-6, help='minimum covariance singular value')
    p.add_argument('--out', required=True, help='output JSON file')
    p.set_defaults(func=select)

    p = sub.add_parser('reduce', help='keep a list of feature indices')
    p.add_argument('--targets', required=True, help='folder with <key>.npy vectors')
    p.add_argument('--indices', required=True, help='JSON list or text file with feature_<i> lines')
    p.add_argument('--out', required=True, help='output folder')
    p.set_defaults(func=reduce)

    p = sub.add_parser('combine', help='union of index lists, padded with random indices')
    p.add_argument('--indices', nargs='+', required=True, help='JSON index lists (e.g. one per box)')
    p.add_argument('--size', type=int, default=640, help='final number of features')
    p.add_argument('--n-features', type=int, default=15360, help='size of the original feature pool')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True, help='output JSON file')
    p.set_defaults(func=combine)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
