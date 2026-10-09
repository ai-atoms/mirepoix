"""Singular values of the covariance matrix of a targets_N feature set."""
import argparse
import os

import numpy as np
from natsort import natsorted
from scipy.linalg import svd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='data/fpa70a/features/rs_169006142')
    ap.add_argument('--target', default='targets_15360')
    args = ap.parse_args()

    target_path = os.path.join(args.base, args.target)
    files = natsorted(f for f in os.listdir(target_path) if f.endswith('.npy'))
    y = np.vstack([np.load(os.path.join(target_path, f)) for f in files])
    print('Collected targets shape:', y.shape)

    cov = np.cov(y - y.mean(axis=0), rowvar=False)
    s = svd(cov, compute_uv=False)

    out_dir = os.path.join(args.base, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{args.target}.npy'), s)
    print('Saved singular values to', out_dir)


if __name__ == '__main__':
    main()
