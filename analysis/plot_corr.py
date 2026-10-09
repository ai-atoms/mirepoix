"""Correlation plot between true and predicted u_640 (all frames, all boxes).

  python scripts/plot_corr.py --encoder convnext_in --label f
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--encoder', default='convnext_in')
ap.add_argument('--label', default='f')
ap.add_argument('--data', default='data/d567b')
ap.add_argument('--out')
args = ap.parse_args()

true_dir = os.path.join(args.data, 'u640')
pred_dir = os.path.join(args.data, 'results', args.encoder)
files = sorted(set(os.listdir(true_dir)) & set(os.listdir(pred_dir)))
true = np.concatenate([np.load(os.path.join(true_dir, f)).ravel() for f in files])
pred = np.concatenate([np.load(os.path.join(pred_dir, f)).ravel() for f in files])
print(f'{len(files)} frames, {true.size:,} values, RMSE {np.sqrt(np.mean((true - pred) ** 2)):.6f}')

plt.style.use(['science', 'nature'])
plt.rcParams.update({'font.size': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20})
plt.figure(figsize=(8, 6))
plt.hist2d(true, pred, bins=110, cmap='YlGnBu', norm='log', vmax=1e5)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title(f'({args.label})')
plt.xlabel('True Values', fontsize=20)
plt.ylabel('Predicted Values', fontsize=20)
plt.colorbar().set_label('log(Count)', fontsize=20)
plt.tight_layout()
out = args.out or os.path.join(args.data, 'results', f'corr_{args.encoder}.png')
plt.savefig(out, dpi=300)
print('saved', out)
