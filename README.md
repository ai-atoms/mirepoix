# Mirepoix

This repository demonstrates the distortion-state-vector (DSV) framework proposed in:

> **Distortion states as robust embeddings for quantifying and forecasting atomistic damage from electron microscopy**
> npj Computational Materials (2026). DOI: [10.1038/s41524-026-02340-2](https://doi.org/10.1038/s41524-026-02340-2)

The demonstration uses one of the bcc-Fe boxes described in the paper (686,000 atoms) and follows a semi-complete processing chain, from descriptors to published results.

## Processing chain

```
atomistic side                          image side
--------------                          ----------
distortion-state distributions (7, N)   *.cfg  --[generate/]-->  synthetic STEM images (abTEM)
                                                --[extract/]-->   encoded vectors (ConvNeXt)
                       \                          /
                        +---- analysis and figures ----> results of the paper
```

### Atomistic side
The starting point is the set of distortion-state distributions, with shape `(7, N)`.
The raw bSO4 vectors `(55, N)` and the per-atom distortion-score calculation are **omitted** because the raw files are too large. The procedure is described exhaustively in the paper and is straightforward to reproduce.

### Image side
- Synthetic STEM images, generated with [abTEM](https://abtem.readthedocs.io), are provided. The scripts that generate such images from `*.cfg` files are in `generate/`.
- Image embeddings encoded with ConvNeXt are provided. The scripts that encode images with other models are in `extract/`.

## Repository layout

| Folder | Content |
|---|---|
| `generate/` | Generation of synthetic STEM images from `*.cfg` files (abTEM) |
| `extract/` | Extraction of image embeddings and of target data |
| `analysis/` | Image encodings to u_640 (`predict_heads.py`, `plot_corr.py`) and u_640 / u_16 to physical properties (`physprop.py`; shipped u_16 model slightly improved relative to the published one) |
| `analysis/var_forecast.py` | VAR(1) forecast of u_16 (per-variable RMSE table and error plots) |
| `md/` | LAMMPS inputs used to generate the damaged boxes |

## Data

The input data are distributed separately through Zenodo: _DOI to be added_. Download and extract the archive in a `data/` folder at the root of the repository. Data and results (`data/`, `results/`, `*.csv`, `*.png`) are not tracked by git.

## Environment

```bash
pip install -r requirements.txt
```

## Usage

Download the Zenodo archive and extract it as `data/` (folders `fpa70a/`, `d567b/`, `npj_components/`). Run everything from the repository root. See `details.md` for the distillation step.

```bash
# 1. Atomistic side: DSV -> targets -> distilled vectors u_640 / u_64 / u_16 (fpa70a)
python extract/distill.py --help

# 2. Image side: encode STEM images with ConvNeXt (fpa70a)
python extract/extract_cnext.py

# 3. Paper dataset (d567b): image encodings -> u_640 (Fig. 5f)
python analysis/predict_heads.py --encoder convnext_in
python analysis/plot_corr.py --encoder convnext_in

# 4. u_640 / u_16 -> physical properties (test MSE per variable)
python analysis/physprop.py --input u640
python analysis/physprop.py --input u16

# 5. VAR(1) forecast of u_16: load the shipped model (default, 32 frames) ...
python analysis/var_forecast.py --n_train 32
# ... or train a new one on the first N frames of box 70
python analysis/var_forecast.py --train --n_train 40 --lag 1
```

Results are written to `data/d567b/results/`. Trained models are in `data/npj_components/models/`.

## How to cite

If you use this code or data, please cite the paper above.

## Authors

[Camilo A. F. Salvador](https://github.com/camilofs), Clovis Lapointe, Mihai-Cosmin Marinica.

## License

MIT, CEA Paris-Saclay 2025 (C). See `LICENSE`.

## Note on documentation

The documentation files (`README.md`, `details.md`, and the models README) were drafted with the help of an AI coding assistant (GitHub Copilot) and reviewed by the authors.

*This work has been carried out within the Cross-disciplinary initiative for digital science of the French Alternative Energies and Atomic Energy Commission (CEA).*
