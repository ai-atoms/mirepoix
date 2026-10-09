# Generating the distilled DSV vectors

`extract/distill.py` turns the 15360 random-logic features (`targets_15360`) into compact vectors.
The feature lists behind the paper tables are in `data/npj_components/`:

| File | Content |
|---|---|
| `u640.json` | the 640 features of u_640: index in the 15360 set + logic condition |
| `u64.json` | the 64 features of u_64, with their amplitude (max - min over the trajectory, initial state excluded) |
| `u16.json` | the 16 features of u_16, with amplitude and order |
| `per_box/selected_f{50,60,70}.json` | per-box SVD selections (264 for fpa70) |

## Reproducing the published vectors

```bash
T=data/fpa70a/features/rs_169006142/targets_15360
C=data/npj_components
for n in 640 64 16; do
  python extract/distill.py reduce --targets $T --indices $C/u$n.json --out data/fpa70a/distilled/u$n
done
```

All indices refer to the original 15360 features. Results are identical to the vectors used in the paper.

## Re-deriving the lists

```bash
# per-box selection: SVD of the feature covariance, 16 features per component with SV > 1e-6
python extract/distill.py select --targets $T --k 16 --threshold 1e-6 --out selected_f70.json

# union of the three boxes, padded with random indices up to 640
python extract/distill.py combine --indices $C/per_box/selected_f*.json --size 640 --out combined.json

# u_640 -> u_64: same rule, 4 features per component
python extract/distill.py select --targets data/fpa70a/distilled/u640 --k 4 --threshold 1e-6 --out sel64.json
```

The u_16 set is not an automatic step: it was taken from the 64 features by amplitude (large max-min range,
ignoring the initial state), see `amplitude` in `u64.json`, with a final manual check.

Running these on the supplied data alone selects fewer features than the published lists: the paper selections
combine three boxes (fpa50, fpa60, fpa70), the 78 padding indices of u_640 are random, and the 640 to 64 selection
stacks all three boxes (rank 26), while a single box has a lower rank (20). Use the files in `data/npj_components/`
to reproduce the published vectors exactly.

The trained image-to-u_640 regressors are in `data/npj_components/models/` (see its README).

The u_640 / u_16 to physical-property regressors (`physprop_u640/`, `physprop_u16/`) are in the same folder;
run `python analysis/physprop.py --input u640` (or `u16`) to evaluate them. The shipped u_16 model is slightly
improved relative to the published one.
