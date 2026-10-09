"""VAR forecast of u_16: load a shipped model (default VAR(1), 32 frames, box 70) or train a new one.

    python analysis/var_forecast.py --n_train 32                 # load shipped model
    python analysis/var_forecast.py --train --n_train 40 --lag 1 # fit on first 40 frames of --fit_box (needs statsmodels)

As a library:
    from var_forecast import load_model, fit_model, forecast
    m = load_model("data/npj_components/models/var/var_70_ntrain32_lag01.npz")
    preds = forecast(m, u16_frames, n_train=32)   # u16_frames: (T, 16); first n_train rows seed the recursion

For each box (50, 60, 70, 71, 72) the first n_train frames seed a recursive one-step forecast
(normalised with the box-70 training statistics). Outputs in data/d567b/results/var/:
  rmse_ntrain<N>.csv            per-variable forecast RMSE (frames > n_train) for every box
  var_errors_70_ntrain<N>.png   forecast error of each u_16 variable (box --box), with 95 % band
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

DPA_PER_FRAME = (2 * 200) / 686000
BOXES = ["50", "60", "70", "71", "72"]


def load_box(folder, box):
    fn = sorted((f for f in os.listdir(folder) if f.startswith(box + "_")),
                key=lambda f: int(f.split("_")[-1].split(".")[0]))
    return np.stack([np.load(f"{folder}/{f}").squeeze() for f in fn])


def load_model(path):
    return dict(np.load(path))


def fit_model(train, lag=1):
    """OLS VAR(lag) on normalised training frames (n_train, 16); returns a model dict."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    mean, std = train.mean(0), train.std(0)
    res = VAR((train - mean) / (std + 1e-8)).fit(maxlags=lag, method="ols")
    return dict(coefs=res.coefs, intercept=res.intercept, resid=res.resid * std,
                mean=mean, std=std, n_train=len(train), lag=lag)


def forecast(m, true, n_train):
    """Recursive one-step forecast; rows >= n_train of `true` are only used as placeholders (shape)."""
    lag, mean, std = int(m["lag"]), m["mean"], m["std"]
    preds = np.full_like(true, np.nan)
    preds[:n_train] = true[:n_train]
    for t in range(n_train, len(true)):
        z = (preds[t - lag:t] - mean) / (std + 1e-8)
        zn = m["intercept"] + sum(m["coefs"][k] @ z[-1 - k] for k in range(lag))
        preds[t] = zn * std + mean
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=32)
    ap.add_argument("--box", default="70")
    ap.add_argument("--data", default="data/d567b")
    ap.add_argument("--models", default="data/npj_components/models/var")
    ap.add_argument("--train", action="store_true", help="fit a new VAR instead of loading a shipped one")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--fit_box", default="70")
    ap.add_argument("--confidence", type=float, default=0.95)
    a = ap.parse_args()

    out = f"{a.data}/results/var"
    os.makedirs(out, exist_ok=True)
    if a.train:
        m = fit_model(load_box(f"{a.data}/u16", a.fit_box)[:a.n_train], a.lag)
        np.savez(f"{out}/var_{a.fit_box}_ntrain{a.n_train:02d}_lag{a.lag:02d}.npz", **m)
    else:
        m = load_model(f"{a.models}/var_70_ntrain{a.n_train:02d}_lag01.npz")

    rmse, store = {}, {}
    for box in BOXES:
        true = load_box(f"{a.data}/u16", box)
        if len(true) <= a.n_train:
            continue
        preds = forecast(m, true, a.n_train)
        err = true[a.n_train:] - preds[a.n_train:]
        rmse[box] = np.sqrt((err ** 2).mean(0))
        store[box] = (true, preds)
    df = pd.DataFrame(rmse, index=[f"u16_{i:02d}" for i in range(len(next(iter(rmse.values()))))])
    for box, (_, pr) in store.items():
        np.save(f"{out}/pred_{box}_ntrain{a.n_train}.npy", pr)
    df.to_csv(f"{out}/rmse_ntrain{a.n_train}.csv", float_format="%.6f")
    print(df.to_string(float_format="{:.4f}".format))

    # 95 % band: aleatoric (training residuals) combined with running epistemic std of forecast errors
    true, preds = store[a.box]
    n = a.n_train
    al = np.percentile(np.abs(m["resid"]), 100 * a.confidence, axis=0)
    err = true - preds
    ep = np.zeros_like(true)
    for t in range(n + 1, len(true)):
        ep[t] = err[n:t].std(0)
    band = 1.96 * np.sqrt(al ** 2 + ep[n:] ** 2)
    dpa = np.arange(len(true)) * DPA_PER_FRAME

    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.fill_between(dpa[n:], -band[:, i], band[:, i], color="tab:blue", alpha=0.25, lw=0)
        ax.plot(dpa[n:], err[n:, i], color="k", lw=1)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"u16_{i:02d}  RMSE {rmse[a.box][i]:.4f}", fontsize=9)
        if i >= 12:
            ax.set_xlabel("Dose (dpa)")
    fig.suptitle(f"VAR(1), n_train={n}: forecast error (true - predicted), box {a.box}")
    fig.tight_layout()
    fig.savefig(f"{out}/var_errors_{a.box}_ntrain{n}.png", dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
