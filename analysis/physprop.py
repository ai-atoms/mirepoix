"""u -> physical properties (8 targets): evaluate shipped regressors, or retrain.

    python analysis/physprop.py --input u640            # evaluate shipped models
    python analysis/physprop.py --input u16 --train     # retrain (written to --out)

One MLP (in->16->8->1) per target, targets MinMax-scaled to [0, 1]. The reported
metric is the test MSE on scaled targets (80/20 split, random_state=42 over the
sorted file names), as in the paper table. Predictions in physical units are
saved to data/d567b/results/physprop_<input>.npy (all files, sorted order).
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split

NAMES = ["C15 size, median", "C15 size, variance", "C15 size, max", "Vacancies per atom",
         "Disloc. dens. <111>", "<111> loops (number)", "Disloc. dens. <100>", "<100> loops (number)"]


def mlp(n):
    return nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))


def train(Xtr, ytr, seed, epochs):
    torch.manual_seed(seed)
    m = mlp(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x, y = torch.tensor(Xtr), torch.tensor(ytr)[:, None]
    for _ in range(epochs):
        p = torch.randperm(len(x))
        for b in range(0, len(x), 64):
            j = p[b:b + 64]
            loss = ((m(x[j]) - y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="u640", choices=["u16", "u64", "u640"])
    ap.add_argument("--data", default="data/d567b")
    ap.add_argument("--models", default="data/npj_components/models")
    ap.add_argument("--train", action="store_true", help="retrain instead of loading shipped models")
    ap.add_argument("--out", default="data/d567b/results/physprop_retrained")
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fn = sorted(os.listdir(f"{a.data}/{a.input}"))
    X = np.stack([np.load(f"{a.data}/{a.input}/{f}").squeeze() for f in fn]).astype(np.float32)
    Y = np.stack([np.load(f"{a.data}/ptargets/{f}").squeeze() for f in fn]).astype(np.float64)
    itr, ite = train_test_split(np.arange(len(fn)), test_size=0.2, random_state=42)

    mdir = f"{a.out}/physprop_{a.input}" if a.train else f"{a.models}/physprop_{a.input}"
    os.makedirs(mdir, exist_ok=True)
    pred, mse = np.zeros_like(Y), []
    for i in range(8):
        if a.train:
            lo, hi = Y[:, i].min(), Y[:, i].max()
            m = train(X[itr], ((Y[itr, i] - lo) / (hi - lo)).astype(np.float32), a.seed * 100 + i, a.epochs)
            torch.save(m.state_dict(), f"{mdir}/regressor_{i}.pth")
            json.dump({"min": lo, "max": hi}, open(f"{mdir}/scaler_{i}.json", "w"))
        else:
            m = mlp(X.shape[1])
            m.load_state_dict({k.removeprefix("net."): v for k, v in
                               torch.load(f"{mdir}/regressor_{i}.pth", map_location="cpu").items()})
            s = json.load(open(f"{mdir}/scaler_{i}.json")); lo, hi = s["min"], s["max"]
        with torch.no_grad():
            p = m(torch.tensor(X)).numpy()[:, 0]
        pred[:, i] = p * (hi - lo) + lo
        mse.append(float(np.mean((p[ite] - (Y[ite, i] - lo) / (hi - lo)) ** 2)))
        print(f"{NAMES[i]:22s} test MSE {mse[-1]:.4f}")
    os.makedirs(f"{a.data}/results", exist_ok=True)
    np.save(f"{a.data}/results/physprop_{a.input}.npy", pred)
    json.dump(dict(zip(NAMES, mse)), open(f"{mdir}/test_mse.json", "w"), indent=1)


if __name__ == "__main__":
    main()
