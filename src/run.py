"""End-to-end benchmark: BBBP scaffold-split vs random-split, with calibration.

Usage:
    python -m src.run --data /path/to/BBBP.csv --out results/
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score)

from src.split import scaffold_split, random_split, split_diagnostics
from src.features import featurize
from src.models import build_models, tune_on_valid
from src.calibration import (expected_calibration_error, brier_score,
                             calibration_curve_data, plot_reliability_diagram)


def evaluate(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
    }


def run_split(df, X, y, split_fn, split_name, outdir, seed=42):
    tr, va, te = split_fn(df, seed=seed)
    diag = split_diagnostics(df, tr, va, te) if split_name == "scaffold" else {}
    diag.update({
        "split": split_name, "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
        "pos_rate_train": float(y[tr].mean()), "pos_rate_valid": float(y[va].mean()),
        "pos_rate_test": float(y[te].mean()),
    })

    models = build_models(seed=seed)
    tuned = tune_on_valid(models, X[tr], y[tr], X[va], y[va])

    rows, probs = [], {}
    for name, est in tuned.items():
        p = est.predict_proba(X[te])[:, 1]
        probs[name] = (y[te], p)
        row = {"split": split_name, "model": name, **evaluate(y[te], p)}
        rows.append(row)
        # plot-ready per-bin calibration data
        d = calibration_curve_data(np.asarray(y[te]), np.asarray(p))
        centers = (d["bin_edges"][:-1] + d["bin_edges"][1:]) / 2
        pd.DataFrame({
            "bin_center": centers, "mean_pred": d["prob_pred"],
            "frac_pos": d["prob_true"], "count": d["counts"],
        }).to_csv(os.path.join(outdir, f"reliability_{split_name}_{name}.csv"),
                   index=False)

    plot_reliability_diagram(
        probs, os.path.join(outdir, f"reliability_{split_name}.png"),
        title=f"Reliability diagram — {split_name} split (test)")
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(args.data).reset_index(drop=True)
    df = df.dropna(subset=["smiles", "p_np"])
    from src.split import mol_from_smiles
    n0 = len(df)
    df = df[df["smiles"].map(lambda s: mol_from_smiles(s) is not None)].reset_index(drop=True)
    df["p_np"] = df["p_np"].astype(int)
    print(f"Loaded {n0} rows; dropped {n0 - len(df)} unparsable; "
          f"using {len(df)} compounds, positive rate = {df['p_np'].mean():.3f}", flush=True)

    t1 = time.time()
    X = featurize(df["smiles"].tolist())
    y = df["p_np"].to_numpy()
    print(f"Featurized -> {X.shape} in {time.time()-t1:.1f}s", flush=True)

    all_rows, diags = [], []
    for name, fn in (("scaffold", scaffold_split), ("random", random_split)):
        t = time.time()
        rows, diag = run_split(df, X, y, fn, name, args.out, seed=args.seed)
        all_rows.extend(rows)
        diags.append(diag)
        print(f"{name}: splits={diag['n_train']}/{diag['n_valid']}/{diag['n_test']} "
              f"({time.time()-t:.1f}s)", flush=True)
        if name == "scaffold":
            print(f"scaffold diag: unique train/valid/test = "
                  f"{diag['n_unique_scaffolds_train']}/"
                  f"{diag['n_unique_scaffolds_valid']}/"
                  f"{diag['n_unique_scaffolds_test']}; "
                  f"overlap tv/tt/vt = {diag['overlap_train_valid']}/"
                  f"{diag['overlap_train_test']}/{diag['overlap_valid_test']}", flush=True)

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    pd.DataFrame(diags).to_csv(os.path.join(args.out, "splits.csv"), index=False)

    # random-vs-scaffold gap table
    m = pd.DataFrame(all_rows).pivot(index="model", columns="split")
    gap = (m.xs("random", level="split", axis=1) - m.xs("scaffold", level="split", axis=1))
    gap.to_csv(os.path.join(args.out, "random_minus_scaffold_gap.csv"))
    print("\n=== random minus scaffold gap ===")
    print(gap.round(4).to_string())
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
