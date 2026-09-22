"""Run the minimal MPNN on the same scaffold split as the fingerprint models.

Usage: python -m src.gnn_run --data /path/to/BBBP.csv --out results/
Appends a 'gnn' row (scaffold split) to results/metrics_gnn.csv.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.split import scaffold_split, mol_from_smiles
from src.gnn import train_gnn
from src.run import evaluate


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
    df = df[df["smiles"].map(lambda s: mol_from_smiles(s) is not None)].reset_index(drop=True)
    df["p_np"] = df["p_np"].astype(int)
    y = df["p_np"].to_numpy()
    tr, va, te = scaffold_split(df, seed=args.seed)
    print(f"splits: {len(tr)}/{len(va)}/{len(te)}", flush=True)

    p_test, y_test = train_gnn(df, None, y, tr, va, te,
                               os.path.join(args.out, "metrics_gnn.csv"),
                               seed=args.seed)
    row = {"split": "scaffold", "model": "mpnn", **evaluate(y_test, p_test)}
    pd.DataFrame([row]).to_csv(os.path.join(args.out, "metrics_gnn.csv"), index=False)
    print(pd.DataFrame([row]).round(4).to_string(index=False))
    print(f"GNN runtime: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
