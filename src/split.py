"""Murcko scaffold splitting for BBBP (MoleculeNet-style, implemented by hand).

Each molecule's Bemis-Murcko scaffold is computed with RDKit. Scaffolds are
sorted by size (largest first), then whole scaffolds are assigned greedily to
train/valid/test in 80/10/10 proportions, so no scaffold appears in more than
one split. This makes the test set a genuine out-of-distribution chemical-space
challenge, unlike a random split.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def mol_from_smiles(smiles):
    """Parse SMILES; return None if RDKit cannot parse it at all."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    # retry without sanitization; still None => molecule is dropped upstream
    return Chem.MolFromSmiles(smiles, sanitize=False)


def scaffold_smiles(mol):
    """Bemis-Murcko scaffold SMILES; None if scaffold is empty/uncomputable."""
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None  # e.g. unsanitized mol without ring info
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaf)


def scaffold_split(df, smiles_col="smiles", fracs=(0.8, 0.1, 0.1), seed=42):
    """Return (train_idx, valid_idx, test_idx) with scaffold-disjoint splits.

    Scaffolds sorted by size descending; ties broken deterministically with
    the given seed. Whole scaffolds go to a single split.
    """
    rng = np.random.RandomState(seed)
    scaffolds = {}
    for idx, smi in df[smiles_col].items():
        mol = mol_from_smiles(smi)
        scaf = scaffold_smiles(mol)
        scaffolds.setdefault(scaf, []).append(idx)

    # order: largest scaffold first; ties broken deterministically by scaffold key
    order = sorted(scaffolds.keys(),
                   key=lambda s: (-len(scaffolds[s]), str(s)))
    n_total = len(df)
    train_cutoff = fracs[0] * n_total
    valid_cutoff = fracs[1] * n_total  # relative to valid alone

    train_idx, valid_idx, test_idx = [], [], []
    for scaf in order:
        ids = scaffolds[scaf]
        # MoleculeNet-style greedy: whole scaffold goes to the split it still fits in
        if len(train_idx) + len(ids) > train_cutoff:
            if len(valid_idx) + len(ids) > valid_cutoff:
                test_idx.extend(ids)
            else:
                valid_idx.extend(ids)
        else:
            train_idx.extend(ids)

    # fallbacks: ensure nothing is lost if rounding quirks empty a split
    used = set(train_idx) | set(valid_idx) | set(test_idx)
    leftover = [idx for idx in df.index if idx not in used]
    test_idx.extend(leftover)

    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def random_split(df, fracs=(0.8, 0.1, 0.1), seed=42):
    """Stratified-free random split with the same proportions and seed style."""
    rng = np.random.RandomState(seed)
    idx = np.array(df.index)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(fracs[0] * n)
    n_valid = int(fracs[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def split_diagnostics(df, train_idx, valid_idx, test_idx, smiles_col="smiles"):
    """Report scaffold counts and scaffold overlap between splits (must be 0)."""
    def scaf_set(idxs):
        s = set()
        for idx in idxs:
            mol = mol_from_smiles(df.loc[idx, smiles_col])
            s.add(scaffold_smiles(mol))
        return s

    tr, va, te = scaf_set(train_idx), scaf_set(valid_idx), scaf_set(test_idx)
    return {
        "n_unique_scaffolds_train": len(tr),
        "n_unique_scaffolds_valid": len(va),
        "n_unique_scaffolds_test": len(te),
        "overlap_train_valid": len(tr & va),
        "overlap_train_test": len(tr & te),
        "overlap_valid_test": len(va & te),
    }
