"""Molecular featurization: ECFP (Morgan) + MACCS keys via RDKit."""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys


def featurize(smiles_list, ecfp_bits=1024, ecfp_radius=2, use_maccs=True):
    """Return ndarray of shape (n, ecfp_bits + 167 if use_maccs else ecfp_bits)."""
    X_ecfp = np.zeros((len(smiles_list), ecfp_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, ecfp_radius, nBits=ecfp_bits)
        arr = np.zeros((ecfp_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X_ecfp[i] = arr
    if not use_maccs:
        return X_ecfp
    X_maccs = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X_maccs[i] = arr
    return np.hstack([X_ecfp, X_maccs])
