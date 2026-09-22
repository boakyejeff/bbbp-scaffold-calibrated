# Build notes

## Dataset
- URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
- Size: **148,743 bytes** (~145 KB), 2050 data rows after header.
- Columns: `num, name, p_np, smiles`. `p_np` is the binary label; **76.4%
  positive** (majority-positive imbalance).
- Downloaded 2026-09-22; open, no credentialing. Cite Wu et al. 2018
  (MoleculeNet, *Chemical Science*).
- All 2050 rows were usable. One SMILES (a quaternary ammonium) fails RDKit's
  default sanitization ("Explicit valence for atom # 1 N, 4"); it was kept via
  non-sanitized parsing, and excluded from the GNN graph featurization where
  implicit-H info was unavailable (1 graph dropped from training).

## RDKit install notes
- System pip is PEP-668 externally-managed → created a project venv.
- `pip install rdkit scikit-learn pandas numpy matplotlib` in the venv just
  worked (rdkit 2026.03.6). Two API gotchas in this version:
  - `rdkit.Chem.Scaffolds.MurckoScaffold` has **no** `MurckoToSmiles` anymore —
    use `Chem.MolToSmiles(GetScaffoldForMol(mol))`.
  - `MurckoScaffold.GetScaffoldForMol` raises `RuntimeError: RingInfo not
    initialized` on unsanitized mols → wrapped in try/except (returns None).
- torch CPU wheel (~190 MB) via the PyTorch CPU index; needed
  `TMPDIR` pointed at the home disk because `/tmp` (512 MB tmpfs) was nearly
  full. Install took ~12 min on this box — the GNN stretch is only "fast" once
  torch is already installed.

## Honest numbers (seed 42; full tables in README + results/metrics.csv)

- Scaffold split: 1640/205/205 compounds; unique scaffolds 693/205/205;
  overlap between any pair of splits = **0** (verified in `split_diagnostics`).
- Fingerprints (ECFP-1024 + MACCS) under scaffold split: RF AUROC 0.843,
  LogReg 0.819, MLP 0.762. ECE 0.065–0.088 — usable but not great.
- Random-split inflation: **+0.092 to +0.159 AUROC**; balanced accuracy falls
  0.85–0.87 → 0.55–0.68 when moving to scaffold split. AUPRC barely moves
  (high base rate). This is the central result.
- **Null result (kept, not buried):** minimal MPNN in pure torch CPU —
  3 layers AUROC 0.634, 4 layers/128-dim AUROC 0.685, ECE ~0.09–0.10. Both
  well below fingerprint RF (0.843). On 2k molecules with hand-crafted
  fingerprints this strong, learned graph representations lose — exactly the
  expected honest finding. Two configs tried (~2 min + ~6 min); no further
  architecture search, by design.
- Runtime: main benchmark ~35 s CPU; GNN ~2–6 min CPU.

## What I'd do next (not done)
- Multiple seeds + scaffold-split variants (e.g. stratified by label) for
  confidence intervals on the gap.
- Temperature scaling / isotonic recalibration on the valid split, then
  re-measure ECE — the natural follow-up this repo sets up.
- A serious GNN baseline (bond features, attention pooling, pretrained
  embeddings) to test whether the null survives.
