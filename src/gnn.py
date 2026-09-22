"""Minimal message-passing GNN in pure PyTorch (CPU), for the honesty check.

Question: does a learned graph representation beat ECFP/MACCS fingerprints on
2k BBBP molecules under a scaffold split? Prior art suggests no on data this
small — this module exists to test that claim, not to win.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

from src.split import mol_from_smiles

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other"]


def atom_features(atom):
    sym = atom.GetSymbol()
    feats = [1.0 if sym == t else 0.0 for t in ATOM_TYPES[:-1]]
    feats.append(0.0 if sym in ATOM_TYPES[:-1] else 1.0)
    feats += [atom.GetDegree() / 4.0,
              1.0 if atom.GetIsAromatic() else 0.0,
              atom.GetFormalCharge(),
              atom.GetTotalNumHs() / 4.0]
    return feats


def mol_to_graph(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        feats = [atom_features(a) for a in mol.GetAtoms()]
    except Exception:
        return None  # unsanitized mol without implicit-H info; drop it
    x = torch.tensor(feats, dtype=torch.float32)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    if not edges:  # single atom: self loop
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


class GraphDataset(Dataset):
    def __init__(self, smiles_list, y):
        self.items = [(mol_to_graph(s), yy) for s, yy in zip(smiles_list, y)]
        self.items = [(g, yy) for g, yy in self.items if g is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    xs, edge_indices, ys, offset = [], [], [], 0
    for (x, ei), y in batch:
        xs.append(x)
        edge_indices.append(ei + offset)
        ys.append(y)
        offset += x.shape[0]
    x = torch.cat(xs, 0)
    ei = torch.cat(edge_indices, 1)
    # node -> graph assignment
    counts = [b[0][0].shape[0] for b in batch]
    bidx = torch.cat([torch.full((c,), k, dtype=torch.long)
                      for k, c in enumerate(counts)])
    y = torch.tensor(ys, dtype=torch.float32)
    return x, ei, bidx, y


class MPNN(nn.Module):
    """3 message-passing layers, mean aggregation, global mean pool, MLP head."""

    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch_idx):
        h = torch.relu(self.embed(x))
        src, dst = edge_index
        for lin, norm in zip(self.layers, self.norms):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src])
            deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(
                0, dst, torch.ones(dst.shape[0], 1, device=h.device)).clamp_min(1)
            agg = agg / deg
            h = norm(h + self.dropout(torch.relu(lin(torch.cat([h, agg], 1)))))
        n_graphs = int(batch_idx.max().item()) + 1
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch_idx, h)
        counts = torch.zeros(n_graphs, 1, device=h.device).index_add_(
            0, batch_idx, torch.ones(batch_idx.shape[0], 1, device=h.device))
        pooled = pooled / counts.clamp_min(1)
        return self.head(pooled).squeeze(1)


def train_gnn(df, X_unused, y, tr, va, te, out_csv, seed=42, epochs=80,
              batch_size=64, lr=1e-3, patience=12):
    """Train on scaffold split; append gnn row to metrics csv. Returns test probs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    smiles = df["smiles"].tolist()
    tr_ds = GraphDataset([smiles[i] for i in tr], y[tr])
    va_ds = GraphDataset([smiles[i] for i in va], y[va])
    te_ds = GraphDataset([smiles[i] for i in te], y[te])
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=batch_size, collate_fn=collate)
    te_dl = DataLoader(te_ds, batch_size=batch_size, collate_fn=collate)

    in_dim = len(ATOM_TYPES) + 4
    model = MPNN(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def eval_auc(dl):
        from sklearn.metrics import roc_auc_score
        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, ei, b, yy in dl:
                ps.append(torch.sigmoid(model(x, ei, b)))
                ys.append(yy)
        p = torch.cat(ps).numpy()
        return roc_auc_score(torch.cat(ys).numpy(), p), p

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, ei, b, yy in tr_dl:
            opt.zero_grad()
            loss_fn(model(x, ei, b), yy).backward()
            opt.step()
        auc, _ = eval_auc(va_dl)
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.cpu().clone()
                                              for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    _, p_test = eval_auc(te_dl)
    y_test = np.concatenate([yy.numpy() for _, _, _, yy in te_dl])
    return p_test, y_test
