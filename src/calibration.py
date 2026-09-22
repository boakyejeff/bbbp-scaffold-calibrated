"""Calibration utilities: ECE, Brier score, reliability-diagram data."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def calibration_curve_data(y_true, y_prob, n_bins=15):
    """Return dict with per-bin edges, mean predicted prob, and empirical freq."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    prob_true, prob_pred, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        counts.append(mask.sum())
        prob_pred.append(y_prob[mask].mean() if mask.any() else np.nan)
        prob_true.append(y_true[mask].mean() if mask.any() else np.nan)
    return {
        "bin_edges": bins,
        "prob_pred": np.array(prob_pred),
        "prob_true": np.array(prob_true),
        "counts": np.array(counts),
    }


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE: mean |acc - conf| weighted by bin mass."""
    d = calibration_curve_data(y_true, y_prob, n_bins)
    n = len(y_true)
    ece = 0.0
    for acc, conf, c in zip(d["prob_true"], d["prob_pred"], d["counts"]):
        if c > 0:
            ece += (c / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true, y_prob):
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def plot_reliability_diagram(results, path, n_bins=15, title="Reliability diagram"):
    """Plot one reliability curve per model. results: {name: (y_true, y_prob)}."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y_true, y_prob) in results.items():
        d = calibration_curve_data(np.asarray(y_true), np.asarray(y_prob), n_bins)
        ok = ~np.isnan(d["prob_true"])
        ax1.plot(d["prob_pred"][ok], d["prob_true"][ok], "o-", label=name, ms=4)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    for name, (y_true, y_prob) in results.items():
        d = calibration_curve_data(np.asarray(y_true), np.asarray(y_prob), n_bins)
        centers = (d["bin_edges"][:-1] + d["bin_edges"][1:]) / 2
        ax2.bar(centers, d["counts"], width=1 / n_bins * 0.9, alpha=0.4, label=name)
    ax2.set_xlabel("Predicted probability bin")
    ax2.set_ylabel("Count")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
