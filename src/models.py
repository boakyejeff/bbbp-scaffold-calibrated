"""Model zoo: sklearn models trained on fingerprints, tuned lightly on valid."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score


def build_models(seed=42):
    """Name -> unfit estimator. All CPU-fast on 2k molecules."""
    return {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                     random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None,
                                     min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3,
                             max_iter=500, early_stopping=True, n_iter_no_change=20,
                             random_state=seed),
    }


def tune_on_valid(models, X_train, y_train, X_valid, y_valid):
    """Light grid search on valid AUROC; refit the winner on train+valid."""
    grids = {
        "logreg": {"C": [0.1, 1.0, 10.0]},
        "rf": {"max_depth": [None, 12], "min_samples_leaf": [1, 4]},
        "mlp": {"alpha": [1e-4, 1e-3]},
    }
    best = {}
    for name, base in models.items():
        best_name_score, best_est = -1.0, base
        for k, vals in grids[name].items():
            for v in vals:
                est = base.__class__(**{**base.get_params(), k: v})
                est.fit(X_train, y_train)
                score = roc_auc_score(y_valid, est.predict_proba(X_valid)[:, 1])
                if score > best_name_score:
                    best_name_score, best_est = score, est
        # refit winner on train+valid for final test evaluation
        best[name] = best_est.fit(
            __import__("numpy").concatenate([X_train, X_valid]),
            __import__("numpy").concatenate([y_train, y_valid]),
        )
    return best
