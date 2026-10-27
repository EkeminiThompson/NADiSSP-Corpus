"""
Core evaluation metrics used in Chapter 4.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
)


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
    max_samples: int = 2000,
) -> float:
    """
    Biased MMD with RBF kernel (Gretton et al.).
    Subsamples for speed when N is large.
    """
    rng = np.random.default_rng(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    def rbf(a, b, g):
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-g * sq)

    if gamma is None:
        Z = np.vstack([X[:500], Y[:500]])
        d = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        med = np.median(d[d > 0])
        gamma = 1.0 / (2 * med) if med > 0 else 1.0

    Kxx = rbf(X, X, gamma)
    Kyy = rbf(Y, Y, gamma)
    Kxy = rbf(X, Y, gamma)
    m, n = len(X), len(Y)
    mmd2 = Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)
    return float(np.sqrt(max(mmd2, 0.0)))


def compute_wasserstein(X: np.ndarray, Y: np.ndarray, max_samples: int = 1500) -> float:
    """
    Approximate 1-Wasserstein via average per-dimension 1-D Wasserstein
    (scipy). Fast proxy used in the dissertation tables.
    """
    from scipy.stats import wasserstein_distance

    rng = np.random.default_rng(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    dims = min(X.shape[1], Y.shape[1])
    dists = [wasserstein_distance(X[:, i], Y[:, i]) for i in range(dims)]
    return float(np.mean(dists))


def evaluate_predictions(
    y_true_rul: np.ndarray,
    y_pred_rul: np.ndarray,
    y_true_fail: np.ndarray,
    y_pred_fail_proba: np.ndarray,
) -> Dict[str, float]:
    """Standard PdM metrics reported in Chapter 4."""
    rmse = float(np.sqrt(mean_squared_error(y_true_rul, y_pred_rul)))
    results = {"rul_rmse": rmse}

    if y_true_fail.min() != y_true_fail.max():
        results["auprc"] = float(average_precision_score(y_true_fail, y_pred_fail_proba))
        pred_bin = (y_pred_fail_proba >= 0.5).astype(int)
        results["f1"] = float(f1_score(y_true_fail, pred_bin, zero_division=0))
    else:
        results["auprc"] = float("nan")
        results["f1"] = float("nan")
    return results
