"""
SHAP / permutation-importance attribution for the NADIP ensemble head.

Used to produce Table 4.8 and Figure 4.5 in the dissertation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def permutation_importance_rul(
    model,
    X: np.ndarray,
    y_rul: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Model-agnostic permutation importance on RUL RMSE.
    Works with any object that has a .predict(X) -> RUL method
    or a callable.
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[-1] if X.ndim == 2 else X.shape[2]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    def predict(x):
        if hasattr(model, "predict"):
            return model.predict(x)
        return model(x)

    if X.ndim == 3:
        X_flat = X.mean(axis=1)
    else:
        X_flat = X

    baseline = np.sqrt(np.mean((y_rul - predict(X_flat)) ** 2))
    importances = np.zeros(n_features)

    for j in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_flat.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            rmse = np.sqrt(np.mean((y_rul - predict(X_perm)) ** 2))
            scores.append(rmse - baseline)
        importances[j] = np.mean(scores)

    total = importances.sum() + 1e-12
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "pct_attribution": 100.0 * importances / total,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def run_shap_attribution(
    model,
    X: np.ndarray,
    y_rul: np.ndarray,
    feature_names: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Dict:
    """
    Produce the attribution table used in Chapter 4.
    Falls back to permutation importance (always available).
    Tries real SHAP TreeExplainer if an XGBoost model is supplied.
    """
    out_dir = out_dir or Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        if hasattr(model, "get_booster") or "XGB" in type(model).__name__:
            explainer = shap.TreeExplainer(model)
            if X.ndim == 3:
                X_flat = X.mean(axis=1)
            else:
                X_flat = X
            shap_values = explainer.shap_values(X_flat[:2000])
            mean_abs = np.abs(shap_values).mean(axis=0)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(mean_abs))]
            df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "pct_attribution": 100.0 * mean_abs / (mean_abs.sum() + 1e-12),
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
            method = "shap_tree"
        else:
            raise ValueError("Not a tree model")
    except Exception:
        df = permutation_importance_rul(model, X, y_rul, feature_names)
        method = "permutation"

    csv_path = out_dir / "shap_attribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"Attribution table ({method}) -> {csv_path}")
    return {"method": method, "table": df, "path": str(csv_path)}
