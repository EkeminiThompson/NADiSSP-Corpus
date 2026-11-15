"""Feature attribution for the XGBoost ensemble (§4.4.3, Table 4.8).

Exact TreeSHAP values from XGBoost's own implementation (`pred_contribs=True`), so no extra dependency
is needed. The `shap` package's TreeExplainer computes the same quantity; `use_shap_package=True` uses
it when installed. The DeepExplainer path of Table 3.15b (SHAP on the transformer itself) is not implemented.

Inputs of the ensemble are [embedding z (d dims) | interpretable window features | transformer RUL]. Embedding
dimensions have no physical meaning, so they are aggregated into one "embedding (d dims)" row; the
interpretable features carry sensor names and categories.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.schema import CHANNELS, DATASETS
from .features import feature_names, window_features

ROLE = {f"{d}:{c}": r for d in DATASETS for c, r in CHANNELS[d]}


def _category(name: str) -> str:
    if name == "telemetry_gap_proxy":
        return "Nigerian-specific (gap proxy)"
    if name.startswith("embedding"):
        return "embedding"
    if name == "transformer_rul":
        return "transformer RUL"
    ch = name.rsplit(":", 1)[0]
    return {"pressure": "Pressure", "temperature": "Temperature", "vibration": "Vibration",
            "electrical": "Electrical", "flow": "Flow", "other": "Other"}.get(ROLE.get(ch, "other"), "Other")


def ensemble_shap(trained, bundle, rows: np.ndarray, target: str = "clf", max_rows: int = 2000, top_k: int = 12,
                  use_shap_package: bool = False) -> Dict:
    xg = getattr(trained, "xgb", None)
    model = None if xg is None else xg.get(target)
    if model is None:
        return {"note": f"no XGBoost '{target}' model in this variant (need use_xgb and both classes present)"}
    rows = rows[:max_rows]
    X = bundle.X[torch.from_numpy(rows)]
    out = trained.predict(X, bundle.dataset[rows])
    feats, fnames = window_features(X.numpy(), bundle.dataset[rows])
    Z = np.concatenate([out["z"], feats, out["transformer_rul"][:, None]], axis=1)
    names = [f"embedding_{i}" for i in range(out["z"].shape[1])] + fnames + ["transformer_rul"]
    if use_shap_package:
        import shap
        sv = shap.TreeExplainer(model).shap_values(Z)
    else:
        import xgboost as xgb
        sv = model.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]    # last column = bias
    mean_abs = np.abs(sv).mean(0)
    emb = slice(0, out["z"].shape[1])
    groups: Dict[str, Dict] = {}
    groups[f"embedding ({out['z'].shape[1]} dims)"] = {"mean_abs_shap": float(mean_abs[emb].sum()), "category": "embedding",
                                                       "direction": None}
    for j in range(out["z"].shape[1], len(names)):
        col = Z[:, j]
        corr = float(np.corrcoef(col, sv[:, j])[0, 1]) if col.std() > 0 and sv[:, j].std() > 0 else 0.0
        groups[names[j]] = {"mean_abs_shap": float(mean_abs[j]), "category": _category(names[j]),
                            "direction": "up" if corr > 0.05 else "down" if corr < -0.05 else "mixed"}
    total = sum(g["mean_abs_shap"] for g in groups.values()) or 1.0
    table = sorted(({"feature": k, **v, "pct_attribution": 100 * v["mean_abs_shap"] / total} for k, v in groups.items()),
                   key=lambda r: -r["mean_abs_shap"])
    for i, r in enumerate(table):
        r["rank"] = i + 1
    cats: Dict[str, float] = {}
    for r in table:
        cats[r["category"]] = cats.get(r["category"], 0.0) + r["pct_attribution"]
    return {"target": target, "n_rows": int(len(rows)), "method": "shap_package_tree" if use_shap_package else "xgboost_treeshap",
            "top": table[:top_k], "by_category_pct": cats}
