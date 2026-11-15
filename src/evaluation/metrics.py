"""Prognostic metrics (Table 3.19): AUPRC, RUL RMSE, F1, evaluated per split / domain / dataset.

The F1 operating threshold is chosen on the VALIDATION split (never on test), then frozen.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from ..data.schema import DATASETS, DATASET_ID


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")                                   # undefined without both classes -- never reported as 0
    return float(average_precision_score(y, p))


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or y.min() == y.max():
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(f1_score(y, (p >= thr).astype(int), zero_division=0))


def rul_rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(y)[m] - np.asarray(pred)[m]) ** 2)))


def score_rows(out: Dict[str, np.ndarray], bundle, rows: np.ndarray, thr: float, key_rul: str = "rul") -> Dict:
    y = bundle.fail[rows].numpy()
    has = bundle.has_rul[rows].numpy()
    return {
        "n": int(len(rows)),
        "fail_prevalence": float(y.mean()) if len(y) else float("nan"),
        "auprc": auprc(y, out["fail_prob"]),
        "f1": f1_at(y, out["fail_prob"], thr),
        "rul_rmse": rul_rmse(bundle.rul[rows].numpy(), out[key_rul], has),
        "n_with_rul": int(has.sum()),
    }


def evaluate(trained, bundle, split: str = "test", threshold: Optional[float] = None) -> Dict:
    """Overall, per-domain and per-dataset scores on `split`. Threshold: best-F1 on validation."""
    rows_val = bundle.idx("val")
    if threshold is None:
        ov = trained.predict(bundle.X[rows_val], bundle.dataset[rows_val]) if len(rows_val) else None
        threshold = best_f1_threshold(bundle.fail[rows_val].numpy(), ov["fail_prob"]) if ov is not None else 0.5
    rows = bundle.idx(split)
    out = trained.predict(bundle.X[rows], bundle.dataset[rows])
    res: Dict = {"split": split, "threshold": threshold, "overall": score_rows(out, bundle, rows, threshold),
                 "by_domain": {}, "by_dataset": {}}
    for dom, nm in ((0, "source"), (1, "target")):
        m = bundle.domain[rows] == dom
        if m.any():
            sub = {k: v[m] for k, v in out.items() if hasattr(v, "__len__") and len(v) == len(rows)}
            res["by_domain"][nm] = score_rows(sub, bundle, rows[m], threshold)
    for d in DATASETS:
        m = bundle.dataset[rows] == DATASET_ID[d]
        if m.any():
            sub = {k: v[m] for k, v in out.items() if hasattr(v, "__len__") and len(v) == len(rows)}
            res["by_dataset"][d] = score_rows(sub, bundle, rows[m], threshold)
    if "transformer_rul" in out:                              # contribution of the XGBoost ensemble stage
        res["transformer_only"] = score_rows({"rul": out["transformer_rul"], "fail_prob": out["transformer_fail_prob"]},
                                             bundle, rows, threshold)
    return res
