"""Interpretable window features for the XGBoost baseline / ensemble and for SHAP (Table 4.8).

Per unified channel: mean, std, trend (last-first), min, max. Plus one "telemetry gap" proxy: the
number of steps at which every *varying* temperature/pressure channel of the window's own dataset
is unchanged for two consecutive steps -- what forward-filled dropout looks like from the receiver
side, so it can be computed on deployed data without knowing anything about the augmentation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..data.schema import DATASETS, OFFSETS, CHANNELS, N_CHANNELS, role_indices, unified_names

STATS = ["mean", "std", "trend", "min", "max"]


def feature_names() -> List[str]:
    return [f"{n}:{s}" for n in unified_names() for s in STATS] + ["telemetry_gap_proxy"]


def _gap_proxy(X: np.ndarray, dataset_ids: np.ndarray) -> np.ndarray:
    out = np.zeros(len(X), dtype=np.float32)
    for di, name in enumerate(DATASETS):
        rows = np.flatnonzero(dataset_ids == di)
        if len(rows) == 0:
            continue
        cols = OFFSETS[name] + role_indices(name, ["temperature", "pressure"])
        if len(cols) == 0:
            continue
        sub = X[rows][:, :, cols]                                         # (n, T, k)
        same = np.diff(sub, axis=1) == 0                                  # (n, T-1, k)
        varying = sub.std(axis=1) > 1e-6                                  # (n, k)
        all_flat = np.where(varying[:, None, :], same, True).all(axis=2) & varying.any(axis=1)[:, None]
        run2 = all_flat[:, 1:] & all_flat[:, :-1]
        out[rows] = run2.sum(axis=1)
    return out


def window_features(X: np.ndarray, dataset_ids: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    X = np.asarray(X, dtype=np.float32)
    mean, std = X.mean(1), X.std(1)
    trend = X[:, -1, :] - X[:, 0, :]
    mn, mx = X.min(1), X.max(1)
    stacked = np.stack([mean, std, trend, mn, mx], axis=2).reshape(len(X), N_CHANNELS * len(STATS))
    feats = np.concatenate([stacked, _gap_proxy(X, dataset_ids)[:, None]], axis=1)
    return feats.astype(np.float32), feature_names()
