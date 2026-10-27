"""
Unified PdM dataset construction for NADIP.

Combines source benchmarks + real 3W facility data + Nigerian-augmented variants.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .augmentation import NigerianAugmentor
from .loaders import load_cmapss, load_ai4i, load_espset, load_3w, PROCESSED_DIR


class PdMDataset(Dataset):
    """Simple windowed multivariate time-series dataset."""

    def __init__(self, X, y_rul, y_fail, domain, asset, window=50):
        self.X = X.astype(np.float32)
        self.y_rul = y_rul.astype(np.float32)
        self.y_fail = y_fail.astype(np.float32)
        self.domain = domain.astype(np.int64)
        self.asset = asset
        self.window = window

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "rul": self.y_rul[idx],
            "fail": self.y_fail[idx],
            "domain": self.domain[idx],
            "asset": self.asset[idx],
        }


def _window_series(df, feature_cols, window=50, stride=10, rul_col="RUL", fail_col=None):
    Xs, ruls, fails = [], [], []
    unit_col = "unit" if "unit" in df.columns else None
    groups = df.groupby(unit_col) if unit_col else [("all", df)]
    for _, g in groups:
        g = g.sort_values("cycle" if "cycle" in g.columns else g.columns[0])
        arr = g[feature_cols].values.astype(np.float32)
        arr = pd.DataFrame(arr).ffill().bfill().values
        n = len(arr)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            Xs.append(arr[start:start + window])
            ruls.append(g[rul_col].iloc[start + window - 1] if rul_col in g.columns else np.nan)
            fails.append(float(g[fail_col].iloc[start + window - 1]) if fail_col and fail_col in g.columns else 0.0)
    if not Xs:
        return np.empty((0, window, len(feature_cols))), np.empty(0), np.empty(0)
    return np.stack(Xs), np.array(ruls, dtype=np.float32), np.array(fails, dtype=np.float32)


def build_combined_corpus(
    window=50,
    stride=10,
    apply_nigerian_aug=True,
    max_3w_instances=200,
    seed=42,
) -> Dict:
    """Build the full experimental corpus used in the dissertation."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    augmentor = NigerianAugmentor(seed=seed)
    all_X, all_rul, all_fail, all_dom, all_asset = [], [], [], [], []

    # CMAPSS
    try:
        for fd in ["FD001", "FD002", "FD003", "FD004"]:
            train, _ = load_cmapss(fd)
            feat_cols = [c for c in train.columns if c.startswith("s") or c.startswith("op")]
            X, rul, fail = _window_series(train, feat_cols, window, stride)
            n = len(X)
            all_X.append(X); all_rul.append(rul); all_fail.append(fail)
            all_dom.append(np.zeros(n, dtype=np.int64))
            all_asset.append(np.array(["cmaps_turbofan"] * n))
            print(f"  CMAPSS {fd}: {n} windows")
    except FileNotFoundError as e:
        print(f"[WARN] CMAPSS not available: {e}")

    # Real 3W facility data (PRIMARY REAL TARGET)
    try:
        df_3w = load_3w(max_instances=max_3w_instances)
        numeric = df_3w.select_dtypes(include=[np.number]).columns.tolist()
        drop = [c for c in numeric if c.lower() in ("unit", "cycle", "domain_label", "event_label")]
        feat_cols = [c for c in numeric if c not in drop][:21]
        if "cycle" not in df_3w.columns:
            df_3w = df_3w.copy()
            df_3w["cycle"] = df_3w.groupby("unit").cumcount() + 1
        if "RUL" not in df_3w.columns:
            df_3w["RUL"] = 100.0
        X, rul, fail = _window_series(df_3w, feat_cols, window, stride, fail_col=None)
        n = len(X)
        if n > 0:
            all_X.append(X); all_rul.append(rul); all_fail.append(fail)
            all_dom.append(np.ones(n, dtype=np.int64))
            all_asset.append(np.array(["3w_offshore_well"] * n))
            print(f"  3W real facility: {n} windows  <- PRIMARY REAL TARGET DOMAIN")
    except FileNotFoundError as e:
        print(f"[WARN] 3W real facility data not available: {e}")

    # Nigerian-augmented variants
    if apply_nigerian_aug and all_X:
        print("  Generating Nigerian-augmented target variants ...")
        src_X = np.concatenate([x for x, d in zip(all_X, all_dom) if (d == 0).all()], axis=0)
        src_rul = np.concatenate([r for r, d in zip(all_rul, all_dom) if (d == 0).all()], axis=0)
        src_fail = np.concatenate([f for f, d in zip(all_fail, all_dom) if (d == 0).all()], axis=0)
        n_aug = min(len(src_X), 8000)
        idx = rng.choice(len(src_X), size=n_aug, replace=False)
        X_aug_list = []
        for i in idx:
            flat = src_X[i].reshape(-1, src_X.shape[-1])
            cols = [f"s{j}" for j in range(flat.shape[1])]
            aug, _ = augmentor.apply(flat, cols)
            X_aug_list.append(aug.reshape(src_X.shape[1], src_X.shape[2]))
        X_aug = np.stack(X_aug_list)
        all_X.append(X_aug); all_rul.append(src_rul[idx]); all_fail.append(src_fail[idx])
        all_dom.append(np.ones(n_aug, dtype=np.int64))
        all_asset.append(np.array(["nigerian_augmented"] * n_aug))
        print(f"  Nigerian-augmented: {n_aug} windows")

    if not all_X:
        raise RuntimeError("No data could be loaded. Please download at least CMAPSS or 3W.")

    X = np.concatenate(all_X, axis=0)
    rul = np.concatenate(all_rul, axis=0)
    fail = np.concatenate(all_fail, axis=0)
    dom = np.concatenate(all_dom, axis=0)
    asset = np.concatenate(all_asset, axis=0)
    X = np.nan_to_num(X, nan=0.0)

    meta = {
        "n_total": len(X),
        "n_source": int((dom == 0).sum()),
        "n_target": int((dom == 1).sum()),
        "n_real_3w": int((asset == "3w_offshore_well").sum()),
        "window": window,
        "n_channels": X.shape[-1],
        "asset_counts": {a: int((asset == a).sum()) for a in np.unique(asset)},
    }
    print("\n=== Corpus summary ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    out = PROCESSED_DIR / "combined_corpus.npz"
    np.savez_compressed(out, X=X, rul=rul, fail=fail, domain=dom, asset=asset)
    print(f"Saved -> {out}")
    return {"X": X, "rul": rul, "fail": fail, "domain": dom, "asset": asset, "meta": meta}
