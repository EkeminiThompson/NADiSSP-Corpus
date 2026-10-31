"""Preprocessing (Table 3.13): forward-fill imputation, 1.5xIQR clipping, per-sensor z-scoring.

Statistics are fitted on the *clean source, training-split* windows of each dataset only and then
applied to source and Nigerian-augmented windows alike. Fitting on both would normalise the very
shift the study measures; fitting on the test split would leak.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def fill_series(X: np.ndarray) -> np.ndarray:
    """Time-aware forward-fill, then back-fill leading gaps; channels that are entirely missing become 0."""
    df = pd.DataFrame(X).ffill().bfill()
    return df.fillna(0.0).to_numpy(dtype=np.float32)


def missing_fraction(X: np.ndarray) -> float:
    return float(np.isnan(X).mean())


class Normalizer:
    """Per-sensor IQR clipping bounds and z-score statistics for one dataset."""

    def __init__(self, iqr_k: float = 1.5):
        self.iqr_k = iqr_k
        self.lo = self.hi = self.mean = self.std = None

    def fit(self, windows: np.ndarray) -> "Normalizer":
        flat = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
        q1, q3 = np.percentile(flat, [25, 75], axis=0)
        iqr = q3 - q1
        self.lo, self.hi = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
        clipped = np.clip(flat, self.lo, self.hi)
        self.mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)          # constant channels stay constant rather than exploding
        return self

    def clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo, self.hi).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return (Z * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict[str, list]:
        return {k: getattr(self, k).tolist() for k in ("lo", "hi", "mean", "std")} | {"iqr_k": self.iqr_k}

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        n = cls(d["iqr_k"])
        for k in ("lo", "hi", "mean", "std"):
            setattr(n, k, np.asarray(d[k], dtype=np.float64))
        return n


def cap_rul(rul_raw: np.ndarray, cap: float) -> np.ndarray:
    """Piecewise-linear RUL: min(RUL, cap) (Table 3.13)."""
    return np.minimum(rul_raw, cap).astype(np.float32)
