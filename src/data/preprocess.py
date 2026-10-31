"""Normalisation and RUL capping."""

from __future__ import annotations

from typing import Dict

import numpy as np


def cap_rul(rul: np.ndarray, cap: float) -> np.ndarray:
    r = np.asarray(rul, dtype=np.float32).copy()
    r = np.minimum(r, float(cap))
    return r


class Normalizer:
    """Per-channel z-score with optional IQR-style clipping on the raw scale."""

    def __init__(self, iqr_k: float = 1.5):
        self.iqr_k = float(iqr_k)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        q1 = np.nanpercentile(flat, 25, axis=0)
        q3 = np.nanpercentile(flat, 75, axis=0)
        iqr = np.maximum(q3 - q1, 1e-8)
        self.lo_ = q1 - self.iqr_k * iqr
        self.hi_ = q3 + self.iqr_k * iqr
        clipped = np.clip(flat, self.lo_, self.hi_)
        self.mean_ = np.nanmean(clipped, axis=0)
        self.std_ = np.nanstd(clipped, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def clip(self, X: np.ndarray) -> np.ndarray:
        assert self.lo_ is not None
        return np.clip(X, self.lo_, self.hi_).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        m, s = self.mean_, self.std_
        if X.shape[-1] != len(m):
            raise ValueError("inverse expects last dim to match fitted channels")
        return (X * s + m).astype(np.float32)

    def to_dict(self) -> Dict:
        return {
            "iqr_k": self.iqr_k,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "lo": self.lo_.tolist(),
            "hi": self.hi_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        n = cls(d.get("iqr_k", 1.5))
        n.mean_ = np.asarray(d["mean"], dtype=np.float64)
        n.std_ = np.asarray(d["std"], dtype=np.float64)
        n.lo_ = np.asarray(d["lo"], dtype=np.float64)
        n.hi_ = np.asarray(d["hi"], dtype=np.float64)
        return n
