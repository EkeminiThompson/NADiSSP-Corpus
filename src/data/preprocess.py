"""Normalisation and RUL capping for the NADiP corpus."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def cap_rul(rul: np.ndarray, cap: float) -> np.ndarray:
    out = np.asarray(rul, dtype=np.float32).copy()
    mask = np.isfinite(out)
    out[mask] = np.minimum(out[mask], float(cap))
    return out


class Normalizer:
    """Per-dataset channel-wise mean/std fitted on clean source train windows only."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        # X: (N, T, C) or (N, C)
        if X.ndim == 3:
            flat = X.reshape(-1, X.shape[-1])
        else:
            flat = X
        self.mean_ = np.nanmean(flat, axis=0).astype(np.float32)
        self.std_ = np.nanstd(flat, axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted")
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer not fitted")
        return (X * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: Dict[str, np.ndarray]) -> "Normalizer":
        self.mean_ = np.asarray(d["mean"], dtype=np.float32)
        self.std_ = np.asarray(d["std"], dtype=np.float32)
        return self
