"""Physics-informed Nigerian augmentation (Tables 3.9a / 3.9b, §3.3.4, §3.4.3).

Six stressors, applied *conditionally by asset analogy* (the matrix in configs/nadip.yaml) and
*by sensor role* (schema.STRESSOR_ROLES). Channels are selected from the dataset's own role table,
so a stressor can never silently miss its target channels. `tests/test_augmentation.py` asserts that
every stressor in the matrix changes the channels it is supposed to change and nothing else.

Operates on RAW (un-normalised) windows of shape (T, C_d) for a single dataset; the caller
normalises afterwards with statistics fitted on the clean source data, so the shift is preserved.

Interpretation notes (the document is not fully explicit):
* humidity    : slow random-walk baseline shift bounded by +/-drift_bound x baseline (§3.4.3(a)) plus
                Gaussian noise of sigma = U(sigma_range) x RH/100 x channel std (Table 3.9a).
* dropout     : block dropout (5-20 samples) until 25-40% of the window is missing, forward-fill
                imputation (Table 3.9a). The number of dropped blocks is returned as `n_gaps`.
* sabotage    : impulses of 3-8 x channel std with probability `prob` per sample (§3.4.3(c)).
* corrosion   : exponential baseline trend ending at 3-15% of baseline (the "wear-out" magnitude of
                §3.4.3), curvature set by lambda_range, plus 10-20% attenuation of the high-frequency
                part of vibration-role channels (§3.4.3(d)).
* sampling    : sample-and-hold downsampling by one of `factors` with +/-10% index jitter.
* wave        : sinusoid, 0.1-0.5 Hz at 1 Hz sampling, amplitude +/-8% of baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import STRESSORS, STRESSOR_ROLES, role_indices

_EPS = 1e-8


def _ffill(x: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Forward-fill positions where `missing` is True; leading gaps take the first valid value."""
    n = len(x)
    idx = np.where(~missing, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    first_valid = int(np.argmax(~missing)) if (~missing).any() else 0
    idx = np.where(idx < 0, first_valid, idx)
    return x[idx]


class NigerianAugmentor:
    def __init__(self, cfg: Dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ helpers
    def _stats(self, x: np.ndarray) -> Tuple[float, float]:
        s = float(np.std(x))
        base = max(abs(float(np.mean(x))), s, _EPS)
        return max(s, _EPS * base), base

    def _channels(self, dataset: str, name: str) -> np.ndarray:
        return role_indices(dataset, STRESSOR_ROLES[name])

    def _u(self, lo_hi) -> float:
        return float(self.rng.uniform(lo_hi[0], lo_hi[1]))

    # ------------------------------------------------------------------ stressors
    def humidity(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["humidity"]
        scale = c["dataset_scale"].get(dataset, 1.0)
        rh = self._u(c["rh_range"])
        sigma = self._u(c["sigma_range"]) * rh / 100.0 * scale
        T = X.shape[0]
        for j in self._channels(dataset, "humidity"):
            s, base = self._stats(X[:, j])
            walk = np.cumsum(self.rng.normal(size=T))
            walk = walk / (np.max(np.abs(walk)) + _EPS) * self._u((0.3, 1.0))
            X[:, j] += walk * c["drift_bound"] * scale * base + self.rng.normal(0.0, sigma * s, size=T)
        return X, {}

    def dropout(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["dropout"]
        T = X.shape[0]
        target = int(round(self._u(c["fraction_range"][dataset]) * T))
        missing = np.zeros(T, dtype=bool)
        n_blocks, guard = 0, 0
        while missing.sum() < target and guard < 200:
            guard += 1
            length = int(self.rng.integers(c["block_range"][0], c["block_range"][1] + 1))
            start = int(self.rng.integers(0, max(1, T - length + 1)))
            before = missing.sum()
            missing[start:start + length] = True
            if missing.sum() > before:
                n_blocks += 1
        for j in self._channels(dataset, "dropout"):
            X[:, j] = _ffill(X[:, j], missing)            # transmitters lose power together; forward-fill imputation
        return X, {"n_gaps": n_blocks}

    def sabotage(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["sabotage"]
        T = X.shape[0]
        for j in self._channels(dataset, "sabotage"):
            s, _ = self._stats(X[:, j])
            hit = self.rng.random(T) < c["prob"]
            amp = self.rng.uniform(c["amp_range"][0], c["amp_range"][1], size=T) * self.rng.choice([-1.0, 1.0], size=T)
            X[:, j] += hit * amp * s
        return X, {}

    def corrosion(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["corrosion"]
        T = X.shape[0]
        t = np.linspace(0.0, 1.0, T)
        curv = self._u(c["lambda_range"]) * 100.0                     # 1-3: how strongly exponential the trend is
        shape = (np.exp(curv * t) - 1.0) / (np.exp(curv) - 1.0)
        for j in self._channels(dataset, "corrosion"):
            s, base = self._stats(X[:, j])
            X[:, j] += self._u((0.03, 0.15)) * base * shape
        vib = role_indices(dataset, ["vibration"])
        for j in vib:
            a = self._u(c["hf_attenuation"])
            smooth = np.convolve(np.pad(X[:, j], 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
            X[:, j] = X[:, j] - a * (X[:, j] - smooth)
        return X, {}

    def sampling(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["sampling"]
        T = X.shape[0]
        f = int(self.rng.choice(c["factors"]))
        pts = np.arange(0, T, f)
        jit = np.rint(self.rng.uniform(-c["jitter"], c["jitter"], size=len(pts)) * f).astype(int)
        pts = np.unique(np.clip(pts + jit, 0, T - 1))
        hold = np.searchsorted(pts, np.arange(T), side="right") - 1
        hold = np.where(hold < 0, 0, hold)
        return X[pts[hold], :], {}

    def wave(self, X: np.ndarray, dataset: str) -> Tuple[np.ndarray, Dict]:
        c = self.cfg["wave"]
        T = X.shape[0]
        f = self._u(c["freq_hz_range"])
        phase = self.rng.uniform(0, 2 * np.pi)
        wave = np.sin(2 * np.pi * f * np.arange(T) / c["sample_rate_hz"] + phase)
        for j in self._channels(dataset, "wave"):
            _, base = self._stats(X[:, j])
            X[:, j] += c["amplitude"] * base * wave
        return X, {}

    # ------------------------------------------------------------------ driver
    def applicable(self, dataset: str) -> List[str]:
        allowed = set(self.cfg["matrix"].get(dataset, []))
        return [s for s in STRESSORS if s in allowed]

    def apply(self, X: np.ndarray, dataset: str, stressors: Optional[List[str]] = None,
              force: bool = False) -> Tuple[np.ndarray, Dict]:
        """Perturb one raw window. `stressors` restricts to a subset; `force` ignores apply_prob.

        Returns (X_out, info) with info = {"applied": [...], "n_gaps": int}.
        """
        X = np.array(X, dtype=np.float64, copy=True)
        applied: List[str] = []
        n_gaps = 0
        for name in self.applicable(dataset):
            if stressors is not None and name not in stressors:
                continue
            if not force and self.rng.random() >= self.cfg["apply_prob"].get(name, 1.0):
                continue
            X, extra = getattr(self, name)(X, dataset)
            applied.append(name)
            n_gaps += extra.get("n_gaps", 0)
        return X.astype(np.float32), {"applied": applied, "n_gaps": n_gaps}
