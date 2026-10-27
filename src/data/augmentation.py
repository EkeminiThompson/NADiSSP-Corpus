"""
Physics-informed Nigerian augmentation pipeline.

Six perturbation types calibrated to documented Niger Delta / Nigerian
oil & gas operating conditions:

1. Humidity-induced sensor drift
2. Telemetry dropout (power instability)
3. Sabotage / vandalism impulse spikes
4. Progressive corrosion trends
5. Sampling-rate mismatch (legacy vs IoT)
6. Wave-induced cyclic loading (offshore)

Each perturbation is applied selectively to the sensor channels that are
physically capable of exhibiting that degradation mode (see dissertation
Table 2.2).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


class NigerianAugmentor:
    """Apply physics-informed Nigerian stressors to multivariate time-series."""

    def __init__(
        self,
        humidity_drift_std: float = 0.08,
        dropout_prob: float = 0.25,
        sabotage_prob: float = 0.04,
        corrosion_slope: float = 0.015,
        sampling_mismatch_ratio: float = 0.3,
        wave_amplitude: float = 0.12,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.humidity_drift_std = humidity_drift_std
        self.dropout_prob = dropout_prob
        self.sabotage_prob = sabotage_prob
        self.corrosion_slope = corrosion_slope
        self.sampling_mismatch_ratio = sampling_mismatch_ratio
        self.wave_amplitude = wave_amplitude

        self.pressure_channels = ["pressure", "p_pdg", "p_tpt", "p_mon", "discharge_pressure"]
        self.vibration_channels = ["vibration", "vib_x", "vib_y", "vib_z", "vibration_amplitude"]
        self.temperature_channels = ["temperature", "temp", "core_temp", "inlet_temp"]
        self.electrical_channels = ["current", "motor_current", "frequency", "torque"]
        self.flow_channels = ["flow", "flow_rate"]

    def _match_channels(self, columns: List[str], keywords: List[str]) -> List[str]:
        cols_lower = [c.lower() for c in columns]
        matched = []
        for kw in keywords:
            for i, c in enumerate(cols_lower):
                if kw in c:
                    matched.append(columns[i])
        return list(dict.fromkeys(matched))

    def humidity_drift(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        targets = self._match_channels(columns, self.pressure_channels + self.temperature_channels)
        if not targets:
            return X
        n = X.shape[0]
        drift = np.cumsum(self.rng.normal(0, self.humidity_drift_std / np.sqrt(n), size=n))
        for col in targets:
            idx = columns.index(col)
            X[:, idx] = X[:, idx] + drift * np.std(X[:, idx] + 1e-8)
        return X

    def telemetry_dropout(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        targets = self._match_channels(columns, self.temperature_channels + self.pressure_channels)
        if not targets:
            return X
        n = X.shape[0]
        n_blocks = max(1, int(n * self.dropout_prob / 30))
        for _ in range(n_blocks):
            start = self.rng.integers(0, max(1, n - 40))
            length = self.rng.integers(5, 40)
            end = min(n, start + length)
            for col in targets:
                idx = columns.index(col)
                X[start:end, idx] = np.nan
        return X

    def sabotage_impulse(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        targets = self._match_channels(
            columns, self.pressure_channels + self.electrical_channels + self.flow_channels
        )
        if not targets:
            return X
        n = X.shape[0]
        n_events = max(1, int(n * self.sabotage_prob / 50))
        for _ in range(n_events):
            t = self.rng.integers(0, n)
            for col in targets:
                idx = columns.index(col)
                amp = 4.0 * np.std(X[:, idx] + 1e-8) * self.rng.choice([-1.0, 1.0])
                X[t, idx] = X[t, idx] + amp
                if t + 1 < n:
                    X[t + 1, idx] = X[t + 1, idx] + 0.4 * amp
        return X

    def corrosion_trend(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        targets = self._match_channels(columns, self.vibration_channels)
        if not targets:
            return X
        n = X.shape[0]
        t = np.linspace(0, 1, n)
        trend = self.corrosion_slope * t
        for col in targets:
            idx = columns.index(col)
            kernel = np.ones(5) / 5
            smoothed = np.convolve(X[:, idx], kernel, mode="same")
            X[:, idx] = 0.7 * X[:, idx] + 0.3 * smoothed + trend * np.std(X[:, idx] + 1e-8)
        return X

    def sampling_mismatch(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        if self.rng.random() > self.sampling_mismatch_ratio:
            return X
        n = X.shape[0]
        factor = self.rng.choice([2, 4, 5, 10])
        for j in range(X.shape[1]):
            for i in range(0, n, factor):
                end = min(n, i + factor)
                X[i:end, j] = X[i, j]
        return X

    def wave_loading(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        X = X.copy()
        targets = self._match_channels(columns, self.pressure_channels)
        if not targets:
            return X
        n = X.shape[0]
        t = np.arange(n)
        wave = (
            self.wave_amplitude * np.sin(2 * np.pi * t / 45)
            + 0.6 * self.wave_amplitude * np.sin(2 * np.pi * t / 12)
        )
        for col in targets:
            idx = columns.index(col)
            X[:, idx] = X[:, idx] + wave * np.std(X[:, idx] + 1e-8)
        return X

    def apply(
        self,
        X: np.ndarray,
        columns: List[str],
        perturbations: Optional[List[str]] = None,
        return_mask: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        if perturbations is None:
            candidates = [
                "humidity_drift",
                "telemetry_dropout",
                "sabotage_impulse",
                "corrosion_trend",
                "sampling_mismatch",
                "wave_loading",
            ]
            k = self.rng.integers(2, 5)
            perturbations = list(self.rng.choice(candidates, size=k, replace=False))

        meta = {"applied": []}
        X_out = X.copy()

        for name in perturbations:
            fn = getattr(self, name, None)
            if fn is not None:
                X_out = fn(X_out, columns)
                meta["applied"].append(name)

        if return_mask:
            return X_out, meta
        return X_out, None
