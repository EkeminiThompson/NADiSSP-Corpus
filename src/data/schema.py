"""Unified channel schema for NADIP (Table 3.6b sensors + roles).

Each dataset occupies a contiguous slice of a length-N_CHANNELS vector. Channels
outside a dataset's own block are zero. Roles drive Table 3.9b stressor targets.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

CHANNELS: Dict[str, List[Tuple[str, str]]] = {
    "cmapss": [(f"s{i}", r) for i, r in [
        (1, "temperature"), (2, "temperature"), (3, "temperature"), (4, "temperature"),
        (5, "pressure"), (6, "pressure"), (7, "pressure"), (8, "pressure"),
        (9, "speed"), (10, "pressure"), (11, "temperature"), (12, "pressure"),
        (13, "speed"), (14, "pressure"), (15, "pressure"), (16, "pressure"),
        (17, "temperature"), (18, "speed"), (19, "speed"), (20, "flow"), (21, "flow"),
    ]],
    "ai4i": [
        ("air_temp", "temperature"),
        ("process_temp", "temperature"),
        ("rpm", "speed"),
        ("torque", "electrical"),
        ("tool_wear", "wear"),
    ],
    "w3": [
        ("P-PDG", "pressure"),
        ("P-TPT", "pressure"),
        ("T-TPT", "temperature"),
        ("P-MON-CKP", "pressure"),
        ("T-JUS-CKP", "temperature"),
        ("P-JUS-CKGL", "pressure"),
        ("QGL", "flow"),
    ],
    "espset": [
        ("vib_x", "vibration"),
        ("vib_y", "vibration"),
        ("vib_z", "vibration"),
        ("motor_current", "electrical"),
        ("p_discharge", "pressure"),
        ("p_intake", "pressure"),
        ("t_motor", "temperature"),
        ("frequency", "speed"),
    ],
}

DATASETS: List[str] = list(CHANNELS.keys())
DATASET_ID: Dict[str, int] = {d: i for i, d in enumerate(DATASETS)}

_OFFSETS: List[int] = []
_o = 0
for _d in DATASETS:
    _OFFSETS.append(_o)
    _o += len(CHANNELS[_d])
OFFSETS: Dict[str, int] = {d: o for d, o in zip(DATASETS, _OFFSETS)}
N_CHANNELS: int = _o  # 21+5+7+8 = 41

STRESSORS: List[str] = ["humidity", "dropout", "sabotage", "corrosion", "sampling", "wave"]

STRESSOR_ROLES: Dict[str, List[str]] = {
    "humidity": ["temperature", "pressure"],
    "dropout": ["temperature", "pressure"],
    "sabotage": ["pressure", "electrical", "flow", "speed"],
    "corrosion": ["vibration", "pressure"],
    "sampling": ["temperature", "pressure", "vibration", "electrical", "speed", "flow", "wear"],
    "wave": ["pressure"],
}


def channel_names(dataset: str) -> List[str]:
    return [c for c, _ in CHANNELS[dataset]]


def role_indices(dataset: str, roles: Sequence[str]) -> np.ndarray:
    role_set = set(roles)
    return np.array([i for i, (_, r) in enumerate(CHANNELS[dataset]) if r in role_set], dtype=np.int64)


def unified_names() -> List[str]:
    return [f"{d}:{c}" for d in DATASETS for c, _ in CHANNELS[d]]


def to_unified(X, dataset: str = None) -> np.ndarray:
    if isinstance(X, str):
        X, dataset = dataset, X
    o, n = OFFSETS[dataset], len(CHANNELS[dataset])
    if X.ndim == 2:
        out = np.zeros((X.shape[0], N_CHANNELS), dtype=np.float32)
        out[:, o:o + n] = X[:, :n]
        return out
    if X.ndim == 3:
        out = np.zeros((X.shape[0], X.shape[1], N_CHANNELS), dtype=np.float32)
        out[:, :, o:o + n] = X[:, :, :n]
        return out
    raise ValueError(f"expected 2-D or 3-D, got {X.ndim}-D")
