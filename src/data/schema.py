"""Unified channel layout and channel roles.

The four source datasets have different sensors (Table 3.6b). The transformer needs one fixed
input width, so every dataset owns a block of "slots" in a unified vector; slots belonging to other
datasets are zero. Source and Nigerian-augmented variants of the same dataset occupy the same slots,
so the domain shift the model sees is exactly the augmentation.

Every channel carries a *role* (pressure, temperature, vibration, electrical, flow, other). The
augmentor selects channels by role -- never by guessing from column names -- which is what makes
the physics-informed stressors actually reach the signals they are meant to perturb.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

DATASETS: List[str] = ["cmapss", "ai4i", "w3", "espset"]
DATASET_ID: Dict[str, int] = {d: i for i, d in enumerate(DATASETS)}

# (channel name, role)
_CMAPSS_ROLES = {
    "temperature": ["s1", "s2", "s3", "s4"],               # T2, T24, T30, T50
    "pressure": ["s5", "s6", "s7", "s10", "s11"],          # P2, P15, P30, epr, Ps30
    "vibration": ["s8", "s9", "s13", "s14"],               # speed channels: the document's "rotational-speed proxies" (§3.4.3(d))
    "flow": ["s12", "s20", "s21"],                         # phi, W31, W32
    "other": ["s15", "s16", "s17", "s18", "s19"],
}


def _cmapss_channels() -> List[Tuple[str, str]]:
    role_of = {c: r for r, cs in _CMAPSS_ROLES.items() for c in cs}
    return [(f"s{i}", role_of[f"s{i}"]) for i in range(1, 22)]


CHANNELS: Dict[str, List[Tuple[str, str]]] = {
    "cmapss": _cmapss_channels(),
    "ai4i": [
        ("air_temp", "temperature"),
        ("process_temp", "temperature"),
        ("rpm", "electrical"),      # AI4I has no pressure/flow channel; drive speed + torque carry the
        ("torque", "electrical"),   # "motor current / frequency / torque" sabotage channels of §3.4.3(c)
        ("tool_wear", "other"),
    ],
    # the seven 3W sensors of Table 3.6b
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
        ("frequency", "electrical"),
    ],
}

# Slot offsets in the unified vector
OFFSETS: Dict[str, int] = {}
_off = 0
for _d in DATASETS:
    OFFSETS[_d] = _off
    _off += len(CHANNELS[_d])
N_CHANNELS: int = _off

STRESSORS: List[str] = ["humidity", "dropout", "sabotage", "corrosion", "sampling", "wave"]

# Which roles each stressor may touch (§3.3.4.1, Table 3.9a, §3.4.3). "all" = every channel.
STRESSOR_ROLES: Dict[str, List[str]] = {
    "humidity": ["pressure", "vibration", "temperature"],
    "dropout": ["temperature", "pressure"],
    "sabotage": ["pressure", "flow", "electrical"],
    "corrosion": ["pressure", "vibration"],
    "sampling": ["all"],
    "wave": ["pressure", "vibration"],
}


def channel_names(dataset: str) -> List[str]:
    return [c for c, _ in CHANNELS[dataset]]


def channel_roles(dataset: str) -> List[str]:
    return [r for _, r in CHANNELS[dataset]]


def role_indices(dataset: str, roles: List[str]) -> np.ndarray:
    """Indices (within the dataset's own channel list) whose role is in `roles`."""
    if "all" in roles:
        return np.arange(len(CHANNELS[dataset]))
    return np.array([i for i, (_, r) in enumerate(CHANNELS[dataset]) if r in roles], dtype=int)


def unified_names() -> List[str]:
    return [f"{d}:{c}" for d in DATASETS for c in channel_names(d)]


def to_unified(dataset: str, X: np.ndarray) -> np.ndarray:
    """(n, T, C_d) -> (n, T, N_CHANNELS), other datasets' slots zero."""
    out = np.zeros(X.shape[:-1] + (N_CHANNELS,), dtype=np.float32)
    o = OFFSETS[dataset]
    out[..., o:o + X.shape[-1]] = X
    return out


def from_unified(dataset: str, X: np.ndarray) -> np.ndarray:
    o = OFFSETS[dataset]
    return X[..., o:o + len(CHANNELS[dataset])]
