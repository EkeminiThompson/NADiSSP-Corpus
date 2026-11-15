"""Loaders for the four source datasets (Table 3.6a / 3.12b).

Each loader returns a list of `Series` (one per engine / well instance / ESP unit / instance block).
Nothing is invented:

* CMAPSS      RUL from the run-to-failure structure (+ the RUL_FD00x.txt files for the test engines).
* AI4I 2020   per-instance records with no time axis and no RUL. Consecutive instances are grouped
              into blocks purely so that windows and train/test splits have a grouping unit;
              `meta["pseudo_sequence"]` marks this. has_rul=False, so RUL losses/metrics skip it.
* 3W          event-classification data. No RUL. Provenance (real WELL-* / SIMULATED_*) is kept per
              series; hand-drawn instances are excluded. The Table 3.13 event segmentation (+/-200
              samples around the labelled onset) is applied here.
* ESPset      loaded from CSV with flexible column aliases (see ESP_ALIASES). A time-to-failure
              proxy is produced only for units that contain a labelled fault (`rul_source`
              records this); units without a fault have has_rul=False.

A missing dataset raises `DatasetNotFound`; the corpus builder records it instead of fabricating data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import CHANNELS, channel_names


class DatasetNotFound(FileNotFoundError):
    pass


@dataclass
class Series:
    dataset: str
    unit: str
    X: np.ndarray                      # (T, C_d) raw values, schema channel order
    rul_raw: np.ndarray                # (T,) uncapped RUL / TTF proxy; NaN where unknown
    has_rul: bool
    fail: np.ndarray                   # (T,) {0,1}: critical failure / steady event
    health: np.ndarray                 # (T,) {0 normal, 1 early degradation, 2 critical}
    event_class: np.ndarray            # (T,) int stratification code (event type / failure type / fault class)
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------------------- CMAPSS
CMAPSS_SUBSETS = {
    "FD001": {"op_conditions": 1, "fault_modes": 1},
    "FD002": {"op_conditions": 6, "fault_modes": 1},
    "FD003": {"op_conditions": 1, "fault_modes": 2},
    "FD004": {"op_conditions": 6, "fault_modes": 2},
}


def _health_from_rul(rul: np.ndarray, critical: float, early: float) -> np.ndarray:
    h = np.zeros(len(rul), dtype=np.int8)
    h[rul <= early] = 1
    h[rul <= critical] = 2
    return h


def load_cmapss(raw_dir: Path, subsets: List[str], critical_rul: float, early_rul: float) -> List[Series]:
    d = Path(raw_dir) / "CMAPSS"
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    sensor_cols = channel_names("cmapss")
    out: List[Series] = []
    found = False
    for fd in subsets:
        train_p, test_p, rul_p = d / f"train_{fd}.txt", d / f"test_{fd}.txt", d / f"RUL_{fd}.txt"
        if not train_p.exists():
            continue
        found = True
        info = CMAPSS_SUBSETS[fd]
        train = pd.read_csv(train_p, sep=r"\s+", header=None, names=cols)
        frames = [("train", train, None)]
        if test_p.exists() and rul_p.exists():
            test = pd.read_csv(test_p, sep=r"\s+", header=None, names=cols)
            rul_final = pd.read_csv(rul_p, sep=r"\s+", header=None, names=["RUL"])["RUL"].to_numpy()
            frames.append(("test", test, rul_final))
        for part, df, rul_final in frames:
            for i, (unit, g) in enumerate(df.groupby("unit", sort=True)):
                g = g.sort_values("cycle")
                last = g["cycle"].max()
                if rul_final is None:
                    rul = (last - g["cycle"]).to_numpy(dtype=np.float32)
                else:
                    rul = (rul_final[i] + last - g["cycle"]).to_numpy(dtype=np.float32)
                out.append(Series(
                    dataset="cmapss", unit=f"{fd}-{part}-{int(unit)}",
                    X=g[sensor_cols].to_numpy(dtype=np.float32), rul_raw=rul, has_rul=True,
                    fail=(rul <= critical_rul).astype(np.int8),
                    health=_health_from_rul(rul, critical_rul, early_rul),
                    event_class=_health_from_rul(rul, critical_rul, early_rul).astype(np.int16),
                    meta={"subset": fd, "part": part, **info},
                ))
    if not found:
        raise DatasetNotFound(f"No train_FD00x.txt under {d}. Place the NASA CMAPSS .txt files there.")
    return out


# ------------------------------------------------------------------------------------------ AI4I
AI4I_RENAME = {
    "Air temperature [K]": "air_temp", "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
}
AI4I_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_ai4i(raw_dir: Path, block: int = 500) -> List[Series]:
    p = Path(raw_dir) / "AI4I" / "ai4i2020.csv"
    if not p.exists():
        raise DatasetNotFound(f"AI4I file not found: {p}")
    df = pd.read_csv(p).rename(columns=AI4I_RENAME)
    if "UDI" in df.columns:
        df = df.sort_values("UDI")
    df = df.reset_index(drop=True)
    modes = df[[m for m in AI4I_MODES if m in df.columns]].to_numpy()
    code = np.zeros(len(df), dtype=np.int16)
    for j in range(modes.shape[1]):
        code = np.where((code == 0) & (modes[:, j] == 1), j + 1, code)
    fail = df["Machine failure"].to_numpy(dtype=np.int8)
    X = df[channel_names("ai4i")].to_numpy(dtype=np.float32)
    out: List[Series] = []
    for b, start in enumerate(range(0, len(df), block)):
        sl = slice(start, min(len(df), start + block))
        out.append(Series(
            dataset="ai4i", unit=f"block{b:03d}", X=X[sl],
            rul_raw=np.full(sl.stop - sl.start, np.nan, dtype=np.float32), has_rul=False,
            fail=fail[sl], health=(fail[sl] * 2).astype(np.int8), event_class=code[sl],
            meta={"pseudo_sequence": True},
        ))
    return out


# --------------------------------------------------------------------------------------------- 3W
W3_SENSORS = channel_names("w3")
# 3W Dataset 2.0.0 (dataset.ini): classes 0 = normal, 1..9 = undesirable event types (steady), 101..109 = their transients.
STEADY = set(range(1, 10))
TRANSIENT = set(range(101, 110))


def _w3_provenance(path: Path) -> str:
    n = path.name.lower()
    if "drawn" in n:
        return "drawn"
    if n.startswith("simulated") or n.startswith("sim"):
        return "simulated"
    if n.startswith("well"):
        return "real"
    return "unknown"


def load_w3(raw_dir: Path, half_width: int, window: int, max_files: Optional[int] = None) -> List[Series]:
    root = Path(raw_dir) / "3W"
    files = sorted(list(root.rglob("*.parquet")) + list(root.rglob("*.csv")))
    if not files:
        raise DatasetNotFound(f"No 3W instances under {root}. Clone https://github.com/petrobras/3W into data/raw/3W.")
    out: List[Series] = []
    skipped = {"drawn": 0, "unknown": 0, "unreadable": 0}
    for f in files:
        prov = _w3_provenance(f)
        if prov in ("drawn", "unknown"):
            skipped[prov] += 1
            continue
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception:
            skipped["unreadable"] += 1
            continue
        if "class" not in df.columns:
            skipped["unreadable"] += 1
            continue
        cls_raw = pd.to_numeric(df["class"], errors="coerce")
        labelled = cls_raw.notna().to_numpy()                                   # <NA> = unlabelled; never treated as normal
        if labelled.sum() < window:
            skipped["unreadable"] += 1
            continue
        first, last = int(np.argmax(labelled)), int(len(labelled) - np.argmax(labelled[::-1]))
        df, cls_raw = df.iloc[first:last], cls_raw.iloc[first:last]
        cls = cls_raw.fillna(0).to_numpy().astype(int)                          # interior gaps (rare) fall back to normal
        X = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32) if c in df.columns
                      else np.full(len(df), np.nan, dtype=np.float32) for c in W3_SENSORS], axis=1)
        event_idx = np.flatnonzero(cls != 0)
        centre = int(event_idx[0]) if len(event_idx) else len(df) // 2          # Table 3.13: window around event onset
        lo, hi = max(0, centre - half_width), min(len(df), centre + half_width + 1)
        if hi - lo < window:
            skipped["unreadable"] += 1
            continue
        cls, X = cls[lo:hi], X[lo:hi]
        fail = np.isin(cls, list(STEADY)).astype(np.int8)
        health = np.where(fail == 1, 2, np.where(np.isin(cls, list(TRANSIENT)), 1, 0)).astype(np.int8)
        ev = np.where(cls >= 100, cls - 100, cls).astype(np.int16)              # event type 0..9, transient folded in
        out.append(Series(
            dataset="w3", unit=f.stem, X=X, rul_raw=np.full(len(cls), np.nan, dtype=np.float32), has_rul=False,
            fail=fail, health=health, event_class=ev, meta={"provenance": prov},
        ))
        if max_files and len(out) >= max_files:
            break
    if not out:
        raise DatasetNotFound(f"3W files found under {root} but none usable (skipped: {skipped}).")
    return out


# ---------------------------------------------------------------------------------------- ESPset
# Accepted column spellings (case-insensitive). The document's inventory is vibration x/y/z, motor
# current, discharge/intake pressure, motor temperature and VSD frequency (Table 3.6b).
ESP_ALIASES: Dict[str, List[str]] = {
    "vib_x": ["vib_x", "vibration_x", "vx", "accel_x", "ax"],
    "vib_y": ["vib_y", "vibration_y", "vy", "accel_y", "ay"],
    "vib_z": ["vib_z", "vibration_z", "vz", "accel_z", "az"],
    "motor_current": ["motor_current", "current", "i_motor", "amps"],
    "p_discharge": ["p_discharge", "discharge_pressure", "pdis", "p_dis"],
    "p_intake": ["p_intake", "intake_pressure", "pin", "p_in"],
    "t_motor": ["t_motor", "motor_temp", "motor_temperature", "temperature", "temp"],
    "frequency": ["frequency", "freq", "vsd_frequency", "hz"],
}
ESP_UNIT_COLS = ["unit", "esp", "esp_id", "well", "pump", "pump_id", "unit_id"]
ESP_LABEL_COLS = ["fault", "fault_class", "class", "label", "state", "failure"]
ESP_NORMAL = {"0", "normal", "healthy", "none", "nan", ""}


def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def load_espset(raw_dir: Path) -> List[Series]:
    root = Path(raw_dir) / "ESPset"
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise DatasetNotFound(f"No ESPset CSV under {root}. See README (ESPset layout) for the accepted columns.")
    names = channel_names("espset")
    out: List[Series] = []
    for f in files:
        df = pd.read_csv(f)
        cols = {n: _first_col(df, ESP_ALIASES[n]) for n in names}
        present = [n for n, c in cols.items() if c is not None]
        if len(present) < 3:
            continue
        unit_col = _first_col(df, ESP_UNIT_COLS)
        label_col = _first_col(df, ESP_LABEL_COLS)
        fault_map: Dict[str, int] = {"normal": 0}
        if label_col is not None:
            all_lab = df[label_col].astype(str).str.strip().str.lower()
            for i, u in enumerate(sorted(set(all_lab[~all_lab.isin(ESP_NORMAL)]))):
                fault_map[u] = i + 1
        groups =df.groupby(unit_col, sort=True) if unit_col else [(f.stem, df)]
        for unit, g in groups:
            g = g.reset_index(drop=True)
            X = np.stack([pd.to_numeric(g[cols[n]], errors="coerce").to_numpy(dtype=np.float32)
                          if cols[n] is not None else np.full(len(g), np.nan, dtype=np.float32) for n in names], axis=1)
            if label_col is not None:
                lab = g[label_col].astype(str).str.strip().str.lower()
                is_fault = (~lab.isin(ESP_NORMAL)).to_numpy()
                lab = lab.where(is_fault, "normal")
                codes = lab.map(fault_map).fillna(0).to_numpy()      # 0 = normal, 1..k = fault class (per file)
            else:
                is_fault = np.zeros(len(g), dtype=bool)
                codes = np.zeros(len(g), dtype=int)
            idx = np.arange(len(g), dtype=np.float32)
            if is_fault.any():
                t_fault = float(np.flatnonzero(is_fault)[0])
                rul = np.clip(t_fault - idx, 0, None)
                has_rul, rul_src = True, "ttf_proxy"
            else:
                rul = np.full(len(g), np.nan, dtype=np.float32)
                has_rul, rul_src = False, "none"
            health = np.where(is_fault, 2, 0).astype(np.int8)
            out.append(Series(
                dataset="espset", unit=f"{f.stem}:{unit}", X=X, rul_raw=rul.astype(np.float32), has_rul=has_rul,
                fail=is_fault.astype(np.int8), health=health, event_class=codes.astype(np.int16),
                meta={"rul_source": rul_src, "channels_present": present},
            ))
    if not out:
        raise DatasetNotFound(f"ESPset CSVs under {root} did not contain recognisable sensor columns.")
    return out


def load_all(raw_dir: Path, cfg: Dict):
    """Returns (found, missing): dict dataset -> [Series], dict dataset -> reason."""
    """Load every requested dataset; datasets that are absent are skipped and reported by the caller."""
    d = cfg["data"]
    loaders = {
        "cmapss": lambda: load_cmapss(raw_dir, d["cmapss_subsets"], d["critical_rul"], d["early_degradation_rul"]),
        "ai4i": lambda: load_ai4i(raw_dir),
        "w3": lambda: load_w3(raw_dir, d["w3_half_width"], d["window"]),
        "espset": lambda: load_espset(raw_dir),
    }
    found: Dict[str, List[Series]] = {}
    missing: Dict[str, str] = {}
    for name in d["datasets"]:
        try:
            found[name] = loaders[name]()
        except DatasetNotFound as e:
            missing[name] = str(e)
    return found, missing
