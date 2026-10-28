"""
Dataset loaders for NADIP.

Honest labels:
- CMAPSS: simulated turbofan (NASA)
- AI4I 2020: synthetic manufacturing
- ESPset: public ESP data (when present)
- 3W: REAL Petrobras wells only (WELL-*.parquet); skip SIMULATED_* and DRAWN_*

3W is event-classification data (no RUL). Do not invent RUL=100.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_cmapss(fd: str = "FD001", data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """NASA CMAPSS – simulated turbofan (not real plant data)."""
    data_dir = data_dir or RAW_DIR / "CMAPSS"
    data_dir.mkdir(parents=True, exist_ok=True)
    col_names = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    train_path = data_dir / f"train_{fd}.txt"
    test_path = data_dir / f"test_{fd}.txt"
    rul_path = data_dir / f"RUL_{fd}.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"CMAPSS files not found in {data_dir}. Place NASA CMAPSS train/test/RUL .txt files there."
        )
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    def add_rul(df):
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df = df.copy()
        df["RUL"] = max_cycle - df["cycle"]
        return df

    train = add_rul(train)
    test = test.copy()
    test["RUL"] = np.nan
    for i, unit in enumerate(test["unit"].unique()):
        mask = test["unit"] == unit
        max_c = test.loc[mask, "cycle"].max()
        test.loc[mask, "RUL"] = float(rul.iloc[i, 0]) + (max_c - test.loc[mask, "cycle"])
    for df in (train, test):
        df["asset_class"] = "cmaps_simulated"
        df["domain_label"] = 0
        df["has_rul"] = 1
        df["fail"] = (df["RUL"] <= 30).astype(float)
        df["has_fail"] = 1
        df["source_type"] = "simulated"
    return train, test


def load_ai4i(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """UCI AI4I 2020 – synthetic manufacturing (not real plant telemetry)."""
    data_dir = data_dir or RAW_DIR / "AI4I"
    path = data_dir / "ai4i2020.csv"
    if not path.exists():
        raise FileNotFoundError(f"AI4I file not found at {path}.")
    df = pd.read_csv(path)
    rename = {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "fail",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "fail" not in df.columns and "Machine failure" in df.columns:
        df["fail"] = df["Machine failure"]
    df["asset_class"] = "ai4i_synthetic"
    df["domain_label"] = 0
    df["has_rul"] = 0
    df["RUL"] = np.nan
    df["has_fail"] = 1 if "fail" in df.columns else 0
    if "fail" not in df.columns:
        df["fail"] = 0.0
    df["source_type"] = "synthetic"
    df["unit"] = np.arange(len(df))
    df["cycle"] = 1
    return df


def load_espset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = data_dir or RAW_DIR / "ESPset"
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ESPset CSV found in {data_dir}.")
    df = pd.read_csv(csvs[0])
    df["asset_class"] = "espset"
    df["domain_label"] = 0
    df["source_type"] = "public_esp"
    if "RUL" not in df.columns:
        df["RUL"] = np.nan
        df["has_rul"] = 0
    else:
        df["has_rul"] = 1
    if "fail" not in df.columns and "failure" in df.columns:
        df["fail"] = df["failure"]
    if "fail" not in df.columns:
        df["fail"] = 0.0
        df["has_fail"] = 0
    else:
        df["has_fail"] = 1
    if "unit" not in df.columns:
        df["unit"] = 0
    return df


def _infer_3w_source(path: Path) -> str:
    """3W v2 naming: WELL-*.parquet = real; SIMULATED_*; DRAWN_*."""
    name = path.name.lower()
    if name.startswith("drawn") or "drawn" in name:
        return "hand_drawn"
    if name.startswith("simulated") or name.startswith("sim"):
        return "simulated"
    if name.startswith("well-") or name.startswith("real"):
        return "real"
    parts = "/".join(x.lower() for x in path.parts)
    if "drawn" in parts:
        return "hand_drawn"
    if "simul" in parts:
        return "simulated"
    if "real" in parts:
        return "real"
    return "unknown"


def load_3w_real(
    data_dir: Optional[Path] = None,
    max_instances: Optional[int] = None,
    require_real: bool = True,
) -> pd.DataFrame:
    """
    Load Petrobras 3W – REAL instances only by default (WELL-*.parquet).

    Event classification only. has_rul=0, RUL=NaN. Never invent RUL=100.
    fail: non-normal event labels.
    """
    data_dir = data_dir or RAW_DIR / "3W"
    parquet_files = list(data_dir.rglob("*.parquet"))
    csv_files = list(data_dir.rglob("*.csv"))
    files = parquet_files if parquet_files else csv_files
    if not files:
        raise FileNotFoundError(
            f"3W data not found in {data_dir}. Clone https://github.com/petrobras/3W into data/raw/3W"
        )

    frames = []
    n_real = n_skip = 0
    for f in tqdm(files, desc="Loading 3W"):
        source = _infer_3w_source(f)
        if require_real and source in ("simulated", "hand_drawn"):
            n_skip += 1
            continue
        if require_real and source == "unknown":
            n_skip += 1
            continue
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
            n_skip += 1
            continue

        df = df.copy()
        df["unit"] = f.stem
        df["instance_path"] = str(f)
        df["source_type"] = "real"
        df["asset_class"] = "3w_offshore_well_real"
        df["domain_label"] = 1
        df["has_rul"] = 0
        df["RUL"] = np.nan

        label_col = next((c for c in ("class", "label", "event", "state") if c in df.columns), None)
        if label_col is not None:
            vals = pd.to_numeric(df[label_col], errors="coerce")
            is_event = vals.notna() & (vals != 0) & ((vals < 100) | (vals >= 101))
            df["fail"] = is_event.astype(float)
            df["event_class"] = vals
            df["has_fail"] = 1
        else:
            parent = f.parent.name
            if parent.isdigit():
                cls = int(parent)
                df["event_class"] = cls
                df["fail"] = float(cls != 0)
                df["has_fail"] = 1
            else:
                df["fail"] = 0.0
                df["has_fail"] = 0
                df["event_class"] = 0

        if "cycle" not in df.columns:
            df["cycle"] = np.arange(1, len(df) + 1)

        frames.append(df)
        n_real += 1
        if max_instances and n_real >= max_instances:
            break

    if not frames:
        raise RuntimeError("No real 3W instances loaded (WELL-*.parquet).")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {n_real} REAL 3W instances (skipped {n_skip} sim/drawn). Rows={len(combined):,}")
    return combined


load_3w = load_3w_real


def list_available() -> Dict[str, bool]:
    _ensure_dirs()
    return {
        "CMAPSS_simulated": (RAW_DIR / "CMAPSS").exists() and any((RAW_DIR / "CMAPSS").glob("train_*.txt")),
        "AI4I_synthetic": (RAW_DIR / "AI4I" / "ai4i2020.csv").exists(),
        "ESPset": any((RAW_DIR / "ESPset").glob("*.csv")),
        "3W_real_facility": any((RAW_DIR / "3W").rglob("WELL-*.parquet")),
    }
