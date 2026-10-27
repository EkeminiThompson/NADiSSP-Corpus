"""
Dataset loaders for NADIP.

Supported:
- NASA CMAPSS (FD001-FD004)
- UCI AI4I 2020
- ESPset (electrical submersible pumps)
- Petrobras 3W Dataset 2.0.0  <- primary REAL FACILITY target domain
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_cmapss(fd: str = "FD001", data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or RAW_DIR / "CMAPSS"
    data_dir.mkdir(parents=True, exist_ok=True)
    col_names = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    train_path = data_dir / f"train_{fd}.txt"
    test_path = data_dir / f"test_{fd}.txt"
    rul_path = data_dir / f"RUL_{fd}.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"CMAPSS files not found in {data_dir}. Run scripts/download_datasets.py or place NASA CMAPSS .txt files manually."
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
        test.loc[mask, "RUL"] = rul.iloc[i, 0] + (max_c - test.loc[mask, "cycle"])
    train["asset_class"] = "cmaps_turbofan"
    test["asset_class"] = "cmaps_turbofan"
    train["domain_label"] = 0
    test["domain_label"] = 0
    return train, test


def load_ai4i(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = data_dir or RAW_DIR / "AI4I"
    data_dir.mkdir(parents=True, exist_ok=True)
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
        "Machine failure": "failure",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["asset_class"] = "ai4i_manufacturing"
    df["domain_label"] = 0
    return df


def load_espset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = data_dir or RAW_DIR / "ESPset"
    data_dir.mkdir(parents=True, exist_ok=True)
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No ESPset CSV found in {data_dir}.")
    df = pd.read_csv(csvs[0])
    df["asset_class"] = "espset_pump"
    df["domain_label"] = 0
    return df


def load_3w(
    data_dir: Optional[Path] = None,
    version: str = "2.0.0",
    max_instances: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Petrobras 3W Dataset (real offshore oil-well undesirable events).
    This is the primary real facility target domain used in the dissertation.
    """
    data_dir = data_dir or RAW_DIR / "3W"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = list(data_dir.rglob("*.parquet"))
    csv_files = list(data_dir.rglob("*.csv"))
    if not parquet_files and not csv_files:
        raise FileNotFoundError(
            f"3W data not found in {data_dir}.\n"
            "Download from https://github.com/petrobras/3W or "
            "https://doi.org/10.6084/m9.figshare.29205836.v1\n"
            "Extract into data/raw/3W/ and re-run."
        )
    frames = []
    files = parquet_files if parquet_files else csv_files
    if max_instances:
        files = files[:max_instances]
    for f in tqdm(files, desc="Loading 3W real facility instances"):
        try:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            df["unit"] = f.stem
            df["asset_class"] = "3w_offshore_well"
            df["domain_label"] = 1  # real facility target
            parent = f.parent.name
            if parent.isdigit() or parent.startswith("class"):
                df["event_label"] = parent
            frames.append(df)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    if not frames:
        raise RuntimeError("No 3W instances could be loaded.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames)} real 3W instances, total rows: {len(combined):,}")
    return combined


def list_available() -> Dict[str, bool]:
    _ensure_dirs()
    return {
        "CMAPSS": (RAW_DIR / "CMAPSS").exists() and any((RAW_DIR / "CMAPSS").glob("train_*.txt")),
        "AI4I": (RAW_DIR / "AI4I" / "ai4i2020.csv").exists(),
        "ESPset": any((RAW_DIR / "ESPset").glob("*.csv")),
        "3W_real_facility": any((RAW_DIR / "3W").rglob("*.parquet")) or any((RAW_DIR / "3W").rglob("*.csv")),
    }
