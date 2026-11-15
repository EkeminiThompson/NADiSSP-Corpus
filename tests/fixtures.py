"""SYNTHETIC test fixtures in the on-disk formats of the four datasets.

These files exist only so the pipeline can be exercised end-to-end without the real datasets.
They are random signals with a crude degradation trend. NOTHING produced from them is a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_cmapss(root: Path, rng: np.random.Generator, n_train=14, n_test=6) -> None:
    d = root / "CMAPSS"
    d.mkdir(parents=True, exist_ok=True)
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        def engines(n, truncate):
            rows, finals = [], []
            for u in range(1, n + 1):
                life = int(rng.integers(140, 230))
                keep = life - int(rng.integers(20, 60)) if truncate else life
                finals.append(life - keep)
                t = np.arange(1, keep + 1)
                for c in t:
                    deg = (c / life) ** 2
                    sensors = 500 + 40 * rng.normal(size=21) + 80 * deg * np.linspace(0.2, 1.0, 21)
                    rows.append([u, c, 0.1, 0.2, 100.0] + list(sensors))
            return pd.DataFrame(rows), np.array(finals)
        train, _ = engines(n_train, False)
        test, fin = engines(n_test, True)
        train.to_csv(d / f"train_{fd}.txt", sep=" ", header=False, index=False)
        test.to_csv(d / f"test_{fd}.txt", sep=" ", header=False, index=False)
        pd.DataFrame(fin).to_csv(d / f"RUL_{fd}.txt", header=False, index=False)


def make_ai4i(root: Path, rng: np.random.Generator, n=3000) -> None:
    d = root / "AI4I"
    d.mkdir(parents=True, exist_ok=True)
    modes = {m: (rng.random(n) < p).astype(int) for m, p in zip(["TWF", "HDF", "PWF", "OSF", "RNF"], [0.01, 0.01, 0.01, 0.01, 0.005])}
    fail = np.clip(sum(modes.values()), 0, 1)
    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1), "Product ID": [f"M{i}" for i in range(n)], "Type": "M",
        "Air temperature [K]": 300 + rng.normal(size=n), "Process temperature [K]": 310 + rng.normal(size=n),
        "Rotational speed [rpm]": 1500 + 100 * rng.normal(size=n), "Torque [Nm]": 40 + 5 * rng.normal(size=n),
        "Tool wear [min]": rng.integers(0, 250, size=n), "Machine failure": fail, **modes,
    })
    df.to_csv(d / "ai4i2020.csv", index=False)


def make_w3(root: Path, rng: np.random.Generator, n_real=30, n_sim=10, length=420) -> None:
    base = root / "3W" / "dataset"
    sensors = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP", "P-JUS-CKGL", "QGL"]
    k = 0
    for prov, n in (("WELL", n_real), ("SIMULATED", n_sim)):
        for i in range(n):
            event = int(rng.choice([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]))
            (base / str(event)).mkdir(parents=True, exist_ok=True)
            data = {s: 100 + 5 * rng.normal(size=length) for s in sensors}
            cls = np.zeros(length, dtype=int)
            if event:
                cls[length // 2:] = event
                for s in sensors:
                    data[s][length // 2:] += 15
            df = pd.DataFrame(data)
            df["class"] = cls
            df["state"] = cls
            df.to_parquet(base / str(event) / f"{prov}-{k:05d}_20170101000000.parquet")
            k += 1
    # one hand-drawn file that the loader must skip
    (base / "0").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({**{s: rng.normal(size=length) for s in sensors}, "class": 0}).to_parquet(base / "0" / "DRAWN_00001.parquet")


def make_espset(root: Path, rng: np.random.Generator, n_units=8, length=500) -> None:
    d = root / "ESPset"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for u in range(n_units):
        n = length
        lab = np.array(["normal"] * n, dtype=object)
        if u % 2 == 0:
            lab[int(n * 0.7):] = rng.choice(["bearing", "cavitation"])
        f = pd.DataFrame({
            "unit": f"esp{u}", "vibration_x": rng.normal(size=n), "vibration_y": rng.normal(size=n),
            "vibration_z": rng.normal(size=n), "current": 50 + rng.normal(size=n),
            "discharge_pressure": 200 + rng.normal(size=n), "intake_pressure": 50 + rng.normal(size=n),
            "motor_temp": 80 + rng.normal(size=n), "frequency": 50 + 0.1 * rng.normal(size=n), "fault": lab,
        })
        frames.append(f)
    pd.concat(frames).to_csv(d / "espset_sample.csv", index=False)


def make_all(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    make_cmapss(root, rng)
    make_ai4i(root, rng)
    make_w3(root, rng)
    make_espset(root, rng)
    return root
