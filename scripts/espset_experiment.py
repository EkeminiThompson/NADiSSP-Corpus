#!/usr/bin/env python
"""Separate ESPset experiment on the published spectra (spectrum.csv + features.csv in data/raw/ESPset/).

    python scripts/espset_experiment.py                  # espset.experiment.n_seeds repeated group-wise splits
    python scripts/espset_experiment.py --seeds 3

Writes results/espset/{seed*.json, summary.json, tables.md}. Reported separately from the time-series study.
"""
import argparse

import _boot  # noqa: F401
from src.config import add_common_args, load_config, resolve_path
from src.espset_experiment import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    raw = a.raw_dir or resolve_path(cfg, "raw_dir")
    res = run(cfg, raw, resolve_path(cfg, "results_dir") / "espset", a.seeds)
    print((resolve_path(cfg, "results_dir") / "espset" / "tables.md").read_text())
