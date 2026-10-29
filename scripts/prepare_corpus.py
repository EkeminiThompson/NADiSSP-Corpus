#!/usr/bin/env python
"""Stage 1-3 of data collection (§3.4): load the four datasets, preprocess, stratify, augment, split, save."""
import argparse
import _boot  # noqa: F401
from src.config import add_common_args, load_config
from src.pipeline import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    meta = prepare(cfg, a.raw_dir)["meta"]
    print(f"corpus: {meta['n_total']} windows  (source {meta['n_source']}, target {meta['n_target']})")
    for k, v in meta["datasets"].items():
        print(f"  {k:7s} source {v['n_source']:6d}/{v['requested_source']:<6d} target {v['n_target']:6d}/{v['requested_target']:<6d}")
    for k, why in meta["missing_datasets"].items():
        print(f"  MISSING {k}: {why.splitlines()[0]}")
    for s in meta["shortfalls"]:
        print(f"  shortfall: {s}")
    if meta["prevalence_outside_8_20pct"]:
        print(f"  note: train failure prevalence outside the 8-20% band for {meta['prevalence_outside_8_20pct']}")
