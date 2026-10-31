#!/usr/bin/env python
"""Stage 1-3 of data collection: load datasets, preprocess, stratify, augment, split, save."""
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
    print(f"corpus: {meta.get('n_total')} windows  (source {meta.get('n_source')}, target {meta.get('n_target')})")
    for k, v in (meta.get("datasets") or {}).items():
        print(f"  {k:7s} source {v.get('n_source')} target {v.get('n_target')}")
    missing = meta.get("missing") or meta.get("missing_datasets") or {}
    for k, why in missing.items():
        msg = why if isinstance(why, str) else str(why)
        print(f"  MISSING {k}: {msg.splitlines()[0]}")
