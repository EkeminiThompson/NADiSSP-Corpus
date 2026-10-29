#!/usr/bin/env python
"""Baselines + ablations x n seeds (Tables 3.4 / 4.3 / 4.4) -> results/ablation_results.json, results/tables.md."""
import argparse
import _boot  # noqa: F401
from src.config import add_common_args, load_config
from src.pipeline import run_ablation
from src.training.variants import VARIANTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--variants", nargs="*", default=None, choices=sorted(VARIANTS))
    ap.add_argument("--seeds", type=int, default=None, help="default: n_seeds from the config (10)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_ablation(load_config(a.config, a.overrides), a.variants, a.seeds, a.force)
    print("wrote results/ablation_results.json and results/tables.md")
