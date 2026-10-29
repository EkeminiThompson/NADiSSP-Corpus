#!/usr/bin/env python
"""Train one variant (default: the full NADIP) for one or more seeds, with stage checkpointing/resume."""
import argparse
import _boot  # noqa: F401
from src.config import add_common_args, load_config
from src.pipeline import get_bundle, run_one, seeds_for
from src.training.variants import VARIANTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--variant", default="full_nadip", choices=sorted(VARIANTS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    bundle, meta = get_bundle(cfg)
    for s in seeds_for(cfg, a.seeds):
        rec = run_one(cfg, bundle, meta, a.variant, s, a.force)
        o = rec.get("test", {}).get("overall", {})
        print(f"{a.variant} seed {s}: {rec['status']}  test AUPRC={o.get('auprc')}  RMSE={o.get('rul_rmse')}  F1={o.get('f1')}")
