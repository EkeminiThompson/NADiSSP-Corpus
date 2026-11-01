#!/usr/bin/env python
"""Whole pipeline: corpus -> ablation (all variants x seeds) -> evaluation -> economics -> deployment -> figures.

    python scripts/run_all.py                              # full protocol (hours on CPU; needs the datasets)
    python scripts/run_all.py --config configs/smoke.yaml --raw-dir <synthetic fixtures>   # plumbing check only
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    common = (["--config", a.config] if a.config else []) + [x for o in a.overrides for x in ("--set", o)]
    steps = [("prepare_corpus.py", common + (["--raw-dir", a.raw_dir] if a.raw_dir else [])),
             ("run_ablation.py", common + (["--seeds", str(a.seeds)] if a.seeds else [])),
             ("evaluate.py", common), ("run_economics.py", common), ("deployment_sim.py", common), ("make_figures.py", common)]
    from src.config import load_config, resolve_path
    from src.data.espset_spectral import has_published_files
    raw = a.raw_dir or resolve_path(load_config(a.config, a.overrides), "raw_dir")
    if has_published_files(raw):                     # separate frequency-domain experiment, only if the files exist
        steps.insert(1, ("espset_experiment.py", common + (["--raw-dir", a.raw_dir] if a.raw_dir else [])
                         + (["--seeds", str(a.seeds)] if a.seeds else [])))
    for script, args in steps:
        print(f"\n########## {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script), *args], check=True)
