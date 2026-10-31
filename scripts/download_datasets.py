#!/usr/bin/env python
"""Dataset acquisition (Table 3.12b). Fetches what is publicly scriptable and tells you where to put the rest.

  CMAPSS   NASA PCoE repository      -> data/raw/CMAPSS/{train,test,RUL}_FD00x.txt
  AI4I     UCI repository (id 601)   -> data/raw/AI4I/ai4i2020.csv
  3W       github.com/petrobras/3W   -> data/raw/3W/   (sparse clone: dataset/ only)
  ESPset   NOT USED: published only as spectra (spectrum.csv), no time axis; see README

Use --verify to print SHA-256 sums of what is present so a run can be tied to exact files.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import _boot  # noqa: F401
from src.config import ROOT

RAW = ROOT / "data" / "raw"


def clone_3w():
    dst = RAW / "3W"
    if dst.exists():
        print("3W present:", dst); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "https://github.com/petrobras/3W.git", str(dst)])
    subprocess.check_call(["git", "-C", str(dst), "sparse-checkout", "set", "dataset"])


def verify():
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.suffix in (".txt", ".csv", ".parquet", ".h5"):
            print(hashlib.sha256(p.read_bytes()).hexdigest(), p.relative_to(RAW))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        clone_3w()
        print(__doc__)
        for d, f in (("CMAPSS", "train_FD001.txt"), ("AI4I", "ai4i2020.csv")):
            ok = (RAW / d / f).exists() if f else any((RAW / d).glob("*.csv")) if (RAW / d).exists() else False
            print(f"  {d:7s}: {'present' if ok else 'MISSING - download manually (see above)'}")
