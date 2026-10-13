#!/usr/bin/env python3
"""
Download helpers for the two remaining public datasets used by NADiSSP.

3W  (Petrobras)  : https://github.com/petrobras/3W
ESPset           : https://github.com/NINFA-UFES/ESPset
                   + Mendeley / Google Drive links for the large spectrum files

Run after cloning / downloading; then re-run generate_nadissp_corpus.py
(or extend it) to incorporate the new sequences.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def download_3w():
    target = RAW / "3W"
    if target.exists():
        print("3W already present")
        return
    print("Cloning Petrobras 3W (sparse, this may take a few minutes) ...")
    cmd = [
        "git", "clone", "--depth", "1",
        "--filter=blob:none", "--sparse",
        "https://github.com/petrobras/3W.git", str(target)
    ]
    subprocess.check_call(cmd)
    # checkout only the dataset directory
    subprocess.check_call(["git", "-C", str(target), "sparse-checkout", "set", "dataset"])
    print("3W dataset directory ready at", target / "dataset")


def download_espset():
    print("""
ESPset data files are hosted on Google Drive / Mendeley (≈110 MB spectrum.zip).
Manual steps:
  1. Visit https://github.com/NINFA-UFES/ESPset
  2. Follow the link to the Drive folder or
     https://doi.org/10.17632/m268jsw339
  3. Download spectrum.zip and features.zip
  4. Place them under raw/ESPset/
""")
    (RAW / "ESPset").mkdir(exist_ok=True)


if __name__ == "__main__":
    download_3w()
    download_espset()
    print("Done. Re-run generate_nadissp_corpus.py after data is in place.")
