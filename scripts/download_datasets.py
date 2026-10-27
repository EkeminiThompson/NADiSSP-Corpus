#!/usr/bin/env python
"""
Helper to download / locate the public datasets used by NADIP.

CMAPSS, AI4I and ESPset require manual download (or known public mirrors).
3W (real facility) can be cloned from the official Petrobras repository.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("NADIP Dataset Acquisition Guide")
    print("=" * 60)

    print("""
1. NASA CMAPSS (source domain)
   - Official: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
   - Place train_*.txt, test_*.txt, RUL_*.txt under data/raw/CMAPSS/

2. UCI AI4I 2020 (source domain)
   - https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
   - Save as: data/raw/AI4I/ai4i2020.csv

3. ESPset (source domain)
   - https://data.mendeley.com/datasets/m268jsw339/1
   - Place CSV(s) under: data/raw/ESPset/

4. Petrobras 3W Dataset 2.0.0  ← PRIMARY REAL FACILITY TARGET
   Official repository: https://github.com/petrobras/3W
""")

    three_w = RAW / "3W"
    if not three_w.exists():
        print("   Attempting to clone 3W repository …")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/petrobras/3W.git", str(three_w)],
                check=True,
            )
            print("   3W cloned successfully.")
        except Exception as e:
            print(f"   Automatic clone failed ({e}).")
            print("   Please clone manually: cd data/raw && git clone https://github.com/petrobras/3W.git")
    else:
        print(f"   3W directory already present: {three_w}")

    print("""
After all datasets are in place, run:

    python scripts/prepare_corpus.py
    python scripts/train_nadip.py
""")


if __name__ == "__main__":
    main()
