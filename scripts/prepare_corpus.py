#!/usr/bin/env python
"""
Prepare the full experimental corpus:
  - Load public benchmarks
  - Load real Petrobras 3W facility data
  - Generate Nigerian-augmented variants
  - Save combined_corpus.npz
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_combined_corpus


if __name__ == "__main__":
    print("=" * 60)
    print("NADIP – Corpus Preparation")
    print("Primary real facility target: Petrobras 3W")
    print("=" * 60)
    corpus = build_combined_corpus(
        window=50,
        stride=10,
        apply_nigerian_aug=True,
        max_3w_instances=300,
        seed=42,
    )
    print("\nDone. Corpus ready for training.")
