#!/usr/bin/env python
"""Run the Monte-Carlo TCO / NPV analysis (Chapter 4)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.economic import run_monte_carlo_tco

if __name__ == "__main__":
    run_monte_carlo_tco(n_runs=10_000, seed=42)
