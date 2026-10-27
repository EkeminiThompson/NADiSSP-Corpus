#!/usr/bin/env python
"""
Generate the key Chapter-4 figures from saved results / corpus.

Produces:
  - fig4_1 style t-SNE (domain separation pre/post conceptually)
  - fig4_2 ablation bar chart
  - fig4_6 Monte-Carlo TCO distributions
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = RESULTS / "tables"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


def fig_ablation():
    path = TABLE_DIR / "ablation_results.json"
    if not path.exists():
        print("  [skip] ablation_results.json not found - run scripts/run_ablation.py first")
        return
    with open(path) as f:
        abl = json.load(f)

    names = list(abl.keys())
    auprcs = [abl[n].get("auprc", 0) or 0 for n in names]
    rmses = [abl[n].get("rul_rmse", 0) for n in names]

    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Ablation Study - Marginal Contribution of NADIP Components", fontsize=13)

    colors = ["#2ecc71", "#f39c12", "#e67e22", "#3498db", "#95a5a6"]
    x = np.arange(len(names))

    axes[0].bar(x, auprcs, color=colors[:len(names)], edgecolor="k", alpha=0.85)
    axes[0].axhline(0.85, color="red", ls="--", lw=1, label="Target 0.85")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("AUPRC")
    axes[0].legend()

    axes[1].bar(x, rmses, color=colors[:len(names)], edgecolor="k", alpha=0.85)
    axes[1].axhline(13.5, color="red", ls="--", lw=1, label="Target 13.5")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("RUL RMSE (cycles)")
    axes[1].legend()

    plt.tight_layout()
    out = FIG_DIR / "fig_ablation.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig_monte_carlo():
    path = TABLE_DIR / "monte_carlo_tco_runs.csv"
    if not path.exists():
        print("  [skip] monte_carlo_tco_runs.csv not found - run economic module first")
        return
    import pandas as pd
    df = pd.read_csv(path)

    style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["annual_tco_reactive"] / 1e6, bins=60, alpha=0.55, color="#e74c3c",
            label="Reactive baseline", density=True)
    ax.hist(df["annual_tco_nadip"] / 1e6, bins=60, alpha=0.55, color="#27ae60",
            label="NADIP PdM", density=True)
    ax.set_xlabel("Annual TCO (USD million)")
    ax.set_ylabel("Density")
    ax.set_title("Monte-Carlo TCO Distributions - Reactive vs NADIP\n(10 000 runs, Nigerian refinery parameters)")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_monte_carlo_tco.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["npv_5yr"] / 1e6, bins=60, color="#2980b9", alpha=0.75, density=True)
    ax.axvline(df["npv_5yr"].mean() / 1e6, color="black", ls="--", label=f"Mean = ${df['npv_5yr'].mean()/1e6:.0f} M")
    ax.set_xlabel("5-year NPV of NADIP (USD million)")
    ax.set_ylabel("Density")
    ax.set_title("5-Year NPV Distribution (NADIP vs Reactive)")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_npv_distribution.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def fig_domain_shift_placeholder():
    """Simple illustrative pre/post MMD bar (values from dissertation Table 4.2)."""
    style()
    metrics = ["Composite MMD", "Wasserstein-1", "Domain Acc."]
    pre = [0.82, 3.07, 0.914]
    post = [0.26, 1.24, 0.512]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, pre, w, label="Pre-adaptation", color="#e74c3c", edgecolor="k")
    ax.bar(x + w/2, post, w, label="Post-GRL", color="#27ae60", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_title("Domain Shift Reduction (Objective 1)\nSource benchmarks -> Real 3W + Nigerian target")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig_domain_shift.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def main():
    print("Generating Chapter-4 style figures ...")
    fig_domain_shift_placeholder()
    fig_ablation()
    fig_monte_carlo()
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
