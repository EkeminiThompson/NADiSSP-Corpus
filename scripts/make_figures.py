#!/usr/bin/env python
"""Figures 4.1, 4.2, 4.6 and the packet-loss curve, drawn ONLY from files written by the other scripts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _boot  # noqa: F401
from src.config import add_common_args, load_config, resolve_path


def main(cfg):
    res = resolve_path(cfg, "results_dir")
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    pre, post = res / "figures_data" / "tsne_pre.npz", res / "figures_data" / "tsne_post.npz"
    if pre.exists() and post.exists():
        ev = json.load(open(res / "evaluation.json"))["embedding_shift"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        for a, p, key, title in ((ax[0], pre, "pre_source_only", "Source-only transformer"), (ax[1], post, "post_adapted", "NADIP (adapted)")):
            d = np.load(p)
            for dom, lab, mk in ((0, "source", "o"), (1, "Nigerian target", "x")):
                m = d["domain"] == dom
                a.scatter(d["emb"][m, 0], d["emb"][m, 1], c=d["rul"][m], cmap="viridis", s=10, marker=mk, label=lab, alpha=0.8)
            a.set_title(f"{title}\nsilhouette = {ev[key].get('tsne_silhouette', float('nan')):.2f}"); a.legend()
        fig.tight_layout(); fig.savefig(figs / "fig4_1_tsne.png", dpi=160); plt.close(fig)

    abl = res / "ablation_results.json"
    if abl.exists():
        agg = json.load(open(abl))["variants"]
        names = [n for n, v in agg.items() if v.get("status") == "ok"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for a, m, lab in ((ax[0], "auprc", "AUPRC"), (ax[1], "rul_rmse", "RUL RMSE (cycles)")):
            mean = [agg[n]["overall"][m]["mean"] for n in names]
            err = [agg[n]["overall"][m]["mean"] - agg[n]["overall"][m]["ci95_low"] if agg[n]["n_seeds"] > 1 else 0 for n in names]
            a.bar(range(len(names)), mean, yerr=err, capsize=3); a.set_xticks(range(len(names)))
            a.set_xticklabels(names, rotation=35, ha="right"); a.set_ylabel(lab)
            a.axhline(cfg["evaluation"]["targets"]["auprc" if m == "auprc" else "rul_rmse"], color="red", ls="--", lw=1, label="document target")
            a.legend()
        fig.suptitle("Baselines and ablations (test split, mean with 95% CI over seeds)")
        fig.tight_layout(); fig.savefig(figs / "fig4_2_ablation.png", dpi=160); plt.close(fig)

    runs = res / "economics_runs.npz"
    if runs.exists():
        d = np.load(runs)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(d["tco_reactive"] / 1e6, 60, alpha=0.55, label="reactive", density=True)
        ax[0].hist(d["tco_nadip"] / 1e6, 60, alpha=0.55, label="NADIP", density=True)
        ax[0].set_xlabel("annual TCO (USD M)"); ax[0].legend()
        ax[1].hist(d["npv"] / 1e6, 60, density=True); ax[1].axvline(d["npv"].mean() / 1e6, color="k", ls="--")
        ax[1].set_xlabel("5-year NPV (USD M)")
        fig.tight_layout(); fig.savefig(figs / "fig4_6_monte_carlo.png", dpi=160); plt.close(fig)

    dep = res / "deployment.json"
    if dep.exists():
        pl = json.load(open(dep))["packet_loss"]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r["packet_loss"] * 100 for r in pl], [r["auprc"] for r in pl], "o-")
        ax.set_xlabel("packet loss (%)"); ax.set_ylabel("AUPRC")
        fig.tight_layout(); fig.savefig(figs / "fig_packet_loss.png", dpi=160); plt.close(fig)
    print("figures ->", figs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    a = ap.parse_args()
    main(load_config(a.config, a.overrides))
