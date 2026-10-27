#!/usr/bin/env python
"""
Quick evaluation of a trained NADIP checkpoint on the held-out portion
of the combined corpus (including real 3W facility windows).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.nadip import NADIP

PROCESSED = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    data = np.load(PROCESSED / "combined_corpus.npz", allow_pickle=True)
    X = data["X"]
    rul = np.nan_to_num(data["rul"], nan=50.0)
    fail = np.nan_to_num(data["fail"], nan=0.0)
    domain = data["domain"]
    asset = data["asset"]

    model = NADIP(
        n_channels=ckpt["n_channels"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mask = domain == 1
    if mask.sum() == 0:
        mask = np.ones(len(domain), dtype=bool)

    X_t = torch.from_numpy(X[mask]).float().to(device)
    with torch.no_grad():
        out = model(X_t, return_embedding=True)
        pred_rul = out["rul"].cpu().numpy()
        pred_fail = torch.sigmoid(out["fail_logit"]).cpu().numpy()

    true_rul = rul[mask]
    true_fail = fail[mask]

    rmse = float(np.sqrt(mean_squared_error(true_rul, pred_rul)))
    if true_fail.sum() > 0 and true_fail.min() != true_fail.max():
        auprc = float(average_precision_score(true_fail, pred_fail))
    else:
        auprc = float("nan")

    asset_masked = asset[mask]
    per_asset = {}
    for a in np.unique(asset_masked):
        m = asset_masked == a
        if m.sum() < 5:
            continue
        r = float(np.sqrt(mean_squared_error(true_rul[m], pred_rul[m])))
        per_asset[str(a)] = {"n": int(m.sum()), "rul_rmse": r}

    results = {
        "checkpoint": str(args.checkpoint),
        "n_target_windows": int(mask.sum()),
        "rul_rmse": rmse,
        "auprc": auprc,
        "per_asset": per_asset,
        "note": "Target domain includes real Petrobras 3W facility data when present.",
    }

    out_path = RESULTS / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 50)
    print("NADIP Evaluation Summary")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "nadip_full.pt"))
    args = p.parse_args()
    main(args)
