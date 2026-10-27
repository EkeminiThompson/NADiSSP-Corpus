#!/usr/bin/env python
"""
Ablation study (Table 4.4 / Figure 4.2).

Trains several reduced configurations and records AUPRC + RUL RMSE
so the marginal contribution of each NADIP component can be quantified.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.nadip import NADIP, nt_xent_loss
from src.evaluation.metrics import evaluate_predictions

PROCESSED = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
RESULTS = ROOT / "results" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "full":          {"use_grl": True,  "use_con": True,  "use_fail": True},
    "no_grl":        {"use_grl": False, "use_con": True,  "use_fail": True},
    "no_simclr":     {"use_grl": True,  "use_con": False, "use_fail": True},
    "no_fail_head":  {"use_grl": True,  "use_con": True,  "use_fail": False},
    "baseline":      {"use_grl": False, "use_con": False, "use_fail": True},
}


def train_one(name: str, cfg: dict, X, rul, fail, domain, args, device):
    print(f"\n=== Ablation: {name} ===")
    N, T, C = X.shape
    X_t = torch.from_numpy(X).float()
    rul_t = torch.from_numpy(rul).float()
    fail_t = torch.from_numpy(fail).float()
    dom_t = torch.from_numpy(domain).float()

    ds = TensorDataset(X_t, rul_t, fail_t, dom_t)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = NADIP(n_channels=C, d_model=args.d_model, num_layers=2).to(device)
    if not cfg["use_grl"]:
        model.set_grl_lambda(0.0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        if cfg["use_grl"]:
            p = epoch / args.epochs
            lam = 1.0 * (2 / (1 + np.exp(-10 * p)) - 1)
            model.set_grl_lambda(lam)
        for xb, rb, fb, db in train_loader:
            xb, rb, fb, db = [t.to(device) for t in (xb, rb, fb, db)]
            opt.zero_grad()
            out = model(xb, return_embedding=True)
            loss = nn.functional.mse_loss(out["rul"], rb)
            if cfg["use_fail"]:
                loss = loss + nn.functional.binary_cross_entropy_with_logits(out["fail_logit"], fb)
            if cfg["use_grl"]:
                loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(out["domain_logit"], db)
            if cfg["use_con"]:
                noise = 0.05 * torch.randn_like(xb)
                out2 = model(xb + noise, return_embedding=True)
                loss = loss + 0.1 * nt_xent_loss(out["projection"], out2["projection"])
            loss.backward()
            opt.step()

    model.eval()
    preds_rul, preds_fail, trues_rul, trues_fail = [], [], [], []
    with torch.no_grad():
        for xb, rb, fb, db in val_loader:
            xb = xb.to(device)
            out = model(xb, return_embedding=True)
            preds_rul.append(out["rul"].cpu().numpy())
            preds_fail.append(torch.sigmoid(out["fail_logit"]).cpu().numpy())
            trues_rul.append(rb.numpy())
            trues_fail.append(fb.numpy())

    y_pred_rul = np.concatenate(preds_rul)
    y_pred_fail = np.concatenate(preds_fail)
    y_true_rul = np.concatenate(trues_rul)
    y_true_fail = np.concatenate(trues_fail)

    metrics = evaluate_predictions(y_true_rul, y_pred_rul, y_true_fail, y_pred_fail)
    print(f"  {name}: RUL RMSE={metrics['rul_rmse']:.3f}  AUPRC={metrics.get('auprc', float('nan')):.3f}")
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(PROCESSED / "combined_corpus.npz", allow_pickle=True)
    X = data["X"]
    rul = np.nan_to_num(data["rul"], nan=50.0)
    fail = np.nan_to_num(data["fail"], nan=0.0)
    domain = data["domain"]

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = train_one(name, cfg, X, rul, fail, domain, args, device)

    out = RESULTS / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=64)
    args = p.parse_args()
    main(args)
