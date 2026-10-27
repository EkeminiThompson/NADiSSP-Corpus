#!/usr/bin/env python
"""
Train the full NADIP model (Transformer + GRL + optional contrastive).
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

PROCESSED = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def cosine_lambda(epoch: int, max_epochs: int, max_lambda: float = 1.0) -> float:
    p = epoch / max_epochs
    return max_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(PROCESSED / "combined_corpus.npz", allow_pickle=True)
    X = data["X"]
    rul = np.nan_to_num(data["rul"], nan=50.0)
    fail = np.nan_to_num(data["fail"], nan=0.0)
    domain = data["domain"]

    N, T, C = X.shape
    print(f"Corpus: N={N}, T={T}, C={C}")
    print(f"  source={(domain==0).sum()}, target={(domain==1).sum()}")

    X_t = torch.from_numpy(X).float()
    rul_t = torch.from_numpy(rul).float()
    fail_t = torch.from_numpy(fail).float()
    dom_t = torch.from_numpy(domain).float()

    ds = TensorDataset(X_t, rul_t, fail_t, dom_t)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = NADIP(n_channels=C, d_model=args.d_model, num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        lam = cosine_lambda(epoch, args.epochs, max_lambda=args.max_lambda)
        model.set_grl_lambda(lam)

        tr_loss = 0.0
        for xb, rb, fb, db in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            xb, rb, fb, db = xb.to(device), rb.to(device), fb.to(device), db.to(device)
            opt.zero_grad()
            out = model(xb, return_embedding=True)
            loss_rul = nn.functional.mse_loss(out["rul"], rb)
            loss_fail = nn.functional.binary_cross_entropy_with_logits(out["fail_logit"], fb)
            loss_dom = nn.functional.binary_cross_entropy_with_logits(out["domain_logit"], db)
            noise = 0.05 * torch.randn_like(xb)
            out2 = model(xb + noise, return_embedding=True)
            loss_con = nt_xent_loss(out["projection"], out2["projection"])
            loss = loss_rul + loss_fail + args.lambda_dom * loss_dom + args.lambda_con * loss_con
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        val_rul_mse = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, rb, fb, db in val_loader:
                xb, rb = xb.to(device), rb.to(device)
                out = model(xb, return_embedding=True)
                val_rul_mse += nn.functional.mse_loss(out["rul"], rb).item()
                n_val_batches += 1
        val_rul_mse /= max(1, n_val_batches)
        val_rmse = val_rul_mse ** 0.5

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss / max(1, len(train_loader)),
            "val_rul_rmse": val_rmse,
            "grl_lambda": lam,
        })
        print(f"Epoch {epoch+1:03d} | train_loss={history[-1]['train_loss']:.4f} | val_RUL_RMSE={val_rmse:.3f} | lambda_GRL={lam:.3f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "model_state": model.state_dict(),
                "n_channels": C,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "epoch": epoch + 1,
                "val_rmse": val_rmse,
            }, CKPT_DIR / "nadip_full.pt")
            print(f"  -> saved best checkpoint (RMSE={val_rmse:.3f})")

    with open(CKPT_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val RUL RMSE = {best_val:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--max_lambda", type=float, default=1.0)
    p.add_argument("--lambda_dom", type=float, default=0.5)
    p.add_argument("--lambda_con", type=float, default=0.1)
    args = p.parse_args()
    main(args)
