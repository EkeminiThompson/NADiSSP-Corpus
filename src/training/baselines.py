"""Pre-intervention baselines of Table 3.4 / 4.3 trained on SOURCE data only: XGBoost and an unaligned LSTM.

(The source-only Transformer is the NADIP backbone with every adaptation switch off -- see variants.py.)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.features import window_features
from .data import Bundle
from .variants import Variant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XGBBaseline:
    def __init__(self, cfg: Dict, seed: int):
        self.cfg, self.seed, self.reg, self.clf = cfg, seed, None, None
        self.cap = float(cfg["data"]["rul_cap"])

    def fit(self, b: Bundle, rows: np.ndarray, rows_val: np.ndarray) -> "XGBBaseline":
        import xgboost as xgb
        c = self.cfg["training"]["stage5_xgboost"]
        F_tr, _ = window_features(b.X[rows].numpy(), b.dataset[rows])
        kw = dict(n_estimators=c["n_estimators"], max_depth=c["max_depth"], random_state=self.seed, n_jobs=2,
                  tree_method="hist", learning_rate=0.1)
        has = b.has_rul[rows].numpy()
        if has.sum() > 10:
            self.reg = xgb.XGBRegressor(**kw).fit(F_tr[has], b.rul[rows].numpy()[has], verbose=False)
        y = b.fail[rows].numpy()
        if 0 < y.sum() < len(y):
            self.clf = xgb.XGBClassifier(**kw).fit(F_tr, y, verbose=False)
        return self

    def predict(self, X: torch.Tensor, dataset_ids: np.ndarray) -> Dict[str, np.ndarray]:
        f, _ = window_features(X.numpy(), dataset_ids)
        rul = np.clip(self.reg.predict(f), 0, self.cap) if self.reg is not None else np.full(len(f), self.cap / 2)
        p = self.clf.predict_proba(f)[:, 1] if self.clf is not None else np.zeros(len(f))
        return {"rul": rul.astype(np.float32), "fail_prob": p.astype(np.float32)}


class LSTMNet(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 64, layers: int = 2, cap: float = 130.0):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, layers, batch_first=True)
        self.rul = nn.Linear(hidden, 1)
        self.fail = nn.Linear(hidden, 1)
        self.cap = cap

    def forward(self, x):
        h, _ = self.lstm(x)
        h = h[:, -1]
        return self.rul(h).squeeze(-1) * self.cap, self.fail(h).squeeze(-1)


class LSTMBaseline:
    def __init__(self, cfg: Dict, n_channels: int, seed: int):
        self.cfg, self.seed = cfg, seed
        self.cap = float(cfg["data"]["rul_cap"])
        self.net = LSTMNet(n_channels, cap=self.cap).to(DEVICE)

    def fit(self, b: Bundle, rows: np.ndarray, epochs: int) -> "LSTMBaseline":
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        bs = self.cfg["training"]["batch_size"]
        for _ in range(epochs):
            self.net.train()
            order = rng.permutation(rows)
            for i in range(0, len(order), bs):
                r = torch.from_numpy(order[i:i + bs])
                if len(r) < 2:
                    continue
                rul_p, fail_p = self.net(b.X[r].to(DEVICE))
                m = b.has_rul[r].to(DEVICE)
                l = nn.functional.binary_cross_entropy_with_logits(fail_p, b.fail[r].to(DEVICE))
                if m.any():
                    l = l + (((rul_p - b.rul[r].to(DEVICE)) / self.cap) ** 2)[m].mean()
                opt.zero_grad(); l.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, dataset_ids: np.ndarray) -> Dict[str, np.ndarray]:
        self.net.eval()
        r, f = [], []
        for i in range(0, len(X), 256):
            a, c = self.net(X[i:i + 256].to(DEVICE))
            r.append(a.cpu()); f.append(torch.sigmoid(c).cpu())
        return {"rul": torch.cat(r).clamp(0, self.cap).numpy(), "fail_prob": torch.cat(f).numpy()}


def train_baseline(cfg: Dict, bundle: Bundle, variant: Variant, seed: int, log=print):
    from .runner import Trained, seed_everything
    seed_everything(seed)
    rows = bundle.idx("train", 0, list(variant.datasets) if variant.datasets else None)
    if len(rows) == 0:
        raise RuntimeError(f"{variant.name}: no source training rows")
    t = Trained(variant, seed, kind=variant.kind)
    if variant.kind == "xgboost_source_only":
        t.baseline = XGBBaseline(cfg, seed).fit(bundle, rows, bundle.idx("val", 0))
    else:
        epochs = int(cfg["training"]["stage3_joint"]["epochs"])
        t.baseline = LSTMBaseline(cfg, bundle.C, seed).fit(bundle, rows, epochs)
    t.info = {"n_source_train": int(len(rows))}
    log(f"[{variant.name}|seed {seed}] trained")
    return t
