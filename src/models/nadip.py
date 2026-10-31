"""NADIP architecture (§3.10, Table 3.15b).

    x (B,T,C) -> linear embed + positional encoding -> N-layer Transformer encoder -> mean-pool -> z (B,d)
    z -> RUL regression head (C) and fault head
    z -> GRL -> 3-layer MLP domain discriminator (D)
    z -> 3-layer MLP projector (SimCLR)
    z -> Weibull-modulated proportional-hazards head: S(t|z) = exp(-(t/eta)^beta * exp(w^T z))

Not claimed here: Kinetics-400 initialisation. `load_compatible` copies tensors from a checkpoint
*only where name and shape match* and reports exactly what it loaded; no public video-transformer
checkpoint matches d_model=192 / 8 heads / 12 layers, so by default the backbone is randomly initialised.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def ganin_lambda(progress: float, max_lambda: float = 1.0) -> float:
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe.unsqueeze(0)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_channels: int, d_model: int = 192, nhead: int = 8, num_layers: int = 12,
                 dim_feedforward: int = 768, dropout: float = 0.1, positional: str = "sinusoidal",
                 max_len: int = 256):
        super().__init__()
        self.input = nn.Linear(n_channels, d_model)
        self.positional = positional
        if positional == "sinusoidal":
            self.register_buffer("pe", sinusoidal_pe(max_len, d_model), persistent=False)
            self.pos_emb = None
        else:
            self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
            self.pe = None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        T = h.size(1)
        if self.pe is not None:
            h = h + self.pe[:, :T]
        else:
            h = h + self.pos_emb[:, :T]
        h = self.encoder(h)
        return self.norm(h.mean(dim=1))

    def set_frozen_lower(self, fraction: float, frozen: bool) -> None:
        n = max(1, int(self.num_layers * fraction))
        for i, layer in enumerate(self.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = not (frozen and i < n)


def load_compatible(module: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    cur = module.state_dict()
    load, skip = {}, []
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            load[k] = v
        else:
            skip.append(k)
    missing, unexpected = module.load_state_dict(load, strict=False)
    return {"loaded": list(load.keys()), "skipped": skip, "missing": list(missing), "unexpected": list(unexpected)}


def mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = d_in
    for i in range(max(1, n_layers - 1)):
        layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(0.1)]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class WeibullPHHead(nn.Module):
    def __init__(self, d_model: int, eta_init: float = 100.0):
        super().__init__()
        self.w = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.w.weight, std=0.01)
        self.log_eta = nn.Parameter(torch.tensor(math.log(eta_init)))
        self.raw_beta = nn.Parameter(torch.tensor(0.5))

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.raw_beta)

    @property
    def eta(self) -> torch.Tensor:
        return torch.exp(self.log_eta)

    def risk(self, z: torch.Tensor) -> torch.Tensor:
        return self.w(z).squeeze(-1).clamp(-8.0, 8.0)

    def log_survival(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return -((t / self.eta) ** self.beta) * torch.exp(self.risk(z))

    def nll(self, t: torch.Tensor, event: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t = t.clamp(min=1.0)
        r = self.risk(z)
        log_s = -((t / self.eta) ** self.beta) * torch.exp(r)
        log_h = torch.log(self.beta) - self.beta * self.log_eta + (self.beta - 1.0) * torch.log(t) + r
        ll = log_s + event * log_h
        return -ll.mean()

    def cox_partial_nll(self, t: torch.Tensor, event: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r = self.risk(z)
        order = torch.argsort(t, descending=True)
        r_s, e_s = r[order], event[order]
        log_risk_set = torch.logcumsumexp(r_s, dim=0)
        n_ev = e_s.sum().clamp(min=1.0)
        return -((r_s - log_risk_set) * e_s).sum() / n_ev

    @torch.no_grad()
    def rul_median(self, z: torch.Tensor) -> torch.Tensor:
        return self.eta * (math.log(2.0) / torch.exp(self.risk(z))) ** (1.0 / self.beta)


class NADIP(nn.Module):
    def __init__(self, n_channels: int, mcfg: Dict, rul_scale: float = 130.0):
        super().__init__()
        d = int(mcfg["d_model"])
        self.backbone = TimeSeriesTransformer(
            n_channels, d_model=d, nhead=int(mcfg["nhead"]), num_layers=int(mcfg["num_layers"]),
            dim_feedforward=int(mcfg["dim_feedforward"]), dropout=float(mcfg.get("dropout", 0.1)),
            positional=str(mcfg.get("positional", "sinusoidal")),
        )
        hh = int(mcfg.get("head_hidden", 64))
        self.rul_head = mlp(d, hh, 1, 2)
        self.fail_head = mlp(d, hh, 1, 2)
        self.domain_disc = mlp(d, hh, 1, int(mcfg.get("discriminator_layers", 3)))
        self.projector = mlp(d, d, 64, int(mcfg.get("projector_layers", 3)))
        self.grl = GradientReversal(1.0)
        self.weibull = WeibullPHHead(d, eta_init=rul_scale * 0.75)
        self.rul_scale = rul_scale

    def set_grl_lambda(self, lam: float) -> None:
        self.grl.lambda_ = lam

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "z": z,
            "rul": self.rul_head(z).squeeze(-1) * self.rul_scale,
            "fail_logit": self.fail_head(z).squeeze(-1),
            "domain_logit": self.domain_disc(self.grl(z)).squeeze(-1),
            "proj": self.projector(z),
        }

    def param_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        disc = list(self.domain_disc.parameters())
        ids = {id(p) for p in disc}
        return [p for p in self.parameters() if id(p) not in ids], disc


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over rows where mask is True; 0 when no row is labelled.

    Non-masked targets may be NaN (no RUL); zero them before the difference so NaN*0 never appears.
    """
    m = mask.float()
    n = m.sum()
    if n.item() == 0:
        return pred.sum() * 0.0
    tgt = torch.where(mask.bool(), target, torch.zeros_like(target))
    pr = torch.where(mask.bool(), pred, torch.zeros_like(pred))
    return (((pr - tgt) ** 2) * m).sum() / n


def masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    n = m.sum()
    if n.item() == 0:
        return logit.sum() * 0.0
    tgt = torch.where(mask.bool(), target, torch.zeros_like(target))
    return (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * m).sum() / n
