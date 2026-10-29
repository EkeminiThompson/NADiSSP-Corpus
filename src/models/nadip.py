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


# ------------------------------------------------------------------------------------- GRL
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
    """Eq. 3: lambda_d(p) = 2 / (1 + exp(-10 p)) - 1, scaled by max_lambda."""
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


# ------------------------------------------------------------------------------- backbone
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
        self.input_proj = nn.Linear(n_channels, d_model)
        if positional == "learnable":
            self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            self.register_buffer("pos", sinusoidal_pe(max_len, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([_clone_layer(layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B,T,C) -> (B,d)
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        for blk in self.layers:
            h = blk(h)
        return self.norm(h).mean(dim=1)

    # transfer-learning controls (§3.10.2: lower layers frozen in early epochs when pretrained weights are used)
    def set_frozen_lower(self, fraction: float, frozen: bool) -> None:
        n = int(round(len(self.layers) * fraction))
        for mod in [self.input_proj, *self.layers[:n]]:
            for p in mod.parameters():
                p.requires_grad = not frozen


def _clone_layer(layer: nn.Module) -> nn.Module:
    import copy
    new = copy.deepcopy(layer)
    for p in new.parameters():                                    # deepcopy would share init across layers
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return new


def load_compatible(module: nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Copy tensors whose name and shape match; report what was and was not loaded."""
    own = module.state_dict()
    loaded, shape_mismatch, unexpected = [], [], []
    for k, v in state.items():
        key = k[len("backbone."):] if k.startswith("backbone.") else k
        if key not in own:
            unexpected.append(k)
        elif own[key].shape != v.shape:
            shape_mismatch.append(k)
        else:
            own[key] = v
            loaded.append(key)
    module.load_state_dict(own)
    return {"loaded": len(loaded), "total_params_tensors": len(own), "shape_mismatch": len(shape_mismatch),
            "unexpected": len(unexpected)}


# ---------------------------------------------------------------------------------- heads
def mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = d_in
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, d_hidden), nn.ReLU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class WeibullPHHead(nn.Module):
    """Weibull-modulated proportional hazards (eq. 4-6).

        S(t|z) = exp( -(t/eta)^beta * exp(r) ),   r = w^T z,   beta > 1
        h(t|z) = (beta/eta) (t/eta)^(beta-1) exp(r)

    Here t is *remaining* time (cycles). Equivalent to T | z ~ Weibull(beta, eta * exp(-r/beta)), so
    larger r = higher hazard = shorter RUL. Right-censored samples (event=0) contribute log S only.
    """

    def __init__(self, d_model: int, eta_init: float = 100.0):
        super().__init__()
        self.w = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.w.weight, std=0.01)
        self.log_eta = nn.Parameter(torch.tensor(math.log(eta_init)))
        self.raw_beta = nn.Parameter(torch.tensor(0.5))

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.raw_beta)                    # beta > 1 (wear-out regime)

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
        """Eq. 6: negative Cox partial log-likelihood over events (risk set = samples with time >= t_i)."""
        r = self.risk(z)
        order = torch.argsort(t, descending=True)                 # cumulative logsumexp over the risk set
        r_s, e_s = r[order], event[order]
        log_risk_set = torch.logcumsumexp(r_s, dim=0)
        n_ev = e_s.sum().clamp(min=1.0)
        return -((r_s - log_risk_set) * e_s).sum() / n_ev

    @torch.no_grad()
    def rul_median(self, z: torch.Tensor) -> torch.Tensor:
        """Median remaining time: S(t|z)=0.5."""
        return self.eta * (math.log(2.0) / torch.exp(self.risk(z))) ** (1.0 / self.beta)


class NADIP(nn.Module):
    def __init__(self, n_channels: int, mcfg: Dict, rul_scale: float = 130.0):
        super().__init__()
        d, h = mcfg["d_model"], mcfg["head_hidden"]
        self.backbone = TimeSeriesTransformer(
            n_channels, d, mcfg["nhead"], mcfg["num_layers"], mcfg["dim_feedforward"], mcfg["dropout"],
            mcfg.get("positional", "sinusoidal"))
        self.grl = GradientReversal(1.0)
        self.rul_head = mlp(d, h, 1, 2)
        self.fail_head = mlp(d, h, 1, 2)
        self.domain_disc = mlp(d, h, 1, mcfg.get("discriminator_layers", 3))
        self.projector = mlp(d, d, 64, mcfg.get("projector_layers", 3))
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
            "rul": self.rul_head(z).squeeze(-1) * self.rul_scale,   # cycles
            "fail_logit": self.fail_head(z).squeeze(-1),
            "domain_logit": self.domain_disc(self.grl(z)).squeeze(-1),
            "proj": self.projector(z),
        }

    def param_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """(everything except the discriminator, discriminator) -- the 'dual Adam' of Table 3.18."""
        disc = list(self.domain_disc.parameters())
        ids = {id(p) for p in disc}
        return [p for p in self.parameters() if id(p) not in ids], disc


# ------------------------------------------------------------------------------- losses
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """SimCLR NT-Xent over 2N views (cosine similarity, tau = temperature)."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over rows where mask is True; 0 (with grad graph intact) when no row is labelled."""
    m = mask.float()
    n = m.sum()
    if n.item() == 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * m).sum() / n


def masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    n = m.sum()
    if n.item() == 0:
        return logit.sum() * 0.0
    return (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * m).sum() / n
