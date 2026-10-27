"""
NADIP core architecture (PyTorch).

Components:
- Transformer encoder (time-series)
- Gradient Reversal Layer (GRL) for adversarial domain adaptation
- Optional SimCLR-style projection head
- Weibull RUL head + binary failure head
- Optional XGBoost on frozen embeddings (post-hoc)
"""

from __future__ import annotations
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
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.d_model = d_model

    def forward(self, x):  # x: (B, T, C)
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos_emb[:, :T, :]
        h = self.encoder(h)
        return h.mean(dim=1)  # (B, d_model)


class NADIP(nn.Module):
    def __init__(
        self,
        n_channels: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.backbone = TimeSeriesTransformer(
            n_channels=n_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.grl = GradientReversal(lambda_=grl_lambda)

        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.fail_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.domain_disc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),
        )

    def set_grl_lambda(self, lambda_: float):
        self.grl.lambda_ = lambda_

    def forward(self, x, return_embedding: bool = False):
        emb = self.backbone(x)
        rul = self.rul_head(emb).squeeze(-1)
        fail_logit = self.fail_head(emb).squeeze(-1)
        domain_logit = self.domain_disc(self.grl(emb)).squeeze(-1)
        proj = self.projector(emb)
        if return_embedding:
            return {
                "rul": rul,
                "fail_logit": fail_logit,
                "domain_logit": domain_logit,
                "embedding": emb,
                "projection": proj,
            }
        return rul, fail_logit, domain_logit


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Simple NT-Xent (SimCLR) loss for two views."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(N, device=z.device)
    targets = torch.cat([targets + N, targets], dim=0)
    loss = F.cross_entropy(sim, targets)
    return loss
