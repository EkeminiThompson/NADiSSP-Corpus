"""Contrastive views (§3.4.3, §3.10.2 component 3).

view A : the stored Nigerian-adapted target window
view B : a *fresh* physics-informed augmentation of the same underlying clean source window
both  : + standard augmentations -- 10% channel masking, time-warp (gamma), jitter (sigma)

View B needs the raw (un-normalised) clean base window, so `ViewMaker` inverts the per-dataset
source normaliser stored in corpus_meta.json, augments, and re-normalises.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..data.augmentation import NigerianAugmentor
from ..data.preprocess import Normalizer
from ..data.schema import DATASETS, OFFSETS, CHANNELS, to_unified


def mask_channels(x: torch.Tensor, frac: float, gen: torch.Generator | None = None) -> torch.Tensor:
    keep = (torch.rand(x.shape[0], 1, x.shape[2], generator=gen) >= frac).to(x.dtype)
    return x * keep


def jitter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + sigma * torch.randn_like(x)


def time_warp(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smooth monotone re-timing; gamma scales the warp strength (0 = identity)."""
    B, T, _ = x.shape
    if gamma <= 0:
        return x
    speed = 1.0 + gamma * 0.5 * torch.randn(B, 4)                       # 4 knots of local playback speed
    speed = torch.nn.functional.interpolate(speed.unsqueeze(1), size=T, mode="linear", align_corners=True).squeeze(1)
    speed = speed.clamp(min=0.2)
    pos = torch.cumsum(speed, dim=1)
    pos = (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1] + 1e-8) * (T - 1)
    lo = pos.floor().long().clamp(0, T - 1)
    hi = (lo + 1).clamp(max=T - 1)
    w = (pos - lo.float()).unsqueeze(-1)
    g = lambda i: torch.gather(x, 1, i.unsqueeze(-1).expand(-1, -1, x.shape[2]))
    return g(lo) * (1 - w) + g(hi) * w


def standard_augment(x: torch.Tensor, cfg: Dict) -> torch.Tensor:
    x = time_warp(x, cfg["time_warp_gamma"])
    x = mask_channels(x, cfg["mask_channel_frac"])
    return jitter(x, cfg["jitter_sigma"])


class ViewMaker:
    """Holds clean base windows for a set of target rows and re-augments them on demand."""

    def __init__(self, bundle, meta: Dict, aug_cfg: Dict, seed: int):
        self.bundle = bundle
        self.aug = NigerianAugmentor(aug_cfg, seed=seed)
        self.norms = {k: Normalizer.from_dict(v) for k, v in meta["normalizers"].items()}
        self.names = {i: d for i, d in enumerate(DATASETS)}

    def view_b(self, rows: np.ndarray) -> torch.Tensor:
        """Fresh Nigerian augmentation of the clean source window behind each target row."""
        b = self.bundle
        out = np.empty((len(rows), b.T, b.C), dtype=np.float32)
        for k, r in enumerate(rows):
            base = int(b.c["base"][r])
            name = self.names[int(b.dataset[r])]
            if base < 0 or name not in self.norms:
                out[k] = b.X[r].numpy()                                     # no clean base: reuse the stored window
                continue
            o, n = OFFSETS[name], len(CHANNELS[name])
            norm = self.norms[name]
            raw = norm.inverse(b.X[base, :, o:o + n].numpy())
            xa, _ = self.aug.apply(raw, name)
            out[k] = to_unified(name, norm.transform(xa)[None])[0]
        return torch.from_numpy(out)
