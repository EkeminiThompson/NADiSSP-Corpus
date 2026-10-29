"""Domain diagnostics (Objective 1; Tables 3.17, 4.1, 4.2).

MMD       RBF kernel, sigma = 1.0 (biased estimator, square-rooted). Distances are divided by the feature
          dimension so that sigma = 1 means the same thing for 3-D per-channel features and 100-D
          composite features; features are standardised with SOURCE statistics.
W1        1-Wasserstein via log-domain Sinkhorn, eps = 0.01, Euclidean cost / sqrt(d). Per-channel 1-D
          Wasserstein (scipy, exact) is reported alongside for the channel heat maps.
Embedding-space numbers use the same estimators on z (standardised with source statistics), so
"pre" (source-only model) and "post" (adapted model) are comparable with each other. Absolute values
are NOT comparable to the dissertation's tables unless its exact feature space and normalisation are
reproduced; reductions (%) are the meaningful quantity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..data.schema import CHANNELS, DATASETS, DATASET_ID, N_CHANNELS, OFFSETS, STRESSORS, STRESSOR_ROLES, role_indices

INTERVENTION_THRESHOLD = 0.40          # §4.2.1


# ------------------------------------------------------------------------------ estimators
def _sub(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2, b2 = (A ** 2).sum(1)[:, None], (B ** 2).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)


def mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, max_samples: int = 1000, seed: int = 42,
        dim_normalise: bool = True) -> float:
    rng = np.random.default_rng(seed)
    X, Y = _sub(np.asarray(X, np.float64), max_samples, rng), _sub(np.asarray(Y, np.float64), max_samples, rng)
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    d = X.shape[1] if dim_normalise else 1
    k = lambda A, B: np.exp(-_sqdist(A, B) / (2.0 * sigma ** 2 * d))
    m2 = k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean()
    return float(np.sqrt(max(m2, 0.0)))


def sinkhorn_w1(X: np.ndarray, Y: np.ndarray, eps: float = 0.01, max_samples: int = 1000, n_iter: int = 200,
                seed: int = 42) -> float:
    """Transport cost <P, C> of the entropy-regularised plan, C_ij = ||x_i - y_j|| / sqrt(d)."""
    rng = np.random.default_rng(seed)
    X, Y = _sub(np.asarray(X, np.float64), max_samples, rng), _sub(np.asarray(Y, np.float64), max_samples, rng)
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    x, y = torch.from_numpy(X), torch.from_numpy(Y)
    C = torch.cdist(x, y) / np.sqrt(X.shape[1])
    n, m = len(X), len(Y)
    log_a, log_b = torch.full((n,), -np.log(n), dtype=torch.float64), torch.full((m,), -np.log(m), dtype=torch.float64)
    f, g = torch.zeros(n, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    for _ in range(n_iter):
        f = -eps * torch.logsumexp((g[None, :] - C) / eps + log_b[None, :], dim=1)
        g = -eps * torch.logsumexp((f[:, None] - C) / eps + log_a[:, None], dim=0)
    P = torch.exp((f[:, None] + g[None, :] - C) / eps + log_a[:, None] + log_b[None, :])
    return float((P * C).sum())


def w1_per_dim(X: np.ndarray, Y: np.ndarray, max_samples: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    X, Y = _sub(np.asarray(X), max_samples, rng), _sub(np.asarray(Y), max_samples, rng)
    return float(np.mean([wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])]))


def mmd_permutation_p(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, n_perm: int = 200, max_samples: int = 400,
                      seed: int = 42) -> float:
    """p-value for H0: X and Y come from the same distribution (permutation test on the MMD statistic)."""
    rng = np.random.default_rng(seed)
    X, Y = _sub(np.asarray(X, np.float64), max_samples, rng), _sub(np.asarray(Y, np.float64), max_samples, rng)
    if len(X) < 4 or len(Y) < 4:
        return float("nan")
    obs = mmd(X, Y, sigma, max_samples=len(X) + len(Y), seed=seed)
    Z = np.vstack([X, Y])
    n = len(X)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        ge += mmd(Z[p[:n]], Z[p[n:]], sigma, max_samples=len(Z), seed=seed) >= obs
    return float((ge + 1) / (n_perm + 1))


# -------------------------------------------------------------------------- feature spaces
def _own_block(bundle, rows: np.ndarray, dataset: str, channels: Optional[np.ndarray] = None) -> np.ndarray:
    o, n = OFFSETS[dataset], len(CHANNELS[dataset])
    X = bundle.X[torch.from_numpy(rows)][:, :, o:o + n].numpy()
    if channels is not None:
        X = X[:, :, channels]
    return np.concatenate([X.mean(1), X.std(1), X[:, -1] - X[:, 0]], axis=1)


def _standardise(ref: np.ndarray, *others: np.ndarray):
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(a - mu) / sd for a in (ref, *others)]


def _pair_metrics(S: np.ndarray, T: np.ndarray, cfg: Dict, perm: bool = False) -> Dict:
    ev = cfg["evaluation"]
    S, T = _standardise(S, T)
    out = {"mmd": mmd(S, T, ev["mmd_sigma"], ev["max_samples"]),
           "w1_sinkhorn": sinkhorn_w1(S, T, ev["sinkhorn_eps"], ev["max_samples"]),
           "w1_1d_mean": w1_per_dim(S, T), "n_source": int(len(S)), "n_target": int(len(T))}
    if perm:
        out["mmd_perm_p"] = mmd_permutation_p(S, T, ev["mmd_sigma"])
    return out


# ------------------------------------------------------------------------ input-space report
def input_space_report(bundle, cfg: Dict, splits: Tuple[str, ...] = ("train", "val", "test")) -> Dict:
    """Clean source vs Nigerian-adapted target, per dataset, per stressor and per channel (Table 4.1 analogue)."""
    rep: Dict = {"per_dataset": {}, "per_stressor": [], "per_channel": []}
    pool_s, pool_t = [], []
    for name in bundle.present_datasets():
        did = DATASET_ID[name]
        rs = np.concatenate([bundle.idx(s, 0, [name]) for s in splits])
        rt = np.concatenate([bundle.idx(s, 1, [name]) for s in splits])
        if len(rs) < 4 or len(rt) < 4:
            continue
        S, T = _own_block(bundle, rs, name), _own_block(bundle, rt, name)
        rep["per_dataset"][name] = _pair_metrics(S, T, cfg, perm=True)
        pool_s.append(S); pool_t.append(T)
        # per channel
        for j, (cn, role) in enumerate(CHANNELS[name]):
            Sj, Tj = _own_block(bundle, rs, name, np.array([j])), _own_block(bundle, rt, name, np.array([j]))
            rep["per_channel"].append({"dataset": name, "channel": cn, "role": role, **_pair_metrics(Sj, Tj, cfg)})
        # per stressor (flags are stored per target row)
        for si, s in enumerate(STRESSORS):
            flagged = rt[bundle.c["flags"][rt, si] == 1]
            ch = role_indices(name, STRESSOR_ROLES[s])
            if len(flagged) < 8 or len(ch) == 0:
                continue
            rep["per_stressor"].append({"dataset": name, "stressor": s, "channels": [CHANNELS[name][j][0] for j in ch],
                                        **_pair_metrics(_own_block(bundle, rs, name, ch),
                                                        _own_block(bundle, flagged, name, ch), cfg)})
    if pool_s:
        # composite: per-dataset standardised features, padded to a common width (zeros elsewhere)
        width = max(p.shape[1] for p in pool_s)
        pad = lambda arr: np.pad(arr, ((0, 0), (0, width - arr.shape[1])))
        Sc = np.vstack([pad(_standardise(s, s)[0]) for s in pool_s])
        Tc = np.vstack([pad(_standardise(s, t)[1]) for s, t in zip(pool_s, pool_t)])
        ev = cfg["evaluation"]
        rep["composite"] = {"mmd": mmd(Sc, Tc, ev["mmd_sigma"], ev["max_samples"]),
                            "w1_sinkhorn": sinkhorn_w1(Sc, Tc, ev["sinkhorn_eps"], ev["max_samples"]),
                            "w1_1d_mean": w1_per_dim(Sc, Tc), "n_source": int(len(Sc)), "n_target": int(len(Tc))}
        rep["composite"]["above_intervention_threshold"] = bool(rep["composite"]["mmd"] > INTERVENTION_THRESHOLD)
    rep["intervention_threshold"] = INTERVENTION_THRESHOLD
    return rep


# --------------------------------------------------------------------- embedding-space report
def discriminator_accuracy(Z: np.ndarray, domain: np.ndarray, seed: int = 42) -> float:
    """Cross-validated accuracy of a fresh logistic-regression domain classifier on z. ~0.5 means aligned."""
    if len(np.unique(domain)) < 2 or min(np.bincount(domain)) < 5:
        return float("nan")
    Zs = StandardScaler().fit_transform(Z)
    return float(cross_val_score(LogisticRegression(max_iter=500, class_weight="balanced"), Zs, domain,
                                 cv=min(5, int(np.bincount(domain).min())), scoring="balanced_accuracy").mean())


def tsne_embed(Z: np.ndarray, seed: int = 42, perplexity: float = 30.0, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(Z), size=min(len(Z), max_points), replace=False))
    perp = min(perplexity, max(5.0, (len(idx) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(Z[idx]), idx


def embedding_shift(Z: np.ndarray, domain: np.ndarray, cfg: Dict, seed: int = 42, tsne: bool = True) -> Dict:
    ev = cfg["evaluation"]
    S, T = Z[domain == 0], Z[domain == 1]
    if len(S) < 4 or len(T) < 4:
        return {"note": "too few rows in one domain"}
    Zs, Ss, Ts = _standardise(Z, S, T)
    out = {"mmd": mmd(Ss, Ts, ev["mmd_sigma"], ev["max_samples"]),
           "w1_sinkhorn": sinkhorn_w1(Ss, Ts, ev["sinkhorn_eps"], ev["max_samples"]),
           "w1_1d_mean": w1_per_dim(Ss, Ts), "discriminator_accuracy": discriminator_accuracy(Zs, domain, seed)}
    if tsne and len(np.unique(domain)) == 2:
        emb, idx = tsne_embed(Zs, seed)
        out["tsne_silhouette"] = float(silhouette_score(emb, domain[idx])) if len(np.unique(domain[idx])) == 2 else float("nan")
    return out


def reduction(pre: float, post: float) -> float:
    return float("nan") if (pre is None or np.isnan(pre) or pre == 0) else float((pre - post) / pre)


def domain_report(pre_trained, post_trained, bundle, cfg: Dict, split: str = "test") -> Dict:
    """Pre-adaptation (source-only model) vs post-adaptation (adapted model) embedding shift, plus the model's own
    discriminator accuracy. Targets are echoed from the config for comparison -- not asserted."""
    rows = bundle.idx(split)
    dom = bundle.domain[rows]
    rep: Dict = {"split": split, "n_rows": int(len(rows)), "targets": cfg["evaluation"]["targets"]}
    for label, t in (("pre_source_only", pre_trained), ("post_adapted", post_trained)):
        out = t.predict(bundle.X[rows], bundle.dataset[rows])
        rep[label] = embedding_shift(out["z"], dom, cfg)
        if t.model is not None:
            with torch.no_grad():
                zt = torch.from_numpy(out["z"]).to(next(t.model.parameters()).device)
                logit = t.model.domain_disc(zt).squeeze(-1).cpu()
            rep[label]["model_discriminator_accuracy"] = float(((logit > 0).numpy() == dom.astype(bool)).mean())
    a, b = rep["pre_source_only"], rep["post_adapted"]
    rep["reduction"] = {k: reduction(a.get(k), b.get(k)) for k in ("mmd", "w1_sinkhorn", "w1_1d_mean")}
    tg = cfg["evaluation"]["targets"]
    rep["meets_targets"] = {"mmd_le_target": bool(b.get("mmd", np.inf) <= tg["mmd"]),
                            "w1_reduction_ge_target": bool((rep["reduction"]["w1_sinkhorn"] or -1) >= tg["wasserstein_reduction"])}
    return rep
