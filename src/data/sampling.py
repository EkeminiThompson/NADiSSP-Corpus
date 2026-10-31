"""Group-wise splits and stratified sampling (Tables 3.8, 3.13)."""

from __future__ import annotations

import numpy as np


def assign_splits(
    groups: np.ndarray,
    rng: np.random.Generator,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
) -> np.ndarray:
    """Assign each *group* to train(0)/val(1)/test(2). Never splits a group across folds."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(uniq)
    if n < 3:
        return np.zeros(len(groups), dtype=np.int8)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test / n <= test_frac and n_test + 1 + n_val < n:
        n_test += 1
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
        if n_test + n_val >= n:
            return np.zeros(len(groups), dtype=np.int8)
    test_g = set(uniq[:n_test])
    val_g = set(uniq[n_test:n_test + n_val])
    out = np.zeros(len(groups), dtype=np.int8)
    for i, g in enumerate(groups):
        if g in test_g:
            out[i] = 2
        elif g in val_g:
            out[i] = 1
    return out


def stratified_sample(
    strata: np.ndarray,
    size: int,
    rng: np.random.Generator,
    min_cell: int = 5,
) -> np.ndarray:
    """Sample `size` indices with at least min_cell per stratum when possible."""
    strata = np.asarray(strata)
    keys, inv, counts = np.unique(strata, return_inverse=True, return_counts=True)
    n = len(strata)
    size = min(size, n)
    alloc = np.minimum(counts, min_cell).astype(int)
    remaining = size - int(alloc.sum())
    if remaining < 0:
        order = np.argsort(-counts)
        alloc[:] = 0
        left = size
        for i in order:
            take = min(counts[i], max(1, left // max(1, len(keys) - list(order).index(i))))
            take = min(take, left, counts[i])
            alloc[i] = take
            left -= take
            if left <= 0:
                break
        remaining = 0
    if remaining > 0:
        capacity = counts - alloc
        while remaining > 0 and capacity.sum() > 0:
            weights = capacity / capacity.sum()
            pick = int(rng.choice(len(keys), p=weights))
            if capacity[pick] > 0:
                alloc[pick] += 1
                capacity[pick] -= 1
                remaining -= 1
            else:
                capacity[pick] = 0
    chosen = []
    for k_i, k in enumerate(keys):
        idx = np.flatnonzero(inv == k_i)
        rng.shuffle(idx)
        chosen.append(idx[: alloc[k_i]])
    out = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    rng.shuffle(out)
    return out.astype(int)


def sample_w3(
    event: np.ndarray,
    provenance: np.ndarray,
    size: int,
    rng: np.random.Generator,
    min_cell: int = 5,
    min_real_fraction: float = 0.7,
) -> np.ndarray:
    """Stratify by event class while enforcing min_real_fraction of real wells per cell."""
    event = np.asarray(event)
    provenance = np.asarray(provenance)
    keys = np.unique(event)
    pools = []
    for e in keys:
        mask = event == e
        real = np.flatnonzero(mask & (provenance == "real"))
        sim = np.flatnonzero(mask & (provenance != "real"))
        rng.shuffle(real)
        rng.shuffle(sim)
        pools.append((e, real, sim))
    per = max(min_cell, size // max(1, len(keys)))
    chosen = []
    for e, real, sim in pools:
        n_real = int(np.ceil(per * min_real_fraction))
        n_real = min(n_real, len(real))
        n_sim = min(per - n_real, len(sim))
        take = list(real[:n_real]) + list(sim[:n_sim])
        chosen.extend(take)
    chosen = np.array(chosen, dtype=int)
    if len(chosen) < size:
        rest = np.setdiff1d(np.arange(len(event)), chosen, assume_unique=False)
        rest_real = rest[provenance[rest] == "real"]
        rest_other = rest[provenance[rest] != "real"]
        rng.shuffle(rest_real)
        rng.shuffle(rest_other)
        fill = np.concatenate([rest_real, rest_other])[: size - len(chosen)]
        chosen = np.concatenate([chosen, fill])
    if len(chosen) > size:
        sim_mask = provenance[chosen] != "real"
        sim_idx = np.flatnonzero(sim_mask)
        rng.shuffle(sim_idx)
        drop = len(chosen) - size
        keep = np.ones(len(chosen), dtype=bool)
        keep[sim_idx[:drop]] = False
        if keep.sum() > size:
            keep[np.flatnonzero(keep)[size:]] = False
        chosen = chosen[keep]
    rng.shuffle(chosen)
    return chosen[:size].astype(int)
