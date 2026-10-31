"""Window enumeration, stratified sampling (§3.3.3, Table 3.8) and group-level splitting.

Splitting is by *group* (engine / well instance / ESP unit / AI4I block): all windows of a unit,
and all Nigerian-augmented variants derived from them, land in the same split. Splitting windows
at random would put overlapping windows (stride 1, length 50) on both sides of the train/test line.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .loaders import Series


def enumerate_windows(series: List[Series], window: int, stride: int) -> Dict[str, np.ndarray]:
    """Candidate windows as parallel arrays (series index, end index and the labels at the window end)."""
    s_idx, ends = [], []
    for i, s in enumerate(series):
        T = len(s.X)
        if T < window:
            continue
        e = np.arange(window - 1, T, stride)
        s_idx.append(np.full(len(e), i, dtype=np.int32))
        ends.append(e.astype(np.int32))
    if not s_idx:
        z = np.empty(0, dtype=np.int32)
        return {"s": z, "end": z, "fail": z, "health": z, "event": z, "has_rul": z.astype(bool),
                "rul_raw": z.astype(np.float32), "group": np.empty(0, dtype=object)}
    s_idx, ends = np.concatenate(s_idx), np.concatenate(ends)
    out = {"s": s_idx, "end": ends}
    for key, attr in (("fail", "fail"), ("health", "health"), ("event", "event_class"), ("rul_raw", "rul_raw")):
        out[key] = np.empty(len(ends), dtype=np.float32 if key == "rul_raw" else np.int16)
    out["has_rul"] = np.zeros(len(ends), dtype=bool)
    out["group"] = np.empty(len(ends), dtype=object)
    pos = 0
    for i, s in enumerate(series):
        m = int((s_idx == i).sum())
        if m == 0:
            continue
        e = ends[pos:pos + m]
        out["fail"][pos:pos + m] = s.fail[e]
        out["health"][pos:pos + m] = s.health[e]
        out["event"][pos:pos + m] = s.event_class[e]
        out["rul_raw"][pos:pos + m] = s.rul_raw[e]
        out["has_rul"][pos:pos + m] = s.has_rul
        out["group"][pos:pos + m] = s.unit
        pos += m
    return out


def assign_splits(groups: np.ndarray, rng: np.random.Generator, test: float, val: float) -> np.ndarray:
    """Split id per window (0 train, 1 val, 2 test) with whole groups assigned together.

    Groups are shuffled and allocated greedily to reach the target window fractions. With fewer than
    three groups there is nothing to hold out without leakage, so everything stays in train.
    """
    split = np.zeros(len(groups), dtype=np.int8)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(uniq) < 3:
        return split
    order = rng.permutation(len(uniq))
    total = counts.sum()
    want = {2: test * total, 1: val * total}
    got = {2: 0, 1: 0}
    assign = np.zeros(len(uniq), dtype=np.int8)
    for g in order:                                   # test first, then val; at least one group each
        for sid in (2, 1):
            if got[sid] < want[sid] or got[sid] == 0:
                assign[g] = sid
                got[sid] += counts[g]
                break
    return assign[inv]


def _allocate(counts: Dict[str, int], n: int, min_cell: int) -> Dict[str, int]:
    """Proportional allocation with a floor of min(min_cell, stratum size) per stratum."""
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    keys = sorted(counts)
    alloc = {k: min(counts[k], max(int(np.floor(n * counts[k] / total)), min(min_cell, counts[k]))) for k in keys}
    diff = n - sum(alloc.values())
    # trim or top-up the largest strata first
    for k in sorted(keys, key=lambda k: -counts[k]) * (abs(diff) + 1):
        if diff == 0:
            break
        if diff > 0 and alloc[k] < counts[k]:
            alloc[k] += 1
            diff -= 1
        elif diff < 0 and alloc[k] > min(min_cell, counts[k]):
            alloc[k] -= 1
            diff += 1
    return alloc


def stratified_sample(strata: np.ndarray, n: int, rng: np.random.Generator, min_cell: int) -> np.ndarray:
    """Indices of a stratified draw of (up to) n items, without replacement."""
    if n <= 0 or len(strata) == 0:
        return np.empty(0, dtype=int)
    keys, inv = np.unique(strata.astype(str), return_inverse=True)
    counts = {k: int((inv == i).sum()) for i, k in enumerate(keys)}
    alloc = _allocate(counts, n, min_cell)
    picks = []
    for i, k in enumerate(keys):
        members = np.flatnonzero(inv == i)
        picks.append(rng.choice(members, size=alloc[k], replace=False))
    return np.sort(np.concatenate(picks))


def sample_w3(event: np.ndarray, provenance: np.ndarray, n: int, rng: np.random.Generator,
              min_cell: int, min_real_fraction: float) -> np.ndarray:
    """3W: stratify by event type x provenance with real instances making up >= min_real_fraction of each
    event-type draw (Table 3.8); simulated instances only fill the remainder."""
    if n <= 0 or len(event) == 0:
        return np.empty(0, dtype=int)
    ev_keys, inv = np.unique(event, return_inverse=True)
    counts = {str(k): int((inv == i).sum()) for i, k in enumerate(ev_keys)}
    alloc = _allocate(counts, n, min_cell)
    picks = []
    for i, k in enumerate(ev_keys):
        members = np.flatnonzero(inv == i)
        real = members[provenance[members] == "real"]
        sim = members[provenance[members] != "real"]
        need = alloc[str(k)]
        n_real = min(len(real), max(int(np.ceil(min_real_fraction * need)), need - len(sim)))
        n_sim = min(len(sim), need - n_real)
        picks.append(rng.choice(real, size=n_real, replace=False) if n_real else np.empty(0, dtype=int))
        picks.append(rng.choice(sim, size=n_sim, replace=False) if n_sim else np.empty(0, dtype=int))
    return np.sort(np.concatenate(picks)) if picks else np.empty(0, dtype=int)
