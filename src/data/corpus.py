"""Corpus builder (Tables 3.10-3.13): source windows + Nigerian-adapted target variants.

Pipeline per dataset
    load series -> forward-fill -> enumerate windows -> split BY GROUP (train/val/test)
    -> stratified sample inside each split -> clip (source-fitted IQR bounds)
    -> target variants: re-use sampled source windows, augment (conditional by asset analogy)
    -> z-score (source-train statistics) -> place into the unified channel vector.

Everything the builder actually produced (counts, prevalence, which stressors fired, datasets that
were missing, shortfalls against the requested sizes) is written to <processed>/corpus_meta.json.
The requested sizes of Table 3.11 are *targets*: if a dataset has fewer windows, the corpus is
smaller and the metadata says so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .augmentation import NigerianAugmentor
from .loaders import Series, load_all
from .preprocess import Normalizer, cap_rul, fill_series
from .sampling import assign_splits, enumerate_windows, sample_w3, stratified_sample
from .schema import DATASET_ID, DATASETS, N_CHANNELS, STRESSORS, to_unified

SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def _take(series: List[Series], s_idx: np.ndarray, end: np.ndarray, window: int) -> np.ndarray:
    out = np.empty((len(s_idx), window, series[0].X.shape[1]), dtype=np.float32)
    for k, (i, e) in enumerate(zip(s_idx, end)):
        out[k] = series[i].X[e - window + 1:e + 1]
    return out


def _strata(name: str, cand: Dict[str, np.ndarray], series: List[Series]) -> np.ndarray:
    ev, health = cand["event"].astype(int), cand["health"].astype(int)
    if name == "cmapss":
        subset = np.array([series[i].meta["subset"] for i in cand["s"]])
        return np.char.add(np.char.add(subset.astype(str), "|"), health.astype(str))
    if name == "espset" and series[0].meta.get("mode") == "bands":
        return ev.astype(str)                               # frequency-sweep ESPset: stratify by fault class only
    if name == "espset":
        return np.char.add(ev.astype(str), np.char.add("|", cand["group"].astype(str)))
    return ev.astype(str)                                   # ai4i: failure type; w3 handled separately


def build_corpus(cfg: Dict, raw_dir: Path, out_dir: Path) -> Dict:
    d, aug_cfg = cfg["data"], cfg["augmentation"]
    window = d["window"]
    rng = np.random.default_rng(cfg["seed"])
    augmentor = NigerianAugmentor(aug_cfg, seed=cfg["seed"] + 1)

    found, missing = load_all(raw_dir, cfg)
    if not found:
        raise RuntimeError(
            "No datasets could be loaded; nothing was generated.\n" + "\n".join(f"  {k}: {v}" for k, v in missing.items())
        )

    src_blocks, tgt_blocks, norms, strat_log = [], [], {}, []
    meta: Dict = {"seed": cfg["seed"], "window": window, "stride": d["stride"], "missing_datasets": missing,
                  "datasets": {}, "shortfalls": []}
    src_offset = 0

    for name in [x for x in DATASETS if x in found]:
        series = found[name]
        for s in series:
            s.X = fill_series(s.X)
        cand = enumerate_windows(series, window, d["stride"])
        if len(cand["end"]) == 0:
            meta["missing_datasets"][name] = f"no series of length >= {window}"
            continue
        split = assign_splits(cand["group"], rng, d["split"]["test"], d["split"]["val"])
        prov = None
        if name == "w3":
            prov = np.array([series[i].meta["provenance"] for i in cand["s"]])
        strata = _strata(name, cand, series)

        n_src_total, n_tgt_total = d["source_counts"][name], d["target_counts"][name]
        src_rows: List[np.ndarray] = []
        tgt_rows: List[np.ndarray] = []              # (base candidate id, split)
        for sid in (0, 1, 2):
            idx = np.flatnonzero(split == sid)
            if len(idx) == 0:
                continue
            share = len(idx) / len(split)
            n_s = int(round(n_src_total * share))
            if name == "w3":
                pick = idx[sample_w3(cand["event"][idx], prov[idx], n_s, rng, d["min_cell"], d["w3_min_real_fraction"])]
            else:
                pick = idx[stratified_sample(strata[idx], n_s, rng, d["min_cell"])]
            if len(pick) == 0:
                continue
            src_rows.append(pick)
            n_t = int(round(n_tgt_total * share))
            tgt_rows.append(np.resize(rng.permutation(pick), n_t) if n_t > 0 else np.empty(0, dtype=int))
            for k, c in zip(*np.unique(strata[pick], return_counts=True)):
                strat_log.append({"dataset": name, "split": SPLIT_NAMES[sid], "stratum": str(k),
                                  "available": int((strata[idx] == k).sum()), "sampled": int(c)})
        if not src_rows:
            meta["missing_datasets"][name] = "no windows survived sampling"
            continue
        src_pick = np.concatenate(src_rows)
        tgt_pick = np.concatenate(tgt_rows) if tgt_rows else np.empty(0, dtype=int)
        if len(src_pick) < n_src_total:
            meta["shortfalls"].append({"dataset": name, "requested_source": n_src_total, "available": int(len(src_pick))})

        # --- normaliser fitted on clean source TRAIN windows only
        raw_src = _take(series, cand["s"][src_pick], cand["end"][src_pick], window)
        train_mask = split[src_pick] == 0
        fit_on = raw_src[train_mask] if train_mask.any() else raw_src
        norm = Normalizer(d["iqr_k"]).fit(fit_on)
        norms[name] = norm
        src_clipped = norm.clip(raw_src)
        src_Z = norm.transform(src_clipped)

        # --- target variants: base windows -> augmentation -> same normaliser
        pos_of = {int(c): i for i, c in enumerate(src_pick)}
        base_pos = np.array([pos_of[int(c)] for c in tgt_pick], dtype=int)
        tgt_Z = np.empty((len(tgt_pick), window, src_Z.shape[-1]), dtype=np.float32)
        flags = np.zeros((len(tgt_pick), len(STRESSORS)), dtype=np.uint8)
        n_gaps = np.zeros(len(tgt_pick), dtype=np.int16)
        for k, p in enumerate(base_pos):
            xa, info = augmentor.apply(src_clipped[p], name)
            tgt_Z[k] = norm.transform(xa)
            for s_name in info["applied"]:
                flags[k, STRESSORS.index(s_name)] = 1
            n_gaps[k] = info["n_gaps"]

        def labels(rows: np.ndarray) -> Dict[str, np.ndarray]:
            has = cand["has_rul"][rows]
            raw = cand["rul_raw"][rows]
            return {
                "rul": np.where(has, cap_rul(np.nan_to_num(raw, nan=0.0), d["rul_cap"]), 0.0).astype(np.float32),
                "has_rul": has,
                "event": (has & (raw < d["rul_cap"])),
                "fail": cand["fail"][rows].astype(np.float32),
                "health": cand["health"][rows].astype(np.int8),
                "stratum_event": cand["event"][rows].astype(np.int16),
                "group": cand["group"][rows],
                "split": split[rows],
            }

        ls = labels(src_pick)
        lt = labels(tgt_pick) if len(tgt_pick) else {k: v[:0] for k, v in ls.items()}
        n_s, n_t = len(src_pick), len(tgt_pick)
        src_blocks.append({"name": name, "X": to_unified(name, src_Z), **ls,
                           "domain": np.zeros(n_s, np.int8), "base": np.full(n_s, -1, np.int64),
                           "flags": np.zeros((n_s, len(STRESSORS)), np.uint8), "n_gaps": np.zeros(n_s, np.int16),
                           "w3_real": (prov[src_pick] == "real") if prov is not None else np.zeros(n_s, bool)})
        tgt_blocks.append({"name": name, "X": to_unified(name, tgt_Z), **lt,
                           "domain": np.ones(n_t, np.int8), "base": src_offset + base_pos.astype(np.int64),
                           "flags": flags, "n_gaps": n_gaps,
                           "w3_real": (prov[tgt_pick] == "real") if prov is not None else np.zeros(n_t, bool)})
        src_offset += n_s

        # per-dataset report (what was really produced)
        fail_prev = {SPLIT_NAMES[s]: float(ls["fail"][ls["split"] == s].mean()) if (ls["split"] == s).any() else None for s in (0, 1, 2)}
        meta["datasets"][name] = {
            "n_series": len(series), "n_candidate_windows": int(len(cand["end"])),
            "n_source": int(n_s), "n_target": int(n_t),
            "requested_source": n_src_total, "requested_target": n_tgt_total,
            "source_by_split": {SPLIT_NAMES[s]: int((ls["split"] == s).sum()) for s in (0, 1, 2)},
            "fail_prevalence_source": fail_prev,
            "has_rul_fraction": float(ls["has_rul"].mean()),
            "stressor_rates_target": {s: float(flags[:, i].mean()) if n_t else 0.0 for i, s in enumerate(STRESSORS)},
            "norm_fitted_on_windows": int(len(fit_on)),
        }
        if name == "w3":
            meta["datasets"][name]["real_fraction_source"] = float((prov[src_pick] == "real").mean())

    blocks = src_blocks + tgt_blocks
    cat = lambda k: np.concatenate([b[k] for b in blocks]) if blocks else np.empty(0)
    N = sum(len(b["X"]) for b in blocks)
    ds_id = np.concatenate([np.full(len(b["X"]), DATASET_ID[b["name"]], np.int8) for b in blocks])
    split_all = cat("split").astype(np.int8)
    domain = cat("domain")

    # 30% of Nigerian-target labels are retained, 70% stripped (§3.4.3). Source labels are always available.
    label_avail = np.ones(N, dtype=bool)
    tgt_mask = domain == 1
    label_avail[tgt_mask] = rng.random(int(tgt_mask.sum())) < d["target_label_fraction"]

    corpus = {
        "X": cat("X").astype(np.float32), "rul": cat("rul"), "has_rul": cat("has_rul"), "event": cat("event"),
        "fail": cat("fail"), "health": cat("health"), "stratum_event": cat("stratum_event"),
        "group": np.array([f"{n}:{g}" for n, g in zip(np.concatenate([[b['name']] * len(b['X']) for b in blocks]), cat("group"))]),
        "dataset": ds_id, "domain": domain, "split": split_all, "base": cat("base"), "flags": cat("flags"),
        "n_gaps": cat("n_gaps"), "label_avail": label_avail, "w3_real": cat("w3_real"),
    }
    meta["n_total"] = int(N)
    meta["n_source"] = int((domain == 0).sum())
    meta["n_target"] = int((domain == 1).sum())
    meta["n_channels"] = N_CHANNELS
    meta["split_counts"] = {SPLIT_NAMES[s]: {"source": int(((split_all == s) & (domain == 0)).sum()),
                                           "target": int(((split_all == s) & (domain == 1)).sum())} for s in (0, 1, 2)}
    meta["prevalence_outside_8_20pct"] = [
        n for n, v in meta["datasets"].items()
        if v["fail_prevalence_source"].get("train") is not None and not (0.08 <= v["fail_prevalence_source"]["train"] <= 0.20)
    ]
    meta["normalizers"] = {k: v.to_dict() for k, v in norms.items()}
    meta["config_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                      sort_keys=True, default=str).encode()).hexdigest()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "corpus.npz", **corpus)
    with open(out_dir / "corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    try:
        import pandas as pd
        pd.DataFrame(strat_log).to_csv(out_dir / "stratification_log.csv", index=False)
    except Exception:
        pass
    return {"corpus": corpus, "meta": meta}


def load_corpus(processed_dir: Path) -> Dict:
    z = np.load(Path(processed_dir) / "corpus.npz", allow_pickle=True)
    corpus = {k: z[k] for k in z.files}
    with open(Path(processed_dir) / "corpus_meta.json") as f:
        meta = json.load(f)
    return {"corpus": corpus, "meta": meta}
