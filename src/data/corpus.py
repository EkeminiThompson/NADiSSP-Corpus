"""Build and load the experimental corpus (windows, domains, splits, meta)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .augmentation import NigerianAugmentor
from .loaders import Series, load_all
from .preprocess import Normalizer, cap_rul
from .sampling import assign_splits, sample_w3, stratified_sample
from .schema import CHANNELS, DATASET_ID, DATASETS, N_CHANNELS, OFFSETS, STRESSORS, to_unified


def _window_series(ser: Series, window: int, stride: int):
    X, n = ser.X, len(ser.X)
    if n < window:
        return [], [], [], [], [], [], []
    Xs, ruls, fails, events, hrs, groups, provs = [], [], [], [], [], [], []
    prov = str(ser.meta.get("provenance", "unknown"))
    for start in range(0, n - window + 1, stride):
        end = start + window
        Xs.append(X[start:end])
        ruls.append(float(ser.rul_raw[end - 1]) if ser.has_rul else float("nan"))
        fails.append(int(ser.fail[end - 1]))
        events.append(int(ser.event_class[end - 1]))
        hrs.append(1 if ser.has_rul else 0)
        groups.append(ser.unit)
        provs.append(prov)
    return Xs, ruls, fails, events, hrs, groups, provs


def _build_domain_block(series_list, dataset, window, stride, n_keep, rng, min_cell, is_w3, min_real_fraction):
    all_X, all_rul, all_fail, all_evt, all_hr, all_grp, all_prov = [], [], [], [], [], [], []
    for ser in series_list:
        Xs, ruls, fails, events, hrs, groups, provs = _window_series(ser, window, stride)
        all_X.extend(Xs); all_rul.extend(ruls); all_fail.extend(fails)
        all_evt.extend(events); all_hr.extend(hrs); all_grp.extend(groups); all_prov.extend(provs)
    if not all_X:
        return {}
    X = np.stack(all_X).astype(np.float32)
    rul = np.asarray(all_rul, dtype=np.float32)
    fail = np.asarray(all_fail, dtype=np.int8)
    event = np.asarray(all_evt, dtype=np.int16)
    has_rul = np.asarray(all_hr, dtype=np.int8)
    group = np.asarray(all_grp, dtype=object)
    prov = np.asarray(all_prov, dtype=object)
    n_keep = min(n_keep, len(X))
    if is_w3:
        idx = sample_w3(event, prov, n_keep, rng, min_cell=min_cell, min_real_fraction=min_real_fraction)
    else:
        idx = stratified_sample(event.astype(object), n_keep, rng, min_cell=min_cell)
    return {"X": X[idx], "rul": rul[idx], "fail": fail[idx], "event": event[idx],
            "has_rul": has_rul[idx], "group": group[idx], "provenance": prov[idx]}


def build_corpus(cfg: Dict, raw_dir: Path, out_dir: Path) -> Dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["data"]
    window = int(dcfg.get("window", 50))
    stride = int(dcfg.get("stride", 1))
    min_cell = int(dcfg.get("min_cell", 30))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    found, missing = load_all(raw_dir, cfg)
    if not found:
        raise RuntimeError(f"No datasets could be loaded from {raw_dir}. Missing: {missing}")

    source_counts = dcfg.get("source_counts", {})
    target_counts = dcfg.get("target_counts", {})
    min_real = float(dcfg.get("w3_min_real_fraction", 0.70))
    rul_cap = float(dcfg.get("rul_cap", 130))
    aug = NigerianAugmentor(cfg["augmentation"], seed=seed)

    blocks = []
    meta_ds = {}
    source_row_offset = 0

    for name, series_list in found.items():
        n_src = int(source_counts.get(name, 1000))
        n_tgt = int(target_counts.get(name, n_src))
        blk = _build_domain_block(series_list, name, window, stride, n_src, rng, min_cell,
                                  is_w3=(name == "w3"), min_real_fraction=min_real)
        if not blk:
            meta_ds[name] = {"n_source": 0, "n_target": 0}
            continue

        X_src = np.nan_to_num(to_unified(blk["X"], name), nan=0.0).astype(np.float32)
        split_src = assign_splits(blk["group"], rng,
                                  val_frac=float(dcfg.get("split", {}).get("val", 0.10)),
                                  test_frac=float(dcfg.get("split", {}).get("test", 0.20)))
        group_to_split = {}
        for g, s in zip(blk["group"], split_src):
            group_to_split[str(g)] = int(s)
        n = len(X_src)
        has_rul0 = np.where(np.isnan(blk["rul"]), 0, blk["has_rul"]).astype(np.int8)
        blocks.append({
            "X": X_src, "rul": cap_rul(blk["rul"], rul_cap), "fail": blk["fail"].astype(np.float32),
            "event": blk["event"], "has_rul": has_rul0,
            "domain": np.zeros(n, dtype=np.int8), "split": split_src,
            "dataset": np.full(n, DATASET_ID[name], dtype=np.int8),
            "label_avail": np.ones(n, dtype=np.int8), "n_gaps": np.zeros(n, dtype=np.int16),
            "group": blk["group"],
            "base": np.full(n, -1, dtype=np.int64),
        })

        n_tgt = min(n_tgt, n)
        tgt_idx = rng.choice(n, size=n_tgt, replace=(n_tgt > n))
        X_t_list, gaps, applied_counts = [], [], {s: 0 for s in STRESSORS}
        for i in tgt_idx:
            x_aug, info = aug.apply(blk["X"][i].copy(), name, stressors=None, force=False)
            X_t_list.append(to_unified(np.nan_to_num(x_aug, nan=0.0).astype(np.float32), name))
            gaps.append(int(info.get("n_gaps", 0)))
            for s in info.get("applied", []):
                if s in applied_counts:
                    applied_counts[s] += 1
        rates = {s: applied_counts[s] / max(1, n_tgt) for s in STRESSORS}
        label_frac = float(dcfg.get("target_label_fraction", 0.30))
        label_avail1 = (rng.random(n_tgt) < label_frac).astype(np.int8)
        split_tgt = np.array([group_to_split[str(g)] for g in blk["group"][tgt_idx]], dtype=np.int8)
        blocks.append({
            "X": np.stack(X_t_list).astype(np.float32),
            "rul": cap_rul(blk["rul"][tgt_idx], rul_cap),
            "fail": blk["fail"][tgt_idx].astype(np.float32),
            "event": blk["event"][tgt_idx],
            "has_rul": np.where(np.isnan(blk["rul"][tgt_idx]), 0, blk["has_rul"][tgt_idx]).astype(np.int8),
            "domain": np.ones(n_tgt, dtype=np.int8), "split": split_tgt,
            "dataset": np.full(n_tgt, DATASET_ID[name], dtype=np.int8),
            "label_avail": label_avail1, "n_gaps": np.asarray(gaps, dtype=np.int16),
            "group": blk["group"][tgt_idx],
            "base": (source_row_offset + tgt_idx).astype(np.int64),
        })
        source_row_offset += n + n_tgt
        real_frac = float((blk["provenance"] == "real").mean()) if name == "w3" else None
        meta_ds[name] = {"n_source": int(n), "n_target": int(n_tgt),
                         "stressor_rates_target": rates, "real_fraction_source": real_frac,
                         "channels": [c for c, _ in CHANNELS[name]]}

    if not blocks:
        raise RuntimeError("No datasets could be loaded")

    def cat(key):
        return np.concatenate([b[key] for b in blocks], axis=0)

    corpus = {k: cat(k) for k in
              ["X", "rul", "fail", "event", "has_rul", "domain", "split", "dataset", "label_avail", "n_gaps", "group", "base"]}

    src_train = (corpus["domain"] == 0) & (corpus["split"] == 0)
    iqr_k = float(dcfg.get("iqr_k", 1.5))
    normalizers = {}
    X = corpus["X"].copy()
    for d in DATASETS:
        o, nch = OFFSETS[d], len(CHANNELS[d])
        rows_fit = np.flatnonzero(src_train & (corpus["dataset"] == DATASET_ID[d]))
        if len(rows_fit) == 0:
            rows_fit = np.flatnonzero(corpus["dataset"] == DATASET_ID[d])
        if len(rows_fit) == 0:
            continue
        sub = X[rows_fit][:, :, o:o + nch]
        nrm = Normalizer(iqr_k=iqr_k).fit(sub)
        normalizers[d] = nrm.to_dict()
        rows_all = np.flatnonzero(corpus["dataset"] == DATASET_ID[d])
        clipped = nrm.clip(X[rows_all][:, :, o:o + nch])
        X[np.ix_(rows_all, np.arange(X.shape[1]), np.arange(o, o + nch))] = nrm.transform(clipped)
        other = np.r_[np.arange(0, o), np.arange(o + nch, N_CHANNELS)]
        if len(other):
            X[np.ix_(rows_all, np.arange(X.shape[1]), other)] = 0.0
    corpus["X"] = X

    meta = {
        "n_total": int(len(corpus["X"])),
        "n_source": int((corpus["domain"] == 0).sum()),
        "n_target": int((corpus["domain"] == 1).sum()),
        "window": window, "stride": stride, "n_channels": N_CHANNELS, "seed": seed,
        "datasets": meta_ds, "missing": missing, "normalizers": normalizers,
        "split_counts": {str(s): int((corpus["split"] == s).sum()) for s in (0, 1, 2)},
    }
    np.savez_compressed(out_dir / "corpus.npz", **corpus)
    with open(out_dir / "corpus_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Corpus: n_total={meta['n_total']} source={meta['n_source']} target={meta['n_target']}")
    if missing:
        print("MISSING:", missing)
    return {"corpus": corpus, "meta": meta}


def load_corpus(processed_dir: Path) -> Dict:
    processed_dir = Path(processed_dir)
    data = np.load(processed_dir / "corpus.npz", allow_pickle=True)
    corpus = {k: data[k] for k in data.files}
    with open(processed_dir / "corpus_meta.json") as f:
        meta = json.load(f)
    return {"corpus": corpus, "meta": meta}
