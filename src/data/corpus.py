"""Build and load the experimental corpus (windows, domains, splits, meta)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .augmentation import NigerianAugmentor
from .loaders import Series, load_all
from .preprocess import Normalizer, cap_rul
from .sampling import assign_splits, sample_w3, stratified_sample
from .schema import CHANNELS, DATASET_ID, DATASETS, N_CHANNELS, OFFSETS, to_unified


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
    label_keep = float(dcfg.get("target_label_keep_frac", 0.30))
    aug = NigerianAugmentor(cfg.get("augmentation", {}), seed=seed)

    blocks = []
    meta_ds = {}

    for name, series_list in found.items():
        n_src = int(source_counts.get(name, 1000))
        n_tgt = int(target_counts.get(name, n_src))
        blk = _build_domain_block(series_list, name, window, stride, n_src + n_tgt, rng, min_cell,
                                  is_w3=(name == "w3"), min_real_fraction=min_real)
        if not blk:
            meta_ds[name] = {"n_source": 0, "n_target": 0}
            continue

        X_u = np.nan_to_num(to_unified(blk["X"], name), nan=0.0).astype(np.float32)
        rul = cap_rul(blk["rul"], rul_cap)
        fail = blk["fail"].astype(np.int8)
        event = blk["event"]
        has_rul = blk["has_rul"].astype(np.int8)
        group = blk["group"]
        prov = blk["provenance"]

        # Source / target partition by index
        n = len(X_u)
        n_s = min(n_src, n)
        idx = np.arange(n)
        rng.shuffle(idx)
        src_idx, tgt_idx = idx[:n_s], idx[n_s:n_s + n_tgt]
        if len(tgt_idx) == 0 and n > n_s:
            tgt_idx = idx[n_s:]

        def pack(sel, domain):
            if len(sel) == 0:
                return None
            X = X_u[sel]
            if domain == 1:
                X, rates = aug.apply_batch(X, name)
            else:
                rates = {}
            split = assign_splits(group[sel], rng,
                                  val_frac=float(dcfg.get("split", {}).get("val", 0.2)),
                                  test_frac=float(dcfg.get("split", {}).get("test", 0.1)))
            label_avail = np.ones(len(sel), dtype=np.int8)
            if domain == 1:
                keep = rng.random(len(sel)) < label_keep
                label_avail = keep.astype(np.int8)
            return {
                "X": X, "rul": rul[sel], "fail": fail[sel], "event": event[sel],
                "has_rul": has_rul[sel] * label_avail, "has_fail": label_avail,
                "domain": np.full(len(sel), domain, dtype=np.int8),
                "split": split, "dataset": np.full(len(sel), DATASET_ID[name], dtype=np.int8),
                "label_avail": label_avail, "group": group[sel], "provenance": prov[sel],
                "rates": rates,
            }

        src_pack = pack(src_idx, 0)
        tgt_pack = pack(tgt_idx, 1)
        for p in (src_pack, tgt_pack):
            if p is not None:
                blocks.append(p)

        real_frac = None
        if name == "w3" and len(src_idx):
            real_frac = float(np.mean(prov[src_idx] == "real"))
        meta_ds[name] = {
            "n_source": int(len(src_idx)), "n_target": int(len(tgt_idx)),
            "real_fraction_source": real_frac,
            "channels": [c for c, _ in CHANNELS[name]],
        }

    if not blocks:
        raise RuntimeError("No windows produced")

    def cat(key, default=None):
        parts = [b[key] for b in blocks if key in b]
        if not parts:
            return default
        return np.concatenate(parts, axis=0)

    corpus = {
        "X": cat("X"),
        "rul": cat("rul"),
        "fail": cat("fail"),
        "event": cat("event"),
        "has_rul": cat("has_rul"),
        "domain": cat("domain"),
        "split": cat("split"),
        "dataset": cat("dataset"),
        "label_avail": cat("label_avail"),
        "group": cat("group"),
    }
    # has_fail for masked BCE
    if any("has_fail" in b for b in blocks):
        corpus["has_fail"] = cat("has_fail")
    else:
        corpus["has_fail"] = corpus["label_avail"]

    # Per-dataset normalizer on clean source train only
    normalizers = {}
    X = corpus["X"].copy()
    src_train = (corpus["domain"] == 0) & (corpus["split"] == 0)
    for d in DATASETS:
        o, nch = OFFSETS[d], len(CHANNELS[d])
        rows_fit = np.flatnonzero(src_train & (corpus["dataset"] == DATASET_ID[d]))
        if len(rows_fit) == 0:
            rows_fit = np.flatnonzero(corpus["dataset"] == DATASET_ID[d])
        if len(rows_fit) == 0:
            continue
        sub = X[rows_fit][:, :, o:o + nch]
        nrm = Normalizer().fit(sub)
        normalizers[d] = nrm.state_dict()
        rows_all = np.flatnonzero(corpus["dataset"] == DATASET_ID[d])
        X[np.ix_(rows_all, np.arange(X.shape[1]), np.arange(o, o + nch))] = nrm.transform(
            X[rows_all][:, :, o:o + nch])
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
    np.savez_compressed(out_dir / "corpus.npz", **{k: v for k, v in corpus.items() if k != "group"})
    # groups as object array saved separately
    np.save(out_dir / "corpus_groups.npy", corpus["group"], allow_pickle=True)
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
    gpath = processed_dir / "corpus_groups.npy"
    if gpath.exists():
        corpus["group"] = np.load(gpath, allow_pickle=True)
    with open(processed_dir / "corpus_meta.json") as f:
        meta = json.load(f)
    return {"corpus": corpus, "meta": meta}
