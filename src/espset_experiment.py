"""Separate ESPset experiment: fault diagnosis from the published frequency-domain data.

This is NOT part of the time-series transfer study (CMAPSS / AI4I / 3W). ESPset ships spectra and engineered
features only, so the question here is different: how well do (a) the provided summary features, (b) the spectrum
itself with a tree model, and (c) a 1-D CNN over the spectrum separate normal pumps from faulty ones?

Protocol (every number is produced by running this):
* repeated *group-wise* splits by esp_id -- spectra from one pump never appear in two of train/val/test;
* the fault-probability threshold is chosen on the validation split only and applied to test;
* metrics: AUPRC and F1 for fault-vs-normal, accuracy and macro-F1 over the fault classes;
* n_seeds repeated splits + model seeds, paired t-test / Cohen's d_z / 95% CI between models (same splits).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .data.espset_spectral import band_pool, fault_labels, read_spectra
from .data.sampling import assign_splits
from .evaluation.metrics import auprc, best_f1_threshold, f1_at
from .evaluation.stats import paired_comparison, summarise

MODELS = ["features_xgb", "spectrum_xgb", "spectrum_cnn"]
FEATURE_EXCLUDE = {"esp_id", "label"}


def _split(groups: np.ndarray, y_fail: np.ndarray, seed: int, val: float, test: float, tries: int = 100) -> np.ndarray:
    """Group-wise split in which train, val and test each contain both normal and faulty spectra."""
    for k in range(tries):
        s = assign_splits(groups, np.random.default_rng(seed * 1000 + k), test, val)
        if all(len(np.unique(y_fail[s == i])) == 2 for i in (0, 1, 2)):
            return s
    raise RuntimeError(f"seed {seed}: could not find a group-wise split with both classes in train/val/test "
                       f"({len(np.unique(groups))} pumps). Check labels and espset.normal_labels.")


def _full_proba(proba: np.ndarray, present: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((len(proba), K), dtype=np.float64)
    out[:, present] = proba
    return out


def _xgb(Xtr, ytr, Xte_list, K, params, seed):
    from xgboost import XGBClassifier
    present = np.unique(ytr)
    remap = {c: i for i, c in enumerate(present)}
    yt = np.array([remap[c] for c in ytr])
    m = XGBClassifier(objective="multi:softprob" if len(present) > 2 else "binary:logistic",
                      random_state=seed, n_jobs=2, tree_method="hist", verbosity=0, **params)
    m.fit(Xtr, yt)
    outs = []
    for X in Xte_list:
        p = m.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 1:
            p = np.hstack([1 - p, p])
        outs.append(_full_proba(p, present, K))
    return outs


class _CNN(nn.Module):
    def __init__(self, k: int, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, c, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(c, 2 * c, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(2 * c, 2 * c, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(2 * c, k))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def _cnn(Xtr, ytr, Xva, yva, Xte_list, K, cfg, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    f = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32))
    xtr, xva = f(Xtr), f(Xva)
    ytr_t, yva_t = torch.from_numpy(ytr.astype(np.int64)), torch.from_numpy(yva.astype(np.int64))
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (K * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    model = _CNN(K, int(cfg["channels"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    lossf = nn.CrossEntropyLoss(weight=w)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    bs = int(cfg["batch_size"])
    for _ in range(int(cfg["epochs"])):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(); lossf(model(xtr[b]), ytr_t[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(xva), yva_t))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return [torch.softmax(model(f(X)), dim=1).numpy().astype(np.float64) for X in Xte_list]


def _score(y_class: np.ndarray, proba_val: np.ndarray, y_val: np.ndarray, proba: np.ndarray, K: int) -> Dict:
    fail_val, fail = (y_val > 0).astype(int), (y_class > 0).astype(int)
    p_val, p = 1.0 - proba_val[:, 0], 1.0 - proba[:, 0]
    thr = best_f1_threshold(fail_val, p_val)
    pred = proba.argmax(1)
    labels = sorted(set(y_class.tolist()) | set(pred.tolist()))
    fault_only = [c for c in labels if c > 0]
    return {"auprc": auprc(fail, p), "f1": f1_at(fail, p, thr), "threshold": thr,
            "accuracy": float(accuracy_score(y_class, pred)),
            "macro_f1": float(f1_score(y_class, pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1_faults": float(f1_score(y_class, pred, labels=fault_only, average="macro", zero_division=0))
            if fault_only else float("nan"),
            "fail_prevalence": float(fail.mean()), "n": int(len(y_class))}


def run_seed(spectra, feats, lab, pooled, seed: int, ecfg: Dict) -> Dict:
    y, K = lab["y"].astype(int), len(lab["classes"])
    groups = feats["esp_id"].astype(str).to_numpy()
    sp = _split(groups, lab["fail"], seed, ecfg["split"]["val"], ecfg["split"]["test"])
    tr, va, te = (np.flatnonzero(sp == i) for i in (0, 1, 2))
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va])) \
        and not (set(groups[va]) & set(groups[te])), "pump leaked across splits"
    num = feats.drop(columns=[c for c in feats.columns if c in FEATURE_EXCLUDE]).apply(pd.to_numeric, errors="coerce")
    F = num.to_numpy(dtype=np.float32)
    out: Dict[str, Dict] = {}
    t0 = time.time()
    pv, pt = _xgb(F[tr], y[tr], [F[va], F[te]], K, ecfg["xgb"], seed)
    out["features_xgb"] = _score(y[te], pv, y[va], pt, K)
    pv, pt = _xgb(pooled[tr], y[tr], [pooled[va], pooled[te]], K, ecfg["xgb"], seed)
    out["spectrum_xgb"] = _score(y[te], pv, y[va], pt, K)
    pv, pt = _cnn(pooled[tr], y[tr], pooled[va], y[va], [pooled[va], pooled[te]], K, ecfg["cnn"], seed)
    out["spectrum_cnn"] = _score(y[te], pv, y[va], pt, K)
    return {"seed": seed, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "n_pumps": {"train": int(len(set(groups[tr]))), "val": int(len(set(groups[va]))), "test": int(len(set(groups[te])))},
            "seconds": time.time() - t0, "models": out}


def aggregate(runs: List[Dict]) -> Dict:
    summ: Dict[str, Dict] = {}
    for m in MODELS:
        summ[m] = {k: summarise([r["models"][m][k] for r in runs]) for k in ("auprc", "f1", "accuracy", "macro_f1", "macro_f1_faults")}
    comps = {}
    for a, b in (("spectrum_xgb", "features_xgb"), ("spectrum_cnn", "features_xgb"), ("spectrum_cnn", "spectrum_xgb")):
        comps[f"{a}_vs_{b}"] = {k: paired_comparison([r["models"][a][k] for r in runs], [r["models"][b][k] for r in runs])
                                for k in ("auprc", "macro_f1")}
    return {"n_seeds": len(runs), "models": summ, "paired": comps}


def markdown(summary: Dict, meta: Dict) -> str:
    f = lambda s: "n/a" if not s or s.get("n", 0) == 0 or np.isnan(s["mean"]) else f"{s['mean']:.3f} ± {s['sd']:.3f}"
    lines = ["### ESPset frequency-domain fault diagnosis (test split, group-wise by esp_id)", "",
             f"{meta['n_spectra']} spectra, {meta['n_pumps']} pumps, classes: {', '.join(meta['classes'])}; "
             f"{summary['n_seeds']} repeated splits. Not a time-series result.", "",
             "| Model | AUPRC | F1 | Accuracy | Macro-F1 | Macro-F1 (faults) |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        s = summary["models"][m]
        lines.append(f"| {m} | {f(s['auprc'])} | {f(s['f1'])} | {f(s['accuracy'])} | {f(s['macro_f1'])} | {f(s['macro_f1_faults'])} |")
    lines += ["", "Paired comparisons (AUPRC):", ""]
    for k, v in summary["paired"].items():
        p = v["auprc"]
        lines.append(f"* {k}: " + ("; ".join(f"{a}={b:.4g}" for a, b in p.items() if isinstance(b, (int, float))) if p else "n/a"))
    return "\n".join(lines)


def run(cfg: Dict, raw_dir: Path, out_dir: Path, seeds: Optional[int] = None, log: Callable = print) -> Dict:
    ecfg = cfg["espset"]["experiment"]
    spectra, feats = read_spectra(raw_dir)
    lab = fault_labels(feats, cfg["espset"].get("normal_labels"))
    pooled = band_pool(spectra, int(ecfg["pool_bins"]))[0]
    meta = {"n_spectra": int(len(spectra)), "n_bins": int(spectra.shape[1]), "n_pumps": int(feats["esp_id"].nunique()),
            "classes": lab["classes"], "class_counts": {c: int((lab["y"] == i).sum()) for i, c in enumerate(lab["classes"])},
            "fail_prevalence": float(lab["fail"].mean()), "feature_columns": [c for c in feats.columns if c not in FEATURE_EXCLUDE]}
    log(f"ESPset: {meta['n_spectra']} spectra x {meta['n_bins']} bins, {meta['n_pumps']} pumps, classes {meta['class_counts']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict] = []
    for s in range(int(seeds or ecfg["n_seeds"])):
        p = out_dir / f"seed{s}.json"
        if p.exists():
            runs.append(json.load(open(p))); continue
        r = run_seed(spectra, feats, lab, pooled, 42 + s, ecfg)
        json.dump(r, open(p, "w"), indent=1)
        runs.append(r)
        log(f"[espset seed {42 + s}] " + "  ".join(f"{m}: AUPRC {r['models'][m]['auprc']:.3f}" for m in MODELS))
    summary = aggregate(runs)
    json.dump({"meta": meta, "summary": summary}, open(out_dir / "summary.json", "w"), indent=1, default=float)
    (out_dir / "tables.md").write_text(markdown(summary, meta))
    return {"meta": meta, "summary": summary}
