"""Orchestration shared by the scripts: corpus, multi-seed runs, ablation tables, final evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import resolve_path
from .data import build_corpus, load_corpus
from .evaluation.metrics import evaluate
from .evaluation.stats import paired_comparison, summarise
from .training.data import Bundle
from .training.runner import Trained, train_variant
from .training.variants import ABLATIONS, TABLE_4_3, VARIANTS


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def prepare(cfg: Dict, raw_dir: Optional[Path] = None) -> Dict:
    raw = Path(raw_dir) if raw_dir else resolve_path(cfg, "raw_dir")
    return build_corpus(cfg, raw, resolve_path(cfg, "processed_dir"))


def get_bundle(cfg: Dict):
    r = load_corpus(resolve_path(cfg, "processed_dir"))
    return Bundle(r["corpus"], r["meta"]), r["meta"]


def seeds_for(cfg: Dict, n: Optional[int] = None) -> List[int]:
    return [int(cfg["seed"]) + i for i in range(int(n or cfg["n_seeds"]))]


def run_one(cfg: Dict, bundle: Bundle, meta: Dict, variant_name: str, seed: int, force: bool = False, log=print) -> Dict:
    """Train + evaluate one (variant, seed). Result JSON is cached under results/runs/."""
    res_dir = resolve_path(cfg, "results_dir") / "runs" / variant_name
    path = res_dir / f"seed{seed}.json"
    if path.exists() and not force:
        return json.load(open(path))
    v = VARIANTS[variant_name]
    if variant_name == "no_pretrain" and not cfg["model"].get("pretrained_init"):
        rec = {"variant": variant_name, "seed": seed, "status": "not_applicable",
               "reason": "model.pretrained_init is not set, so the full model is already randomly initialised; "
                         "this ablation would be identical to full_nadip."}
        dump(rec, path)
        return rec
    t0 = time.time()
    t = train_variant(cfg, bundle, meta, v, seed, ckpt_root=resolve_path(cfg, "checkpoint_dir"), force=force, log=log)
    rec = {"variant": variant_name, "seed": seed, "status": "ok", "train_seconds": time.time() - t0, "info": t.info,
           "test": evaluate(t, bundle, "test"), "val": evaluate(t, bundle, "val")["overall"],
           "history_tail": {k: (h[-1] if h else None) for k, h in t.history.items()}}
    dump(rec, path)
    return rec


def aggregate(runs: Dict[str, List[Dict]]) -> Dict:
    """Mean / SD / 95% CI across seeds + paired tests against the full model and the source-only transformer."""
    metrics = ("auprc", "f1", "rul_rmse")
    out: Dict = {"variants": {}, "paired": {}}
    for name, recs in runs.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        if not ok:
            out["variants"][name] = {"status": recs[0].get("status", "missing"), "reason": recs[0].get("reason")}
            continue
        v: Dict = {"status": "ok", "n_seeds": len(ok)}
        for scope, getter in (("overall", lambda r: r["test"]["overall"]),
                              ("target", lambda r: r["test"]["by_domain"].get("target", {})),
                              ("source", lambda r: r["test"]["by_domain"].get("source", {}))):
            v[scope] = {m: summarise([getter(r).get(m, np.nan) for r in ok]) for m in metrics}
        v["by_dataset"] = {d: {m: summarise([r["test"]["by_dataset"].get(d, {}).get(m, np.nan) for r in ok]) for m in metrics}
                           for d in ("cmapss", "ai4i", "w3", "espset")}
        v["train_seconds_mean"] = float(np.mean([r["train_seconds"] for r in ok]))
        out["variants"][name] = v
    by_seed = lambda name, m: {r["seed"]: r["test"]["overall"].get(m, np.nan) for r in runs.get(name, []) if r.get("status") == "ok"}
    for ref in ("full_nadip", "source_only_transformer"):
        for name in runs:
            if name == ref or ref not in runs:
                continue
            for m in ("auprc", "rul_rmse", "f1"):
                a, b = by_seed(ref, m), by_seed(name, m)
                common = sorted(set(a) & set(b))
                if len(common) >= 2:
                    out["paired"].setdefault(f"{ref}_vs_{name}", {})[m] = paired_comparison([a[s] for s in common], [b[s] for s in common])
    return out


def markdown_table(agg: Dict, names: List[str], title: str) -> str:
    f = lambda s: "n/a" if s is None or np.isnan(s["mean"]) else f"{s['mean']:.3f} ± {s['sd']:.3f}"
    lines = [f"### {title}", "", "| Variant | AUPRC | RUL RMSE (cycles) | F1 | seeds |", "|---|---|---|---|---|"]
    for n in names:
        v = agg["variants"].get(n)
        if not v or v.get("status") != "ok":
            lines.append(f"| {n} | — | — | — | {v.get('status', 'not run') if v else 'not run'} |")
            continue
        o = v["overall"]
        lines.append(f"| {n} | {f(o['auprc'])} | {f(o['rul_rmse'])} | {f(o['f1'])} | {v['n_seeds']} |")
    return "\n".join(lines) + "\n"


def run_ablation(cfg: Dict, variants: Optional[List[str]] = None, n_seeds: Optional[int] = None, force: bool = False,
                 log=print) -> Dict:
    bundle, meta = get_bundle(cfg)
    names = variants or sorted(set(TABLE_4_3 + ABLATIONS + ["source_only_lstm"]))
    if "full_nadip" not in names:
        names = ["full_nadip"] + names
    runs: Dict[str, List[Dict]] = {n: [] for n in names}
    for n in names:
        for s in seeds_for(cfg, n_seeds):
            log(f"=== {n} | seed {s}")
            runs[n].append(run_one(cfg, bundle, meta, n, s, force, log))
    agg = aggregate(runs)
    res = resolve_path(cfg, "results_dir")
    dump(agg, res / "ablation_results.json")
    (res / "tables.md").write_text(markdown_table(agg, [n for n in TABLE_4_3 if n in names], "Table 4.3 analogue (test split)")
                                   + "\n" + markdown_table(agg, ["full_nadip"] + [n for n in ABLATIONS if n in names],
                                                           "Table 4.4 analogue (test split)"))
    return agg
