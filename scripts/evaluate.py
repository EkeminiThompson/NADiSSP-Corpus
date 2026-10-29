#!/usr/bin/env python
"""Final evaluation of the full model (seed 0) -> results/evaluation.json

Objective 1  input-space shift + pre/post embedding shift (source-only transformer vs NADIP)
Objective 3  AUPRC / RMSE / F1 on the held-out test split, SHAP attribution of the XGBoost ensemble
"""
import argparse
import numpy as np
import _boot  # noqa: F401
from src.config import add_common_args, load_config, resolve_path
from src.evaluation.diagnostics import domain_report, input_space_report, tsne_embed
from src.evaluation.explain import ensemble_shap
from src.evaluation.metrics import evaluate
from src.pipeline import dump, get_bundle, seeds_for
from src.training.runner import train_variant
from src.training.variants import VARIANTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    bundle, meta = get_bundle(cfg)
    seed = a.seed if a.seed is not None else seeds_for(cfg)[0]
    ck = resolve_path(cfg, "checkpoint_dir")
    full = train_variant(cfg, bundle, meta, VARIANTS["full_nadip"], seed, ckpt_root=ck)
    src = train_variant(cfg, bundle, meta, VARIANTS["source_only_transformer"], seed, ckpt_root=ck)
    res = {"seed": seed,
           "corpus": {k: meta[k] for k in ("n_total", "n_source", "n_target", "split_counts", "shortfalls", "missing_datasets")},
           "full_nadip": evaluate(full, bundle, "test"), "source_only_transformer": evaluate(src, bundle, "test"),
           "input_space_shift": input_space_report(bundle, cfg), "embedding_shift": domain_report(src, full, bundle, cfg),
           "shap_classifier": ensemble_shap(full, bundle, bundle.idx("test"), "clf"),
           "shap_regressor": ensemble_shap(full, bundle, bundle.idx("test"), "reg"),
           "stage_history": {k: (h[-1] if h else None) for k, h in full.history.items()},
           "targets_not_results": cfg["evaluation"]["targets"]}
    out = resolve_path(cfg, "results_dir")
    dump(res, out / "evaluation.json")
    fig = out / "figures_data"
    fig.mkdir(parents=True, exist_ok=True)
    rows = bundle.idx("test")
    for label, t in (("pre", src), ("post", full)):                                  # arrays for Fig. 4.1
        o = t.predict(bundle.X[rows], bundle.dataset[rows])
        s0 = bundle.domain[rows] == 0
        z = (o["z"] - o["z"][s0].mean(0)) / (o["z"][s0].std(0) + 1e-8)
        emb, idx = tsne_embed(z, seed)
        np.savez_compressed(fig / f"tsne_{label}.npz", emb=emb, domain=bundle.domain[rows][idx],
                            rul=bundle.rul[rows].numpy()[idx], dataset=bundle.dataset[rows][idx])
    o = res["full_nadip"]["overall"]
    print(f"test: AUPRC={o['auprc']:.3f}  RUL RMSE={o['rul_rmse']:.2f}  F1={o['f1']:.3f}   (targets: {cfg['evaluation']['targets']})")
