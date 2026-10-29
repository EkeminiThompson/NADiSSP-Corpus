#!/usr/bin/env python
"""Objective 4 (Tables 3.20 / 4.10-4.12): packet-loss curve, outage + EWC recovery, ONNX export and latency.

Writes results/deployment.json and models/nadip.onnx (+ nadip_int8.onnx).
"""
import argparse
import _boot  # noqa: F401
from src.config import ROOT, add_common_args, load_config, resolve_path
from src.evaluation import deployment as dp
from src.evaluation.metrics import best_f1_threshold
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
    t = train_variant(cfg, bundle, meta, VARIANTS["full_nadip"], seed, ckpt_root=resolve_path(cfg, "checkpoint_dir"))
    dep = cfg["deployment"]
    val = bundle.idx("val")
    thr = best_f1_threshold(bundle.fail[val].numpy(), t.predict(bundle.X[val], bundle.dataset[val])["fail_prob"])
    rows = bundle.idx("test")
    res = {"seed": seed, "threshold_from_validation": thr}
    res["packet_loss"] = dp.packet_loss_curve(t, bundle, rows, thr, dep["packet_loss_rates"], cfg["seed"], dep["edge_cache_above_loss"])
    res["outage_recovery"] = {k: dp.simulate_outage(t, bundle, h, cfg, cfg["seed"]) for k, h in dep["outage_windows"].items()}
    out = ROOT / "models"
    onnx = dp.export_onnx(t, out / "nadip.onnx", bundle.T, bundle.C)
    int8 = dp.quantize_int8(onnx, out / "nadip_int8.onnx")
    res["onnx"] = {"parity_fp32": dp.onnx_parity(t, onnx, bundle.X[:16]),
                   "latency_fp32": dp.benchmark_onnx(onnx, bundle.T, bundle.C, dep["latency_repeats"]),
                   "latency_int8": dp.benchmark_onnx(int8, bundle.T, bundle.C, dep["latency_repeats"]),
                   "auprc_fp32": dp.onnx_auprc(onnx, bundle, rows), "auprc_int8": dp.onnx_auprc(int8, bundle, rows)}
    dump(res, resolve_path(cfg, "results_dir") / "deployment.json")
    for r in res["packet_loss"]:
        print(f"loss {r['packet_loss']:.0%} [{r['imputation']}]: AUPRC={r['auprc']:.3f}  RMSE={r['rul_rmse']:.2f}")
    print("latency (this host):", {k: round(res['onnx']['latency_fp32'][k], 2) for k in ('mean_ms', 'p95_ms')})
