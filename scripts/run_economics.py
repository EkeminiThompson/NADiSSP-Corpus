#!/usr/bin/env python
"""Monte-Carlo TCO/NPV (Table 4.9). The incident reduction is tied to the AUPRC measured by scripts/evaluate.py."""
import argparse
import json
import _boot  # noqa: F401
from src.config import add_common_args, load_config, resolve_path
from src.evaluation.economic import run_monte_carlo, sensitivity
from src.pipeline import dump

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--auprc", type=float, default=None, help="override; default: results/evaluation.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    res = resolve_path(cfg, "results_dir")
    auprc = a.auprc
    if auprc is None and cfg["economics"].get("lambda_nadip_override") is None:
        auprc = json.load(open(res / "evaluation.json"))["full_nadip"]["overall"]["auprc"]
    s = run_monte_carlo(cfg["economics"], auprc, cfg["seed"], out_dir=res)
    s["sensitivity_npv_musd"] = sensitivity(cfg["economics"], seed=cfg["seed"])
    dump(s, res / "economics_summary.json")
    d = s["delta"]
    print(f"incident model: {s['incident_model']['how']}  -> lambda_NADIP={s['incident_model']['lambda_nadip']:.2f}")
    print(f"reactive TCO ${s['reactive']['annual_tco_musd']['mean']:.1f}M/yr, NADIP ${s['nadip']['annual_tco_musd']['mean']:.1f}M/yr "
          f"({d['pct_tco_reduction']:.1f}% lower); 5-yr NPV ${d['npv_musd']['mean']:.1f}M +/- {d['npv_musd']['sd']:.1f}M "
          f"(deployment capex included: ${cfg['economics']['deployment_capex_musd']}M)")
