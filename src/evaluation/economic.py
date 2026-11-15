"""Monte-Carlo TCO / NPV (§4.4.4, Tables 4.9a / 4.9b).

Inputs are exactly the Table 4.9a distributions (see configs/nadip.yaml -> economics). Everything that
Table 4.9a does not give is an explicit, named assumption:

* NADIP incident rate: lambda_nadip = lambda_reactive * (1 - reduction), reduction = efficacy * AUPRC.
  `intervention_efficacy` is the share of detected-fault risk that maintenance actually converts into an
  avoided major incident. It is an ASSUMPTION, not an estimate; efficacy = 0.70 maps AUPRC 0.85 to
  lambda ~ 1.9, the value the document uses. With a measured AUPRC, the economics follow from it.
* Deployment cost: the document gives none, so it defaults to 0 (see deployment_capex_musd /
  deployment_opex_musd_per_year). NPV below is therefore GROSS savings unless you set them.
* Incident counts are drawn independently per year; cost parameters once per run.

Nothing here is calibrated to NNPC/NUPRC data (Limitation 6): treat results as scenario analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def lognormal_params(mean: float, sd: float):
    s2 = np.log(1.0 + (sd / mean) ** 2)
    return float(np.log(mean) - s2 / 2.0), float(np.sqrt(s2))


def incident_reduction(auprc: Optional[float], efficacy: float) -> float:
    if auprc is None or np.isnan(auprc):
        raise ValueError("AUPRC is required for the incident mapping (or set economics.lambda_nadip_override)")
    return float(np.clip(efficacy * auprc, 0.0, 0.95))


def run_monte_carlo(eco: Dict, auprc: Optional[float] = None, seed: int = 42, reduction_override: Optional[float] = None,
                    out_dir: Optional[Path] = None, save_runs: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    n, H, r = int(eco["n_runs"]), int(eco["horizon_years"]), float(eco["discount_rate"])
    lam_r = float(eco["lambda_reactive"])
    if reduction_override is not None:
        lam_n, red, how = lam_r * (1 - reduction_override), reduction_override, "reduction_override"
    elif eco.get("lambda_nadip_override") is not None:
        lam_n = float(eco["lambda_nadip_override"]); red = 1 - lam_n / lam_r; how = "lambda_nadip_override"
    else:
        red = incident_reduction(auprc, float(eco["intervention_efficacy"])); lam_n = lam_r * (1 - red)
        how = f"efficacy({eco['intervention_efficacy']}) x AUPRC({auprc:.3f})"

    mu, sg = lognormal_params(eco["downtime_cost_mean_usd"], eco["downtime_cost_sd_usd"])
    cost = rng.lognormal(mu, sg, size=n)                                   # USD per major incident
    prem_lo, prem_hi = eco["spare_premium_range"]
    prem = rng.uniform(prem_lo, prem_hi, size=n)
    prem_ref = (prem_lo + prem_hi) / 2.0
    labour = np.clip(rng.normal(eco["labour_rate_mean"], eco["labour_rate_sd"], size=n), 200, 900)
    inc_r = rng.poisson(lam_r, size=(n, H))
    inc_n = rng.poisson(lam_n, size=(n, H))

    lab_f = labour / eco["labour_rate_mean"]
    spare_f = (1 + prem) / (1 + prem_ref)
    tco_r = inc_r * cost[:, None] + 1e6 * (eco["baseline_labour_musd"] * lab_f + eco["baseline_spares_musd"] * spare_f)[:, None]
    tco_n = inc_n * cost[:, None] + 1e6 * (eco["nadip_labour_musd"] * lab_f + eco["nadip_spares_musd"] * spare_f)[:, None]
    disc = (1 + r) ** -np.arange(1, H + 1)
    capex = 1e6 * eco.get("deployment_capex_musd", 0.0)
    opex = 1e6 * eco.get("deployment_opex_musd_per_year", 0.0)
    savings = tco_r - tco_n                                                # (n, H)
    npv = (savings * disc).sum(1) - capex - opex * disc.sum()
    npv_gross = (savings * disc).sum(1)

    s = lambda a: {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)),
                   "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}
    m = lambda a: a / 1e6
    summary = {
        "n_runs": n, "horizon_years": H, "discount_rate": r, "seed": seed,
        "incident_model": {"how": how, "auprc": auprc, "reduction": red, "lambda_reactive": lam_r, "lambda_nadip": lam_n},
        "reactive": {"annual_tco_musd": s(m(tco_r.mean(1))), "mean_incidents_per_year": float(inc_r.mean())},
        "nadip": {"annual_tco_musd": s(m(tco_n.mean(1))), "mean_incidents_per_year": float(inc_n.mean())},
        "delta": {"annual_savings_musd": s(m(savings.mean(1))),
                  "pct_tco_reduction": float(100 * (1 - tco_n.mean() / tco_r.mean())),
                  "npv_musd": s(m(npv)), "npv_gross_musd": s(m(npv_gross)),
                  "p_npv_positive": float((npv > 0).mean())},
        "assumptions": {"deployment_capex_musd": eco.get("deployment_capex_musd", 0.0),
                        "deployment_opex_musd_per_year": eco.get("deployment_opex_musd_per_year", 0.0),
                        "intervention_efficacy": eco["intervention_efficacy"]},
        "expected_annual_downtime_musd": {"reactive": lam_r * eco["downtime_cost_mean_usd"] / 1e6,
                                          "nadip": lam_n * eco["downtime_cost_mean_usd"] / 1e6},
        "note": "Literature-derived scenario parameters (Limitation 6); not calibrated to operator data. "
                "NPV is gross of deployment cost unless the assumptions above are non-zero.",
    }
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "economics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if save_runs:
            np.savez_compressed(out_dir / "economics_runs.npz", tco_reactive=tco_r.mean(1), tco_nadip=tco_n.mean(1), npv=npv)
    return summary


def sensitivity(eco: Dict, reductions=(0.2, 0.4, 0.6), seed: int = 42) -> Dict:
    """NPV for fixed incident-reduction scenarios (the document's "40% reduction" check, §4.4.4)."""
    return {f"{int(x * 100)}%": run_monte_carlo(eco, reduction_override=x, seed=seed, save_runs=False)["delta"]["npv_musd"]
            for x in reductions}
