"""
Monte-Carlo Total Cost of Ownership (TCO) / Net Present Value (NPV)
simulation for Nigerian refinery context (Chapter 4, Tables 4.9a/b, Figure 4.6).

All monetary values in USD. Parameters are literature-derived and carry
the ±35 % uncertainty band stated in the dissertation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def run_monte_carlo_tco(
    n_runs: int = 10_000,
    horizon_years: int = 5,
    discount_rate: float = 0.05,
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> Dict:
    """
    Compare reactive baseline vs NADIP-enabled PdM for one typical Nigerian refinery.
    """
    rng = np.random.default_rng(seed)
    out_dir = out_dir or Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    downtime_cost = rng.lognormal(mean=np.log(8.4e6), sigma=0.18, size=n_runs)
    incidents_reactive = rng.poisson(lam=4.7, size=n_runs)
    incidents_nadip = rng.poisson(lam=1.9, size=n_runs)
    spare_premium = rng.uniform(0.25, 0.45, size=n_runs)
    labour_rate = rng.normal(485, 72, size=n_runs).clip(200, 900)

    annual_downtime_r = incidents_reactive * downtime_cost
    annual_labour_r = 42.1e6 * (labour_rate / 485)
    annual_spares_r = 26.8e6 * (1 + spare_premium)
    annual_tco_r = annual_downtime_r + annual_labour_r + annual_spares_r

    annual_downtime_n = incidents_nadip * downtime_cost
    annual_labour_n = 31.4e6 * (labour_rate / 485)
    annual_spares_n = 17.3e6 * (1 + 0.5 * spare_premium)
    annual_tco_n = annual_downtime_n + annual_labour_n + annual_spares_n

    savings = annual_tco_r - annual_tco_n
    discount_factors = np.array([(1 + discount_rate) ** -t for t in range(1, horizon_years + 1)])
    npv = savings * discount_factors.sum()

    summary = {
        "n_runs": n_runs,
        "horizon_years": horizon_years,
        "discount_rate": discount_rate,
        "reactive": {
            "mean_annual_tco_musd": float(annual_tco_r.mean() / 1e6),
            "std_annual_tco_musd": float(annual_tco_r.std() / 1e6),
            "mean_incidents": float(incidents_reactive.mean()),
        },
        "nadip": {
            "mean_annual_tco_musd": float(annual_tco_n.mean() / 1e6),
            "std_annual_tco_musd": float(annual_tco_n.std() / 1e6),
            "mean_incidents": float(incidents_nadip.mean()),
        },
        "delta": {
            "mean_annual_savings_musd": float(savings.mean() / 1e6),
            "mean_5yr_npv_musd": float(npv.mean() / 1e6),
            "std_5yr_npv_musd": float(npv.std() / 1e6),
            "pct_tco_reduction": float(100 * (1 - annual_tco_n.mean() / annual_tco_r.mean())),
        },
        "note": (
            "Literature-derived parameters; ±35 % uncertainty band applies. "
            "See dissertation Section 4.4.4 and Limitation 6."
        ),
    }

    dist = pd.DataFrame({
        "annual_tco_reactive": annual_tco_r,
        "annual_tco_nadip": annual_tco_n,
        "annual_savings": savings,
        "npv_5yr": npv,
    })
    dist_path = out_dir / "monte_carlo_tco_runs.csv"
    dist.to_csv(dist_path, index=False)

    summary_path = out_dir / "monte_carlo_tco_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Monte-Carlo TCO/NPV complete")
    print(f"  Mean 5-yr NPV (NADIP): ${summary['delta']['mean_5yr_npv_musd']:.1f} M")
    print(f"  Mean annual TCO reduction: {summary['delta']['pct_tco_reduction']:.1f} %")
    print(f"  Distributions → {dist_path}")
    print(f"  Summary       → {summary_path}")
    return summary
