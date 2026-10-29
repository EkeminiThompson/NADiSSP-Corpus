import numpy as np
import pytest

from src.evaluation.diagnostics import mmd, mmd_permutation_p, sinkhorn_w1, w1_per_dim
from src.evaluation.economic import incident_reduction, lognormal_params, run_monte_carlo, sensitivity
from src.evaluation.metrics import auprc, best_f1_threshold, f1_at, rul_rmse
from src.evaluation.stats import paired_comparison, summarise


def test_metrics_basic():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(auprc(y, p) - 0.8333333) < 1e-5
    assert np.isnan(auprc(np.zeros(4), p))                                  # undefined, not 0
    assert rul_rmse(np.array([10., 20.]), np.array([12., 20.]), np.array([True, True])) == pytest.approx(np.sqrt(2))
    assert np.isnan(rul_rmse(np.array([1.]), np.array([2.]), np.array([False])))
    t = best_f1_threshold(y, p)
    assert f1_at(y, p, t) >= f1_at(y, p, 0.9)


def test_mmd_and_w1_detect_shift_and_not_noise():
    rng = np.random.default_rng(0)
    A, B, C = rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), rng.normal(size=(300, 4)) + 1.5
    assert mmd(A, C) > 4 * mmd(A, B)
    assert sinkhorn_w1(A, C) > 1.5 * sinkhorn_w1(A, B)
    assert w1_per_dim(A, C) > 5 * w1_per_dim(A, B)
    assert mmd_permutation_p(A, C, n_perm=40) < 0.05 < mmd_permutation_p(A, B, n_perm=40)


def test_sinkhorn_equals_known_1d_shift():
    x = np.linspace(-1, 1, 200)[:, None]
    assert sinkhorn_w1(x, x + 0.5, eps=0.01) == pytest.approx(0.5, abs=0.05)   # W1 of a pure translation = shift


def test_paired_test_and_effect_size():
    a, b = [0.80, 0.82, 0.79, 0.81, 0.83], [0.50, 0.55, 0.52, 0.51, 0.49]
    r = paired_comparison(a, b)
    assert r["p"] < 1e-3 and r["significant_0.05"] and r["cohens_dz"] > 5 and r["df"] == 4
    assert paired_comparison([1, 2], [1, 2])["mean_diff"] == 0
    s = summarise([1, 2, 3, 4])
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]


ECO = dict(n_runs=20000, horizon_years=5, discount_rate=0.05, downtime_cost_mean_usd=8.4e6, downtime_cost_sd_usd=1.6e6,
           lambda_reactive=4.7, spare_premium_range=[0.25, 0.45], labour_rate_mean=485, labour_rate_sd=72,
           baseline_labour_musd=42.1, baseline_spares_musd=26.8, nadip_labour_musd=31.4, nadip_spares_musd=17.3,
           intervention_efficacy=0.7, lambda_nadip_override=None, deployment_capex_musd=0.0, deployment_opex_musd_per_year=0.0)


def test_lognormal_parameterisation_hits_mean_and_sd():
    mu, s = lognormal_params(8.4e6, 1.6e6)
    x = np.random.default_rng(0).lognormal(mu, s, 400000)
    assert x.mean() == pytest.approx(8.4e6, rel=0.01) and x.std() == pytest.approx(1.6e6, rel=0.02)


def test_economics_reproducible_and_closed_form():
    a = run_monte_carlo(ECO, auprc=0.85, seed=1, save_runs=False)
    b = run_monte_carlo(ECO, auprc=0.85, seed=1, save_runs=False)
    assert a["delta"]["npv_musd"]["mean"] == b["delta"]["npv_musd"]["mean"]
    lam_r, lam_n = 4.7, a["incident_model"]["lambda_nadip"]
    annuity = sum(1.05 ** -t for t in range(1, 6))
    expected = annuity * ((lam_r - lam_n) * 8.4 + (42.1 - 31.4) + (26.8 - 17.3))
    assert a["delta"]["npv_musd"]["mean"] == pytest.approx(expected, rel=0.03)
    # reactive annual TCO follows from Table 4.9a inputs, NOT from Table 4.9b's $456M
    assert a["reactive"]["annual_tco_musd"]["mean"] == pytest.approx(4.7 * 8.4 + 42.1 + 26.8, rel=0.02)


def test_deployment_cost_reduces_npv_exactly():
    base = run_monte_carlo(ECO, auprc=0.85, seed=1, save_runs=False)["delta"]["npv_musd"]["mean"]
    withc = run_monte_carlo({**ECO, "deployment_capex_musd": 10.0}, auprc=0.85, seed=1, save_runs=False)["delta"]["npv_musd"]["mean"]
    assert base - withc == pytest.approx(10.0, abs=1e-6)


def test_incident_mapping_follows_measured_auprc():
    assert incident_reduction(0.85, 0.7) == pytest.approx(0.595)
    lo = run_monte_carlo(ECO, auprc=0.5, seed=1, save_runs=False)["delta"]["npv_musd"]["mean"]
    hi = run_monte_carlo(ECO, auprc=0.9, seed=1, save_runs=False)["delta"]["npv_musd"]["mean"]
    assert hi > lo
    with pytest.raises(ValueError):
        run_monte_carlo(ECO, auprc=None)
    s = sensitivity({**ECO, "n_runs": 3000})
    assert s["20%"]["mean"] < s["40%"]["mean"] < s["60%"]["mean"]
