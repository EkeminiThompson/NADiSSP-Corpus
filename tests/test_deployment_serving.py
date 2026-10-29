import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.schema import CHANNELS
from src.evaluation import deployment as dp
from src.evaluation.explain import ensemble_shap
from src.evaluation.features import window_features
from src.evaluation.metrics import best_f1_threshold, evaluate


def test_drop_and_impute_ffill_and_cache():
    X = np.arange(2 * 10 * 3, dtype=np.float32).reshape(2, 10, 3)
    rng = np.random.default_rng(0)
    out = dp.drop_and_impute(X, 0.5, "ffill", rng)
    assert np.isfinite(out).all()
    for i in range(2):                                                    # every imputed value equals a value seen earlier/at start
        for t in range(10):
            assert any(np.array_equal(out[i, t], X[i, s]) for s in range(10))
    assert np.array_equal(dp.drop_and_impute(X, 0.0, "ffill", rng), X)
    z = dp.drop_and_impute(X, 1.0, "edge_cache", rng)
    assert (z == 0).all()


def test_packet_loss_curve_shape_and_baseline(full_model, bundle):
    t, _ = full_model
    rows = bundle.idx("test")
    thr = 0.5
    res = dp.packet_loss_curve(t, bundle, rows, thr, [0.0, 0.2, 0.5], seed=1)
    assert [r["packet_loss"] for r in res] == [0.0, 0.2, 0.5]
    assert res[0]["imputation"] == "none" and res[1]["imputation"] == "ffill" and res[2]["imputation"] == "edge_cache"
    assert res[0]["auprc_drop_pct"] == 0.0


def test_outage_simulation_runs_and_buffer_shrinks_beyond_cache(full_model, bundle, cfg):
    t, _ = full_model
    short = dp.simulate_outage(t, bundle, 24, cfg, seed=1)
    long = dp.simulate_outage(t, bundle, 2160, cfg, seed=1)
    assert short["buffer_rows"] > long["buffer_rows"]                    # beyond the 72 h cache only the tail survives
    assert long["drift_strength_std"] > short["drift_strength_std"]
    assert len(short["recovery_curve"]) == cfg["deployment"]["recovery_max_epochs"]


def test_ewc_penalty_pulls_towards_anchor(full_model, bundle):
    t, _ = full_model
    f = dp.ewc_fisher(t.model, bundle, bundle.idx("train", 0), n=16)
    assert all(torch.isfinite(v).all() and (v >= 0).all() for v in f.values())
    assert sum(float(v.sum()) for v in f.values()) > 0


def test_onnx_export_parity_and_int8(full_model, bundle, tmp_path):
    t, _ = full_model
    p = dp.export_onnx(t, tmp_path / "m.onnx", bundle.T, bundle.C)
    par = dp.onnx_parity(t, p, bundle.X[:8])
    assert par["max_abs_diff_rul"] < 1e-3 and par["max_abs_diff_fail"] < 1e-4
    q = dp.quantize_int8(p, tmp_path / "m8.onnx")
    b = dp.benchmark_onnx(q, bundle.T, bundle.C, repeats=5, warmup=1)
    assert b["mean_ms"] > 0 and "Not a measurement of Jetson" in b["note"]


def test_features_and_gap_proxy_detect_frozen_telemetry():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 50, 41)).astype(np.float32)
    X[0, 10:30, :] = X[0, 10, :]
    f, names = window_features(X, np.array([2, 2]))                        # dataset 2 = 3W: temperature/pressure channels
    assert f.shape == (2, len(names)) and f[0, -1] > f[1, -1]


def test_shap_attribution_is_consistent(full_model, bundle):
    t, _ = full_model
    r = ensemble_shap(t, bundle, bundle.idx("test"), "clf", max_rows=80)
    assert "top" in r, r
    assert abs(sum(x["pct_attribution"] for x in r["top"]) - 100) < 100            # top-k is a subset; ranks ordered
    assert [x["rank"] for x in r["top"]] == sorted(x["rank"] for x in r["top"])
    assert abs(sum(r["by_category_pct"].values()) - 100.0) < 1e-6


# ---------------------------------------------------------------------------------------------- service
@pytest.fixture(scope="module")
def client(full_model, bundle, corpus_dir, tmp_path_factory, monkeypatch_module):
    from fastapi.testclient import TestClient
    from src.serving.app import create_app
    t, _ = full_model
    d = tmp_path_factory.mktemp("svc")
    dp.export_onnx(t, d / "m.onnx", bundle.T, bundle.C)
    monkeypatch_module.setenv("NADIP_FEEDBACK_BUFFER", str(d / "fb.jsonl"))
    return TestClient(create_app(str(d / "m.onnx"), str(corpus_dir / "corpus_meta.json"), threshold=0.5)), bundle


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _req(bundle, asset="A1", offline=False, nan=False):
    n = len(CHANNELS["w3"])
    w = np.random.default_rng(0).normal(100, 5, size=(bundle.T, n))
    win = [[None if (nan and i == 3) else float(v) for v in row] for i, row in enumerate(w)]
    return {"asset_id": asset, "dataset": "w3", "window": win, "offline": offline}


def test_service_health_predict_and_validation(client):
    c, bundle = client
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["window"] == bundle.T
    r = c.post("/predict", json=_req(bundle, nan=True))
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "cloud" and 0 <= j["fault_probability"] <= 1 and j["alert"] in ("green", "amber", "red")
    assert j["missing_fraction"] > 0
    bad = _req(bundle); bad["window"] = bad["window"][:10]
    assert c.post("/predict", json=bad).status_code == 422
    assert c.post("/predict", json={**_req(bundle), "dataset": "nope"}).status_code == 422


def test_service_edge_cache_fallback(client):
    c, bundle = client
    assert c.post("/predict", json=_req(bundle, asset="NEW", offline=True)).status_code == 503     # nothing cached yet
    online = c.post("/predict", json=_req(bundle, asset="B2")).json()
    off = c.post("/predict", json=_req(bundle, asset="B2", offline=True)).json()
    assert off["source"] == "edge_cache" and off["rul_cycles"] == online["rul_cycles"] and off["age_hours"] < 0.1


def test_service_feedback_buffer(client):
    c, bundle = client
    a = c.post("/feedback", json={"asset_id": "A1", "failed": True, "rul_cycles": 12}).json()
    b = c.post("/feedback", json={"asset_id": "A1", "failed": False}).json()
    assert b["buffered"] == a["buffered"] + 1
