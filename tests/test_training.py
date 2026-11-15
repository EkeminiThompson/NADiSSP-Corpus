import json

import numpy as np
import torch

from src.pipeline import run_one
from src.training.runner import train_variant
from src.training.variants import ABLATIONS, TABLE_4_3, VARIANTS


def test_variants_cover_tables():
    for n in TABLE_4_3 + ABLATIONS:
        assert n in VARIANTS
    assert not VARIANTS["no_grl"].use_grl and VARIANTS["no_grl"].use_simclr and VARIANTS["no_grl"].use_weibull
    assert not VARIANTS["no_simclr"].use_simclr and VARIANTS["no_simclr"].use_grl
    assert not VARIANTS["no_weibull"].use_weibull and VARIANTS["no_weibull"].use_grl
    assert VARIANTS["no_3w_espset"].datasets == ("cmapss", "ai4i")
    assert not VARIANTS["source_only_transformer"].uses_target


def test_full_model_trains_and_predicts(full_model, bundle):
    t, _ = full_model
    rows = bundle.idx("test")
    out = t.predict(bundle.X[rows], bundle.dataset[rows])
    for k in ("rul", "fail_prob", "z"):
        assert np.isfinite(out[k]).all()
    assert out["fail_prob"].min() >= 0 and out["fail_prob"].max() <= 1
    assert out["rul"].min() >= 0 and out["rul"].max() <= 130
    h = t.history["stage3_joint"]
    assert {"task_rul", "task_cls", "domain", "contrastive", "survival", "disc_acc"} <= set(h[-1])
    assert h[-1]["domain"] > 0 and h[-1]["contrastive"] > 0 and h[-1]["survival"] != 0
    assert h[0]["grl_lambda"] < h[-1]["grl_lambda"]                                    # Ganin ramp is increasing
    assert not t.info["pretrained"]                                                    # no Kinetics-400 weights are used


def test_target_rows_never_in_task_loss_by_default(cfg, bundle, meta, tmp_path):
    assert cfg["data"]["target_label_fraction"] == 0.30 and cfg["data"]["target_label_in_task_loss"] is False
    off = train_variant(cfg, bundle, meta, VARIANTS["full_nadip"], 0, ckpt_root=tmp_path / "a", log=lambda *a: None)
    c_on = {**cfg, "data": {**cfg["data"], "target_label_in_task_loss": True}}
    on = train_variant(c_on, bundle, meta, VARIANTS["no_xgb"], 0, ckpt_root=tmp_path / "b", log=lambda *a: None)
    n_lab_t = int(bundle.label_avail[bundle.idx("train", 1)].sum())
    assert n_lab_t > 0
    assert off.info["n_supervised_train"] == off.info["n_source_train"]                       # default: source labels only
    assert on.info["n_source_train"] == off.info["n_source_train"]
    assert on.info["n_supervised_train"] == off.info["n_source_train"] + n_lab_t              # option adds the retained 30%


def test_resume_reproduces_predictions(cfg, bundle, meta, full_model):
    t, ck = full_model
    t2 = train_variant(cfg, bundle, meta, VARIANTS["full_nadip"], 0, ckpt_root=ck, log=lambda *a: None)      # all stages resumed
    rows = bundle.idx("test")[:64]
    a = t.predict(bundle.X[rows], bundle.dataset[rows])
    b = t2.predict(bundle.X[rows], bundle.dataset[rows])
    assert np.allclose(a["rul"], b["rul"], atol=1e-4) and np.allclose(a["fail_prob"], b["fail_prob"], atol=1e-5)


def test_dataset_restricted_variant_trains_on_fewer_rows(cfg, bundle, meta, tmp_path):
    t = train_variant(cfg, bundle, meta, VARIANTS["no_3w_espset"], 0, ckpt_root=tmp_path, log=lambda *a: None)
    full_rows = len(bundle.idx("train", 0))
    assert t.info["n_source_train"] < full_rows
    assert t.history["stage3_joint"]


def test_no_adaptation_variant_has_no_domain_or_contrastive_loss(cfg, bundle, meta, tmp_path):
    t = train_variant(cfg, bundle, meta, VARIANTS["source_only_transformer"], 0, ckpt_root=tmp_path, log=lambda *a: None)
    last = t.history["stage3_joint"][-1]
    assert last["domain"] == 0 and last["contrastive"] == 0 and last["grl_lambda"] == 0
    assert t.info["n_target_train"] == 0


def test_no_pretrain_is_not_applicable_without_checkpoint(cfg, bundle, meta, tmp_path, monkeypatch):
    c = dict(cfg); c["_root"] = str(tmp_path)
    c["paths"] = {**cfg["paths"], "results_dir": str(tmp_path / "r"), "checkpoint_dir": str(tmp_path / "c")}
    rec = run_one(c, bundle, meta, "no_pretrain", 0)
    assert rec["status"] == "not_applicable" and "pretrained_init" in rec["reason"]


def test_pretrained_init_loads_compatible_weights(cfg, bundle, meta, tmp_path):
    from src.models.nadip import NADIP
    src = NADIP(bundle.C, cfg["model"], 130.0)
    p = tmp_path / "init.pt"
    torch.save({"model_state": src.state_dict()}, p)
    c = {**cfg, "model": {**cfg["model"], "pretrained_init": str(p)}}
    t = train_variant(c, bundle, meta, VARIANTS["no_3w_espset"], 0, ckpt_root=tmp_path / "ck", log=lambda *a: None)
    assert t.info["pretrained"] and t.info["pretrained_report"]["loaded"] > 10


def test_baselines_run(cfg, bundle, meta):
    for n in ("source_only_xgboost", "source_only_lstm"):
        t = train_variant(cfg, bundle, meta, VARIANTS[n], 0, log=lambda *a: None)
        rows = bundle.idx("test")
        o = t.predict(bundle.X[rows], bundle.dataset[rows])
        assert o["rul"].shape == (len(rows),) and np.isfinite(o["fail_prob"]).all()


def test_variant_whose_datasets_are_absent_is_not_applicable(cfg, bundle, meta, tmp_path):
    """e.g. no_3w_espset when only 3W was available: report not_applicable instead of crashing."""
    c = dict(cfg); c["_root"] = str(tmp_path)
    c["paths"] = {**cfg["paths"], "results_dir": str(tmp_path / "r"), "checkpoint_dir": str(tmp_path / "c")}
    m = {**meta, "datasets": {"w3": meta["datasets"]["w3"]}}
    rec = run_one(c, bundle, m, "no_3w_espset", 0)
    assert rec["status"] == "not_applicable" and "not" not in rec.get("error", "")
