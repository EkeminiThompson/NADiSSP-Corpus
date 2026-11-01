"""ESPset as published: frequency-domain spectra + features, two honest uses."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.corpus import build_corpus, load_corpus
from src.data.espset_spectral import band_pool, fault_labels, load_espset_bands, read_spectra
from src.data.loaders import DatasetNotFound, load_espset
from src.espset_experiment import run
from tests.fixtures import make_espset_spectra
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def esp_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("esp")
    make_espset_spectra(root, np.random.default_rng(0))
    return root


def test_reader_shapes_alignment_and_labels(esp_root):
    S, F = read_spectra(esp_root)
    assert S.shape == (18 * 8, 12103) and len(F) == len(S)
    lab = fault_labels(F)
    assert lab["classes"][0] == "normal" and set(lab["classes"]) == {"normal", "unbalance", "misalignment"}
    assert lab["fail"].sum() == (F["label"] != "normal").sum()


def test_unknown_normal_label_stops_with_list_of_labels(esp_root):
    _, F = read_spectra(esp_root)
    F["label"] = F["label"].map({"normal": "A", "unbalance": "B", "misalignment": "C"})
    with pytest.raises(DatasetNotFound, match="normal_labels"):
        fault_labels(F)


def test_misaligned_files_are_rejected(tmp_path):
    make_espset_spectra(tmp_path, np.random.default_rng(1), n_pumps=3, per_pump=2)
    f = pd.read_csv(tmp_path / "ESPset" / "features.csv").iloc[:-1]
    f.to_csv(tmp_path / "ESPset" / "features.csv", index=False)
    with pytest.raises(DatasetNotFound, match="rows"):
        read_spectra(tmp_path)


def test_band_pool_locates_the_1x_harmonic(esp_root):
    S, _ = read_spectra(esp_root)
    mean, mx = band_pool(S[:4], 50)
    assert mean.shape == (4, 50) and (mx >= mean - 1e-6).all()
    assert int(mx[0].argmax()) == int((3002 / 12103) * 50)          # 1X sits in its own band


def test_default_corpus_still_refuses_spectra(esp_root):
    with pytest.raises(DatasetNotFound, match="frequency-domain"):
        load_espset(esp_root, {"espset": {"mode": "off"}})


def test_separate_experiment_runs_group_wise_and_learns(esp_root, tmp_path):
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"))
    e = cfg["espset"]["experiment"]
    e["n_seeds"], e["pool_bins"] = 2, 242
    e["xgb"] = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3}
    e["cnn"] = {**e["cnn"], "epochs": 4, "batch_size": 16, "channels": 8, "patience": 2}
    res = run(cfg, esp_root, tmp_path / "out", log=lambda *a: None)
    assert (tmp_path / "out" / "tables.md").exists() and res["summary"]["n_seeds"] == 2
    for m in ("features_xgb", "spectrum_xgb", "spectrum_cnn"):
        s = res["summary"]["models"][m]["auprc"]
        assert np.isfinite(s["mean"])
    # the planted harmonic structure is easy: tree models must beat the ~0.67 fault prevalence clearly
    assert res["summary"]["models"]["features_xgb"]["auprc"]["mean"] > 0.8


def test_bands_mode_enters_unified_corpus_without_time_stressors(esp_root, tmp_path):
    cfg = load_config(str(ROOT / "configs" / "smoke.yaml"),
                      ["data.datasets=[espset]", "espset.mode=bands", "data.stride=1", "data.min_cell=2",
                       "data.source_counts.espset=100", "data.target_counts.espset=0"])
    cfg["augmentation"]["matrix"]["espset"] = []
    build_corpus(cfg, esp_root, tmp_path / "proc")
    c = load_corpus(tmp_path / "proc"); c = c.get("corpus", c)
    assert (c["domain"] == 0).all() and len(c["X"]) > 0                # no Nigerian-augmented target for ESPset
    assert c["flags"].sum() == 0
    g, sp = c["group"], c["split"]
    assert not (set(g[sp == 0]) & set(g[sp == 2])) and not (set(g[sp == 0]) & set(g[sp == 1]))
    assert set(np.unique(c["fail"])) <= {0.0, 1.0} and c["fail"].max() == 1.0
    ser = load_espset_bands(esp_root, {"n_bands": 50}, 50)
    assert ser[0].X.shape == (50, 8) and np.isnan(ser[0].X[:, 2:]).all() and ser[0].meta["axis"] == "frequency"
