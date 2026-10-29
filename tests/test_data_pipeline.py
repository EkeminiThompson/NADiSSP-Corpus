import numpy as np

from src.data.preprocess import Normalizer, cap_rul
from src.data.sampling import assign_splits, sample_w3, stratified_sample
from src.data.schema import DATASET_ID, DATASETS, N_CHANNELS


def test_normalizer_zscore_and_iqr_clip():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(200, 50, 3)).astype(np.float32)
    X[0, 0, 0] = 1000.0                                                  # outlier
    n = Normalizer(1.5).fit(X)
    Z = n.transform(n.clip(X))
    assert Z.max() < 20                                                  # clipped, not 1000/2
    inl = n.transform(n.clip(X))[1:]
    assert abs(inl.mean()) < 0.1 and abs(inl.std() - 1) < 0.15
    back = Normalizer.from_dict(n.to_dict()).transform(n.clip(X))
    assert np.allclose(back, Z)


def test_rul_cap():
    assert cap_rul(np.array([5, 130, 300.0]), 130).tolist() == [5, 130, 130]


def test_group_splits_never_share_a_group():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(30)], 20).astype(object)
    split = assign_splits(groups, rng, 0.2, 0.1)
    for g in np.unique(groups):
        assert len(set(split[groups == g])) == 1
    assert {0, 1, 2} <= set(split)
    assert 0.1 < (split == 2).mean() < 0.35


def test_too_few_groups_stay_in_train():
    s = assign_splits(np.array(["a"] * 10 + ["b"] * 10, dtype=object), np.random.default_rng(0), 0.2, 0.1)
    assert (s == 0).all()


def test_stratified_sample_respects_min_cell_and_size():
    rng = np.random.default_rng(0)
    strata = np.array(["a"] * 900 + ["b"] * 90 + ["c"] * 10)
    idx = stratified_sample(strata, 100, rng, min_cell=5)
    assert len(idx) == 100 and len(set(idx)) == 100
    picked = strata[idx]
    assert (picked == "c").sum() >= 5 and (picked == "b").sum() >= 5


def test_w3_real_fraction_at_least_70_percent():
    rng = np.random.default_rng(0)
    event = np.repeat([0, 1, 2, 3], 100)
    prov = np.where(np.arange(400) % 2 == 0, "real", "simulated")
    idx = sample_w3(event, prov, 120, rng, min_cell=5, min_real_fraction=0.7)
    for e in range(4):
        sel = idx[event[idx] == e]
        assert (prov[sel] == "real").mean() >= 0.7 - 1e-9


def test_corpus_invariants(bundle, meta, cfg):
    c = bundle.c
    assert not np.isnan(c["X"]).any() and c["X"].shape[-1] == N_CHANNELS
    assert set(np.unique(c["split"])) == {0, 1, 2}
    # no group appears in two splits (within a domain pair)
    for g in np.unique(c["group"]):
        assert len(set(c["split"][c["group"] == g])) == 1, f"{g} leaks across splits"
    tgt = np.flatnonzero(c["domain"] == 1)
    base = c["base"][tgt]
    assert (base >= 0).all() and (c["domain"][base] == 0).all()
    assert (c["dataset"][base] == c["dataset"][tgt]).all()
    assert (c["split"][base] == c["split"][tgt]).all(), "target variant landed in a different split than its clean base"
    assert c["label_avail"][c["domain"] == 0].all()
    frac = c["label_avail"][tgt].mean()
    assert abs(frac - cfg["data"]["target_label_fraction"]) < 0.08
    # a dataset's windows only occupy its own channel slots
    from src.data.schema import CHANNELS, OFFSETS
    for d in DATASETS:
        rows = c["dataset"] == DATASET_ID[d]
        o, n = OFFSETS[d], len(CHANNELS[d])
        other = np.ones(N_CHANNELS, bool); other[o:o + n] = False
        assert (c["X"][rows][:, :, other] == 0).all()
    assert meta["n_total"] == len(c["X"])


def test_only_matrix_stressors_fired_per_dataset(meta):
    rates = {d: v["stressor_rates_target"] for d, v in meta["datasets"].items()}
    assert rates["cmapss"]["sabotage"] == 0 and rates["cmapss"]["wave"] == 0
    assert rates["ai4i"]["corrosion"] == 0 and rates["ai4i"]["wave"] == 0
    assert rates["w3"]["sabotage"] == 0 and rates["w3"]["corrosion"] == 0 and rates["w3"]["sampling"] == 0
    assert rates["espset"]["sabotage"] == 0 and rates["espset"]["wave"] == 0 and rates["espset"]["sampling"] == 0


def test_loaders_skip_drawn_3w_and_flag_pseudo_sequences(raw_dir):
    from src.data.loaders import load_ai4i, load_w3
    series = load_w3(raw_dir, 60, 50)
    assert {s.meta["provenance"] for s in series} == {"real", "simulated"}
    assert not any("DRAWN" in s.unit for s in series)
    ai = load_ai4i(raw_dir)
    assert all(s.meta["pseudo_sequence"] and not s.has_rul for s in ai)


def test_missing_dataset_is_reported_not_fabricated(cfg, tmp_path):
    import pytest
    from src.data import build_corpus
    with pytest.raises(RuntimeError, match="No datasets could be loaded"):
        build_corpus(cfg, tmp_path / "empty", tmp_path / "out")


def test_w3_real_format_class9_and_unlabelled_rows(tmp_path):
    """Real 3W v2.0.0: classes run 0..9 (+100 transients) and class can be <NA> for unlabelled rows."""
    import numpy as np
    import pandas as pd
    from src.data.loaders import load_w3
    from src.data.schema import channel_names

    d = tmp_path / "3W" / "dataset" / "9"
    d.mkdir(parents=True)
    n = 600
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 7)), columns=channel_names("w3"))
    cls = np.r_[np.zeros(250), np.full(30, 109), np.full(320, 9)].astype("float")
    cls[:40] = np.nan                                    # unlabelled head must not be treated as normal
    df["class"] = pd.array(cls, dtype="Int16")
    df["state"] = pd.array(np.zeros(n), dtype="Int16")
    df.to_parquet(d / "WELL-00001_20170101000000.parquet")
    ser = load_w3(tmp_path, 200, 50)
    assert len(ser) == 1
    s = ser[0]
    assert s.fail.max() == 1                              # class 9 counts as an undesirable event
    assert set(np.unique(s.event_class)) <= set(range(10))
    assert s.meta["provenance"] == "real"
