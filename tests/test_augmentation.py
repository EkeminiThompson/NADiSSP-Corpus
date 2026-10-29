import numpy as np
import pytest

from src.config import load_config
from src.data.augmentation import NigerianAugmentor
from src.data.schema import CHANNELS, DATASETS, STRESSORS, STRESSOR_ROLES, role_indices

CFG = load_config()
# Table 3.9b, transcribed independently of the config under test
TABLE_3_9B = {
    "cmapss": {"humidity", "dropout", "corrosion", "sampling"},
    "ai4i": {"humidity", "dropout", "sabotage", "sampling"},
    "w3": {"humidity", "dropout", "wave"},
    "espset": {"humidity", "dropout", "corrosion"},
}


def window(dataset, seed=0, T=50):
    rng = np.random.default_rng(seed)
    n = len(CHANNELS[dataset])
    return (rng.normal(size=(T, n)) * 2 + 10).astype(np.float32)


def test_matrix_matches_table_3_9b():
    a = NigerianAugmentor(CFG["augmentation"], 0)
    for d in DATASETS:
        assert set(a.applicable(d)) == TABLE_3_9B[d]


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_applicable_stressor_changes_its_target_channels_and_only_those(dataset):
    a = NigerianAugmentor(CFG["augmentation"], 1)
    X = window(dataset)
    for s in a.applicable(dataset):
        changed_any = False
        for trial in range(40):                                  # stochastic stressors: at least one trial must fire
            out, info = a.apply(X, dataset, stressors=[s], force=True)
            assert info["applied"] == [s]
            diff = ~np.isclose(out, X, atol=1e-6)
            if s == "sampling":
                changed_any |= diff.any()
                continue
            target = set(role_indices(dataset, STRESSOR_ROLES[s]).tolist())
            if s == "corrosion":                                  # corrosion also attenuates vibration channels
                target |= set(role_indices(dataset, ["vibration"]).tolist())
            touched = set(np.flatnonzero(diff.any(0)).tolist())
            assert touched <= target, f"{dataset}/{s} touched channels outside its roles: {touched - target}"
            changed_any |= bool(touched)
        assert changed_any, f"{s} never changed {dataset} -- the stressor does not reach any channel"


def test_stressors_outside_the_matrix_never_applied():
    a = NigerianAugmentor(CFG["augmentation"], 2)
    for d in DATASETS:
        for s in set(STRESSORS) - TABLE_3_9B[d]:
            out, info = a.apply(window(d), d, stressors=[s], force=True)
            assert info["applied"] == []
            assert np.array_equal(out, window(d))


def test_dropout_fraction_and_forward_fill():
    cfg = CFG["augmentation"]
    a = NigerianAugmentor(cfg, 3)
    X = window("w3", T=200)
    out, info = a.apply(X, "w3", stressors=["dropout"], force=True)
    cols = role_indices("w3", ["temperature", "pressure"])
    frozen = (np.diff(out[:, cols[0]]) == 0).mean()              # forward-filled steps repeat the previous value
    lo, hi = cfg["dropout"]["fraction_range"]["w3"]
    assert lo - 0.08 <= frozen <= hi + 0.08
    assert info["n_gaps"] >= 1
    assert np.isfinite(out).all()                                # forward-fill imputation leaves no NaN


def test_sabotage_amplitude_is_3_to_8_std():
    a = NigerianAugmentor(CFG["augmentation"], 4)
    X = window("ai4i", T=400)
    out, _ = a.apply(X, "ai4i", stressors=["sabotage"], force=True)
    j = role_indices("ai4i", ["electrical"])[0]
    spikes = np.abs(out[:, j] - X[:, j]) / X[:, j].std()
    spikes = spikes[spikes > 1e-6]
    assert len(spikes) > 0 and spikes.min() >= 2.99 and spikes.max() <= 8.01


def test_wave_frequency_in_range():
    a = NigerianAugmentor(CFG["augmentation"], 5)
    X = np.full((256, len(CHANNELS["w3"])), 100.0, dtype=np.float32)
    out, _ = a.apply(X, "w3", stressors=["wave"], force=True)
    j = role_indices("w3", ["pressure"])[0]
    sig = out[:, j] - 100.0
    f = np.fft.rfftfreq(256, d=1.0)[np.argmax(np.abs(np.fft.rfft(sig))[1:]) + 1]
    assert 0.08 <= f <= 0.52
    assert abs(np.abs(sig).max() - CFG["augmentation"]["wave"]["amplitude"] * 100.0) < 1.0   # +/-8% of baseline


def test_deterministic_given_seed():
    X = window("cmapss")
    o1, _ = NigerianAugmentor(CFG["augmentation"], 9).apply(X, "cmapss")
    o2, _ = NigerianAugmentor(CFG["augmentation"], 9).apply(X, "cmapss")
    assert np.array_equal(o1, o2)
