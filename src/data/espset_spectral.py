"""ESPset as it is actually published (NINFA-UFES/ESPset): frequency-domain data, no time axis.

    spectrum.csv   6,032 rows x 12,103 columns, ';'-separated, no header. One row = the velocity spectrum of one
                   vibration signal, normalised by the pump rotation frequency X: column 3003 (1-based) is 1X,
                   6006 is 2X, so the frequency resolution is X/3003.
    features.csv   6,032 rows + header: esp_id, label, median(8,13), median(98,102), a, b, peak1x, peak2x,
                   rms(98,102). Row i describes spectrum row i.

Nothing here pretends the spectra are time series. Two uses are provided, and the paper must say which one a
result comes from:

* `read_spectra` + `scripts/espset_experiment.py`  -- a separate fault-classification experiment.
* `load_espset_bands` -- each spectrum becomes a 50-step *frequency sweep* (band-pooled log amplitude) so it can
  enter the unified model. The steps are frequency bands, not time, so no time-domain stressor is applied to it.

Label names are not assumed. Which labels mean "normal" is set by `espset.normal_labels`; if none of the labels in
the file match, loading stops and lists the labels found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .loaders import DatasetNotFound, Series
from .schema import channel_names

ONE_X_INDEX = 3003              # 1-based column of the 1X harmonic in spectrum.csv
DEFAULT_NORMAL = ["normal", "healthy", "ok", "good", "none", "0"]


def find_files(raw_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    root = Path(raw_dir) / "ESPset"
    spec = sorted(root.rglob("spectrum.csv"))
    feat = sorted(root.rglob("features.csv"))
    return (spec[0] if spec else None), (feat[0] if feat else None)


def has_published_files(raw_dir: Path) -> bool:
    s, f = find_files(raw_dir)
    return s is not None and f is not None


def read_spectra(raw_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """(spectra float32 [N, F], features DataFrame with N rows). Raises DatasetNotFound if absent or misaligned."""
    s_path, f_path = find_files(raw_dir)
    if s_path is None or f_path is None:
        raise DatasetNotFound(f"spectrum.csv and features.csv not both found under {Path(raw_dir) / 'ESPset'}.")
    feats = pd.read_csv(f_path)
    for need in ("esp_id", "label"):
        if need not in feats.columns:
            raise DatasetNotFound(f"{f_path.name} has no '{need}' column (found {list(feats.columns)}).")
    spectra = pd.read_csv(s_path, sep=";", header=None, dtype=np.float32, engine="c").to_numpy(dtype=np.float32)
    if len(spectra) != len(feats):
        raise DatasetNotFound(f"spectrum.csv has {len(spectra)} rows but features.csv has {len(feats)}; "
                              "row i of one must describe row i of the other.")
    if not np.isfinite(spectra).all():
        spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
    return spectra, feats


def fault_labels(feats: pd.DataFrame, normal_labels: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Integer class per row (0 = normal, 1..K fault classes in sorted name order) and a binary fault flag."""
    names = feats["label"].astype(str).str.strip()
    wanted = {str(x).strip().lower() for x in (normal_labels or DEFAULT_NORMAL)}
    lower = names.str.lower()
    is_normal = lower.isin(wanted).to_numpy()
    if not is_normal.any():
        raise DatasetNotFound(
            f"None of the labels in features.csv {sorted(names.unique().tolist())} matches the normal labels "
            f"{sorted(wanted)}. Set espset.normal_labels in the config to the label(s) that mean 'no fault'.")
    fault_names = sorted(set(names[~is_normal]))
    code = {n: i + 1 for i, n in enumerate(fault_names)}
    y = np.where(is_normal, 0, names.map(code).fillna(0)).astype(np.int16)
    return {"y": y, "fail": (~is_normal).astype(np.int8), "classes": ["normal"] + fault_names}


def band_pool(spectra: np.ndarray, n_bands: int) -> Tuple[np.ndarray, np.ndarray]:
    """Log1p amplitude pooled into n_bands equal-width frequency bands: (mean, max), each [N, n_bands]."""
    n, f = spectra.shape
    edges = np.linspace(0, f, n_bands + 1).astype(int)
    logs = np.log1p(np.maximum(spectra, 0.0))
    mean = np.empty((n, n_bands), dtype=np.float32)
    mx = np.empty((n, n_bands), dtype=np.float32)
    for b in range(n_bands):
        seg = logs[:, edges[b]:max(edges[b + 1], edges[b] + 1)]
        mean[:, b], mx[:, b] = seg.mean(axis=1), seg.max(axis=1)
    return mean, mx


def load_espset_bands(raw_dir: Path, espcfg: Optional[Dict] = None, window: int = 50) -> List[Series]:
    """One Series per spectrum: T = n_bands frequency steps, channels vib_x = mean log-amplitude, vib_y = max
    log-amplitude (all other ESPset slots are absent). Series.unit = ESP id so every spectrum of one pump stays
    in the same split."""
    espcfg = espcfg or {}
    n_bands = int(espcfg.get("n_bands", window))
    if n_bands < window:
        raise ValueError(f"espset.n_bands ({n_bands}) must be >= data.window ({window}).")
    spectra, feats = read_spectra(raw_dir)
    lab = fault_labels(feats, espcfg.get("normal_labels"))
    mean, mx = band_pool(spectra, n_bands)
    names = channel_names("espset")
    ix, iy = names.index("vib_x"), names.index("vib_y")
    out: List[Series] = []
    for i in range(len(spectra)):
        X = np.full((n_bands, len(names)), np.nan, dtype=np.float32)
        X[:, ix], X[:, iy] = mean[i], mx[i]
        y = int(lab["y"][i]); f = int(lab["fail"][i])
        out.append(Series(
            dataset="espset", unit=f"esp{feats['esp_id'].iloc[i]}", X=X,
            rul_raw=np.full(n_bands, np.nan, dtype=np.float32), has_rul=False,
            fail=np.full(n_bands, f, dtype=np.int8), health=np.full(n_bands, 2 if f else 0, dtype=np.int8),
            event_class=np.full(n_bands, y, dtype=np.int16),
            meta={"mode": "bands", "row": i, "axis": "frequency", "classes": lab["classes"]}))
    return out
