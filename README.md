# NADiSSP-Corpus: NADIP (Nigerian-Adapted Domain-Invariant Prognostics)

Code for the dissertation's methodology (Chapters 3-4). **Nothing in this repository hard-codes a result.**
Every number in a table is produced by running the pipeline; `docs/ALIGNMENT.md` maps each dissertation
section/table to the code that implements it and lists where the document disagrees with itself.

## What the pipeline does

| Stage | Code | Dissertation |
|---|---|---|
| Load CMAPSS FD001-4, AI4I 2020, 3W 2.0.0 (ESPset excluded, see below) | `src/data/loaders.py` | §3.4, Table 3.6b |
| Unified 41-slot channel vector, per-dataset blocks | `src/data/schema.py` | §3.4 |
| Stratified sampling (min cell 30; 3W >= 70% real), group-wise splits | `src/data/sampling.py`, `corpus.py` | Tables 3.8, 3.13 |
| Six Nigerian stressors, applied per Table 3.9b by sensor role | `src/data/augmentation.py` | §3.4.3 |
| Stage 2 SimCLR (NT-Xent, tau 0.5, physics-informed views) | `src/training/runner.py` | §3.10 |
| Stage 3 joint GRL domain adaptation (Ganin lambda ramp) | `src/training/runner.py` | §3.10 |
| Stage 4 Weibull-modulated proportional-hazards head | `src/models/nadip.py` | eq. 6 |
| Stage 5 XGBoost ensemble on frozen embeddings + interpretable features | `src/training/runner.py` | §3.10 |
| 12 variants / ablations, 10 seeds, paired t, Cohen's d_z, 95% CI | `src/training/variants.py`, `src/evaluation/stats.py` | Tables 4.3-4.4 |
| MMD, Sinkhorn W1, per-channel W1, permutation test, discriminator, t-SNE silhouette | `src/evaluation/diagnostics.py` | §4.2 |
| TreeSHAP (native XGBoost `pred_contribs`) | `src/evaluation/explain.py` | §4.6 |
| Monte-Carlo economics | `src/evaluation/economic.py` | Table 4.9a |
| Packet loss, outage drift + buffered EWC, ONNX parity/INT8/latency | `src/evaluation/deployment.py` | §4.8-4.10 |
| FastAPI service, edge-cache fallback, feedback buffer; Docker | `src/serving/app.py`, `Dockerfile` | Table 3.15b |

Key protocol rules, all enforced by tests: splits are by **group (unit/well/series), never by window**; the
normaliser is fitted on clean source *train* windows only; 30% of target labels are kept and 70% stripped;
stressors are applied only to sensors whose role matches the stressor.

## Data layout

Datasets are not redistributed. Put them under `data/raw/` (or pass `--raw-dir`):

```
data/raw/CMAPSS/{train,test,RUL}_FD00{1..4}.txt
data/raw/AI4I/ai4i2020.csv
data/raw/3W/dataset/<class>/{WELL-*,SIMULATED_*,DRAWN_*}.parquet     # petrobras/3W v2.0.0; DRAWN_* are skipped
```

`python scripts/download_datasets.py` fetches what is scriptable (3W sparse clone) and says where to put the rest;
`--verify` prints SHA-256 sums so a run can be tied to exact files.

**ESPset is not used.** It is published (NINFA-UFES/ESPset) only as `spectrum.csv` (6,032 vibration spectra x
12,103 frequency bins, normalised by rotation frequency) plus `features.csv` (labels and summary features). There is
no time axis and no pressure/current/temperature channel, so it cannot feed a time-series model without inventing a
signal, and no ESPset result is reported. The study uses CMAPSS and AI4I as source domains and 3W as the real-facility
data. `data.datasets` in `configs/nadip.yaml` omits it; its channel slots stay in the 41-wide vector as inert zeros so
checkpoints and tests keep a fixed shape. The loader refuses `spectrum.csv`/`features.csv` with an explanatory error.
The `no_3w_espset` ablation therefore removes 3W only: describe it as "without 3W" in the text.

**3W notes (checked against the real v2.0.0 files).** Classes are 0 (normal), 1-9 (events) and 101-109
(transients). Only the seven Table 3.6b sensors are used. Several are mostly absent in real wells
(P-JUS-CKGL ~37% coverage, QGL ~28%); gaps are forward-filled and fully missing channels become 0. Some event
types have almost no real series (e.g. class 5: 11 real vs 439 simulated), so the 70%-real floor cannot hold in
every cell. The achieved fraction is written to `corpus_meta.json` (`real_fraction_source`, 0.698 in a full 3W-only build).
Rows whose `class` is `<NA>` are treated as unlabelled and trimmed, never as normal.

## Running

```bash
pip install -r requirements.txt
python scripts/prepare_corpus.py            # corpus + corpus_meta.json
python scripts/run_ablation.py              # all variants x 10 seeds (hours on CPU)
python scripts/evaluate.py                  # diagnostics, SHAP, statistics
python scripts/run_economics.py             # uses the measured AUPRC from results/evaluation.json
python scripts/deployment_sim.py            # packet loss, outage, ONNX, latency
python scripts/make_figures.py
python scripts/run_all.py                   # all of the above
python scripts/run_all.py --config configs/smoke.yaml --raw-dir <fixtures>   # plumbing check only
python -m pytest -q tests                   # 59 tests on synthetic data in the real on-disk formats
```

Config lives in `configs/nadip.yaml` (every value is commented with its section/table); override with
`--set a.b=c`. Resumable: each stage checkpoints and `--force` recomputes.

## Measured vs projected

* **Measured by this code:** all divergence metrics, AUPRC/F1/RMSE, ablation deltas and tests, SHAP, packet-loss
  curve, outage recovery, ONNX parity, and latency *on the host that runs it*.
* **Projected, not measured:** latency on the Table 4.10 edge devices (Jetson, Raspberry Pi etc.) and anything
  involving a live Nigerian facility. The economic model is a simulation whose incident reduction is
  `efficacy (0.70) x measured AUPRC`; deployment capex/opex default to 0 until you set them.
* **Not run in the authoring environment:** CMAPSS and AI4I were unreachable there, so no full-scale
  result accompanies this commit. Only the 3W loader and corpus build were validated on real data. The tests use
  synthetic fixtures. Run the full protocol where the datasets are available.

## Where the document disagrees with itself

Explicit choices are listed in `docs/ALIGNMENT.md`. The two that affect reported numbers:

1. **Table 4.9b cannot be reproduced from Table 4.9a.** Using 4.9a's own parameters, a reactive 5-year TCO
   comes to about $108M and the NPV of the intervention to about $189M (about $156M for a 40% incident
   reduction), against the $456M TCO / $892M NPV / ">$550M" quoted. The code reports what the model gives.
2. **Kinetics-400 initialisation is not applicable** to this architecture (no compatible checkpoint), so
   `no_pretrain` reports `not_applicable` unless `model.pretrained_init` points at a shape-compatible checkpoint.

`dissertation_updates/*.txt` are earlier drafts of dissertation text and still contain the unreproducible
figures ($892M NPV, AUPRC 0.54 to 0.85, "first published"); they were not edited and should be revised
once the full run exists.
