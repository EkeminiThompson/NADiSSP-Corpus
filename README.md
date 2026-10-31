# NADiP — Nigerian-Adapted Domain-Invariant Prognostics

Implementation of the NADiP methodology (dissertation Chapters 3–4): multi-stage domain-adaptive prognostics for predictive maintenance under Nigerian operating conditions.

All reported metrics are produced by executing the pipeline on the configured datasets. Configuration is in `configs/nadip.yaml`. Mapping from dissertation sections to code is in `docs/ALIGNMENT.md`.

## Pipeline overview

| Stage | Code | Dissertation |
|---|---|---|
| Load CMAPSS FD001–4, AI4I 2020, 3W 2.0.0, ESPset | `src/data/loaders.py` | §3.4, Table 3.6b |
| Unified channel schema (per-dataset blocks) | `src/data/schema.py` | §3.4 |
| Stratified sampling; group-wise train/val/test splits | `src/data/sampling.py`, `corpus.py` | Tables 3.8, 3.13 |
| Nigerian stressors (role-conditioned) | `src/data/augmentation.py` | §3.4.3 |
| Stage 2 — SimCLR (NT-Xent, physics-informed views) | `src/training/runner.py` | §3.10 |
| Stage 3 — Joint GRL domain adaptation | `src/training/runner.py` | §3.10 |
| Stage 4 — Weibull-modulated survival head | `src/models/nadip.py` | eq. 6 |
| Stage 5 — XGBoost on frozen embeddings | `src/training/runner.py` | §3.10 |
| Ablations (12 variants), multi-seed statistics | `src/training/variants.py`, `src/evaluation/stats.py` | Tables 4.3–4.4 |
| Domain diagnostics (MMD, Wasserstein, discriminator) | `src/evaluation/diagnostics.py` | §4.2 |
| TreeSHAP explanations | `src/evaluation/explain.py` | §4.6 |
| Monte-Carlo economic evaluation | `src/evaluation/economic.py` | Table 4.9a |
| Deployment stress tests, ONNX export | `src/evaluation/deployment.py` | §4.8–4.10 |
| Inference service (FastAPI) | `src/serving/app.py`, `Dockerfile` | Table 3.15b |

Protocol constraints enforced by tests: splits are by group (unit/well/series), not by window; the normaliser is fit on clean source training windows only; target labels are partially masked; stressors apply only to matching sensor roles.

## Data layout

Datasets are not redistributed. Place them under `data/raw/`:

```
data/raw/CMAPSS/{train,test,RUL}_FD00{1..4}.txt
data/raw/AI4I/ai4i2020.csv
data/raw/3W/dataset/<class>/{WELL-*,SIMULATED_*}.parquet   # petrobras/3W v2.0.0
data/raw/ESPset/*.csv
```

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py --verify
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests

python scripts/prepare_corpus.py --config configs/smoke.yaml
python scripts/run_all.py --config configs/smoke.yaml --seeds 2

# Full protocol (GPU recommended)
python scripts/prepare_corpus.py --config configs/nadip.yaml
python scripts/run_all.py --config configs/nadip.yaml
```

Outputs are written under `results/` (`evaluation.json`, `economics_summary.json`, `tables.md`, figures).

## Economic evaluation

Incident rates and cost parameters follow Table 4.9a. Annual TCO and five-year NPV are computed by Monte-Carlo simulation in `src/evaluation/economic.py`. The NADiP incident rate is derived from measured AUPRC unless overridden in config. Results are written to `results/economics_summary.json`.

## Citation

If you use this code, please cite the dissertation and the upstream datasets (NASA CMAPSS, UCI AI4I 2020, Petrobras 3W).

## Licence

See repository licence file. Upstream datasets retain their original terms.
