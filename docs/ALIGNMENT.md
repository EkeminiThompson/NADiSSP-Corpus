# Alignment matrix: dissertation -> code

## Choices made where the dissertation is inconsistent or not implementable

| Issue | Dissertation says | Code does | Config key |
|---|---|---|---|
| Pre-training | Kinetics-400 initialisation | Not used. Kinetics weights do not fit a 41-channel 1-D transformer; `load_compatible` loads only name- and shape-matching tensors. `no_pretrain` is `not_applicable` without a checkpoint | `model.pretrained_init: null` |
| Positional encoding | sinusoidal (eq. 1) vs learnable (§3.10.2) | sinusoidal (eq. 1) | `model.positional` |
| GRL lambda schedule | sigmoid vs cosine vs "0.1 to 0" | Ganin: `2/(1+exp(-10p)) - 1` | `stage3_joint.*` |
| Learning rate | 1e-3 vs 3e-4 -> 1e-5 | cosine 3e-4 -> 1e-5 | `stage3_joint.lr_start/lr_end` |
| Survival head | Weibull NLL vs Cox partial likelihood (eq. 6) | Weibull-modulated PH (beta > 1) as default; Cox partial likelihood available | `stage4_weibull.loss` |
| Stressor assignment | §3.4.3 vs Table 3.9b differ | Table 3.9b, by sensor role | `augmentation.matrix` |
| ESPset | four source datasets incl. ESPset (8 time-series channels: vibration x/y/z, current, pressures, temperature, frequency) | Published ESPset has only vibration spectra (no time axis, none of those other channels). Not in the default corpus. Used (1) as a separate frequency-domain fault-diagnosis experiment and (2) optionally as a 50-band frequency sweep in the unified model with no time stressors. The dissertation's ESPset channel description must be corrected | `espset.*`, `data.datasets` |
| 3W event types | "8 event types" | 9 classes (1-9) + transients 101-109, as in 3W 2.0.0 | n/a |
| Economics | Table 4.9b totals vs Table 4.9a parameters | Table 4.9a simulated as written (lognormal downtime cost, Poisson incidents drawn per year, 5% discount, 5-year horizon). 4.9b totals are not reproduced; see below | `economics.*` |
| Incident SD | "4.7 +/- 1.2" | Poisson SD is about 2.2 for the same mean; the simulation uses Poisson | `economics.*` |
| Device latency | Table 4.10 | Measured on the host only; device rows are projections | n/a |

### Table 4.9a vs 4.9b

Closed form (verified by `tests/test_evaluation.py`): NPV mean = annuity x ((lambda_reactive - lambda_NADIP) x 8.4 +
(42.1 - 31.4) + (26.8 - 17.3)) in $M. With the 4.9a inputs this gives a reactive TCO of about $108M and an
intervention NPV of about $189M, not $456M / $892M. Either the 4.9b figures use inputs not listed in 4.9a, or the
parameters in 4.9a are wrong; the dissertation needs one of them corrected.

## Section/table -> code

| Dissertation | Code |
|---|---|
| §3.4 datasets, Table 3.6b sensors | `src/data/loaders.py`, `schema.py` |
| Table 3.8 / 3.13 sampling & windows | `sampling.py`, `corpus.py`, `configs/nadip.yaml::data` |
| §3.4.3 / Table 3.9b stressors | `augmentation.py` (+ `schema.STRESSOR_ROLES`) |
| §3.10 training stages 2-5 | `src/training/runner.py` |
| Tables 4.3-4.4 variants | `src/training/variants.py` (`TABLE_4_3`, `ABLATIONS`) |
| §4.2 diagnostics, threshold 0.40 | `src/evaluation/diagnostics.py` |
| §4.6 SHAP | `src/evaluation/explain.py` |
| Table 4.9a economics | `src/evaluation/economic.py` |
| §4.8-4.10 deployment | `src/evaluation/deployment.py`, `scripts/deployment_sim.py`, `benchmark_latency.py` |
| Table 3.15b service | `src/serving/app.py`, `Dockerfile` |

## Deployment checklist (Table 3.20, Warri / Niger Delta pilot)

1. Export: `scripts/deployment_sim.py` writes `models/nadip.onnx`, checks ONNX parity and INT8 AUPRC drop.
2. Latency: run `scripts/benchmark_latency.py models/nadip.onnx` **on the target device**; Table 4.10 values are projections.
3. Build `docker build -t nadip .` (needs `models/` and `data/processed/corpus_meta.json`).
4. Edge cache: the service falls back to cached statistics above the configured packet-loss threshold (72 h cache limit).
5. Feedback: `/feedback` appends to a buffer used for the EWC fine-tune after an outage; set deployment capex/opex in `economics.*`.
6. Re-measure divergence on local telemetry; retrain if it crosses the 0.40 intervention threshold.
