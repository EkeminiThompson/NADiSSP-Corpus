# NADIP / NADiSSP-Corpus

**Nigerian-Adapted Domain-Invariant Prognostic (NADIP) Framework**

Transfer learning + domain adaptation for PdM under simulated Nigerian stressors, with held-out evaluation on **real Petrobras 3W** (WELL-*.parquet) when installed.

**Author:** Ekemini Thompson · Nnamdi Azikiwe University, Awka · 2023517003FA · 2026

---

## Honest data inventory

| Dataset | Nature | Role |
|---------|--------|------|
| NASA CMAPSS | **Simulated** turbofan | Source (RUL + soft fail) |
| UCI AI4I 2020 | **Synthetic** manufacturing | Source (failure labels; no RUL) |
| ESPset | Public ESP (if installed) | Optional source |
| **Petrobras 3W WELL-*** | **Real** offshore events | **Held-out target** for event detection (AUPRC/F1). **No RUL.** |
| Nigerian-augmented windows | Physics-informed **synthetic** shift | Intermediate domain for GRL — **not** Nigerian plant data |

Older `processed/FULL_43K_SUMMARY.txt` labels ("CMAPSS real", "3W-style synth") are **incorrect** for facility claims. Rebuild after placing real 3W.

---

## What the code supports (post-critique fix)

1. **Masked losses** — RUL only where `has_rul=1`; fail only where `has_fail=1`.
2. **Unit-level splits** — no overlapping-window train/val leakage.
3. **3W protocol** — only `WELL-*.parquet` (skips SIMULATED_*, DRAWN_*); event labels; **never invents RUL=100**.
4. **Economics** — explicit effectiveness + deployment cost. Default NPV is **~$200M / 5y class**, not $892M.
5. **Latency / packet-loss** — **not implemented**. Do not claim sub-120 ms.

---

## Quick start

```bash
git clone https://github.com/EkeminiThompson/NADiSSP-Corpus.git
cd NADiSSP-Corpus
pip install -r requirements.txt

# Real 3W (WELL-* = real wells)
mkdir -p data/raw
git clone --depth 1 https://github.com/petrobras/3W.git data/raw/3W

# Add CMAPSS + AI4I under data/raw/ (see scripts/download_datasets.py)

python scripts/prepare_corpus.py
python scripts/train_nadip.py --epochs 40
python scripts/evaluate.py          # RUL on CMAPSS test; AUPRC on 3w_real test
python scripts/run_economic.py      # reports ~$200M-class NPV under defaults
```

Report metrics from `results/tables/*.json` only.

---

## Claims to drop until results/ backs them

- "$892M NPV", "incidents 4.7→1.9" (1.9 was hardcoded)
- "First" transfer to real oil & gas
- Sub-120 ms / 40% packet-loss resilience
- Kinetics-400 init, Weibull PH head, staged 100+50+20 schedule
- CMAPSS or AI4I as real plant data

---

## License

MIT (code). Datasets: NASA, UCI, Petrobras 3W CC BY 4.0.
