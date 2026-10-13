# NADiSSP Open Reproducible Corpus

**Open, reproducible predictive-maintenance corpus** for the PhD work:

> **Development of a Transfer Learning and Domain Adaptation Framework for Predictive Maintenance in Nigeria’s Heterogeneous Oil and Gas Assets**  
> Ekemini Thompson (2023517003FA) · Nnamdi Azikiwe University, Awka · Supervisor: Prof. V.E. Ejiofor · October 2026

## Corpus size

| Split | Sequences |
|-------|-----------|
| **Source** | **31 679** |
| – NASA CMAPSS (real) | 12 879 |
| – UCI AI4I 2020 (real) | 10 000 |
| – 3W-style (synth matching published stats) | 2 600 |
| – ESPset-style (synth matching published stats) | 6 200 |
| **Target (Nigerian physics-informed augmentations)** | **36 000** |
| **TOTAL** | **67 679** |

(The original seminar claim was ~43 k; this regenerated package exceeds it.)

## Physics-informed Nigerian augmentations (exactly as in the seminar)

1. Humidity Drift – σ = 0.05–0.12 × (RH/100), RH ∈ [80, 95] %
2. Telemetry Loss – 25–40 % block dropout, 5–20 cycle segments
3. Sabotage Spikes – Cauchy impulses, 3–8×, p = 0.03 / cycle
4. Corrosion Trend – λ = 0.01–0.03 baseline shift per 100 cycles
5. Sampling Mismatch – 100 Hz → 1–10 Hz + ±10 % temporal jitter
6. Wave Loading – 0.1–0.5 Hz modulation, ±8 % pressure amplitude

## Repository layout

```
├── README.md
├── processed/
│   ├── source/full_source_43k.npz          # 31 679 source sequences
│   ├── target_batches/batch_*.npz         # 36 000 target sequences (120 × 200)
│   ├── metadata/full_43k_meta.json
│   └── FULL_43K_SUMMARY.txt
├── scripts/
│   ├── load_full_corpus.py                 # load everything into memory
│   ├── generate_nadissp_corpus.py
│   ├── generate_43k_lowmem.py
│   └── download_3w_espset.py               # fetch real Petrobras 3W + ESPset
└── raw/                                    # original public CMAPSS + AI4I files
```

## Quick start

```bash
git clone https://github.com/EkeminiThompson/NADiSSP-Corpus.git
cd NADiSSP-Corpus
pip install numpy scipy pandas
python scripts/load_full_corpus.py
```

## Reproducibility

- Fixed random seed = **2026**
- All augmentation parameters are documented in `processed/metadata/full_43k_meta.json`
- Generation scripts are pure Python

## Citation

Please cite the original public datasets (NASA CMAPSS, UCI AI4I 2020, Petrobras 3W, ESPset) and the NADiSSP thesis / seminar presentation.

## Licence

- Scripts: MIT
- Regenerated sequences inherit the licences of the underlying public sources (NASA, CC BY 4.0, etc.)
- This repository does **not** claim ownership of the original public data.

---

**Author:** Ekemini Thompson · 2023517003FA  
**Supervisor:** Prof. V.E. Ejiofor  
**Department of Computer Science, Nnamdi Azikiwe University, Awka**
