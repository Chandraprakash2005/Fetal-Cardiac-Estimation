# CLAUDE.md — Fetal Cardiac Ageing Clock

Project context for Claude Code. Read this fully before touching any file. This file is the single
source of truth for architecture, conventions, and locked decisions. If a request conflicts with a
LOCKED decision below, stop and ask instead of silently deviating.

---

## 1. What this project is

**Title:** Fetal Cardiac Ageing Clock — A Multi-Cohort Deep Learning Framework for Non-Invasive
Fetal Gestational Age and Cardiac Maturity Estimation.

**Goal:** From noisy multi-channel maternal abdominal ECG, (1) extract a clean fetal ECG, (2) compute
12 interpretable cardiac biomarkers, (3) regress a **Cardiac Age (CA)**, and (4) report
**ΔAge = CA − GA**, where GA is the externally supplied LMP-dated gestational age.
ΔAge is a candidate IUGR screening index.

**Supervisor:** Dr. Muthukumar K A, Assistant Professor, Dept. of CSE, SSN College of Engineering.
**Team:** Bhuvanesh S (signal processing: preprocessing, MEMD, PINN), Chandraprakash K (features,
augmentation), Teja Sree C (CVAE, Transformer, evaluation).
**Dev machine:** Kali Linux, repo root `/mnt/acer/FYP/fetal-cardiac-ageing-clock/`.

**Reference papers (do not fabricate details beyond these):**
- Muthukumar et al., ICC 2026 — 1D-CNN fECG extraction + EMD quality gating. Source of the
  preprocessing recipe (bandpass → z-score → 5000-sample windows). Also the baseline to reproduce.
- Wahbah et al., Front. Physiol. 2024 — stacked BiLSTM fECG extraction (comparison point).
- Raissi, Perdikaris & Karniadakis 2019 — PINN foundations.

---

## 2. LOCKED decisions (do not change without explicit instruction)

| Topic | Decision | Source |
|---|---|---|
| Datasets | **ADFECGDB** (train) + **NInFEA** and **CinC 2013** (zero-shot validation, never trained on). **Amended 2026-10-02 at Bhuvanesh's request:** the CVAE and transformer (NOT the PINN) are trained on ADFECGDB + the NInFEA recordings that pass quality gating (9 of 60), evaluated by pooled leave-one-subject-out CV; NInFEA is therefore no longer a held-out zero-shot set for those two stages. CinC 2013 remains zero-shot and untrained. CinC 2013 was originally dropped per the supervisor's verbal instruction, then reinstated on 2026-09-08 at Bhuvanesh's explicit request as a second zero-shot cohort once real preprocessed data existed for it. | Supervisor, verbal; reinstated 2026-09-08 |
| GA vs CA | Model predicts **CA only**. True GA is an input label from LMP dating, never a model output. ΔAge = CA − GA. | Spec §5, team |
| Ground truth | ADFECGDB `Direct_1` scalp channel = ground truth ONLY, never a model input. NInFEA has **no** scalp ECG; Doppler beat timing → timing metrics (F1, R-R) only, no amplitude SNR. | Spec §6.1 |
| Channels | Everything downstream of preprocessing consumes **4 abdominal channels @ 1000 Hz**. | Spec §5 |
| Kurtosis channel selection | NInFEA: pick top-4 of the 24 abdominal channels by kurtosis, computed **after** bandpass filtering (raw drift inflates kurtosis). Exclude non-physiological channels `dc1–dc4`, `saw`, `sync`. | Team |
| Resampling order | Filter at native rate first, then resample NInFEA 2048 → 1000 Hz, then normalise, then window. | Team |
| Notch | 50 Hz notch applies to **NInFEA only** (measured spectral prominence). ADFECGDB does not need it. | Team, measured |
| Window | 5000 samples (5 s @ 1 kHz), non-overlapping by default. Stride is a config param. | ICC 2026 |
| Normalisation | Per-channel z-score (zero mean, unit variance) per recording. | ICC 2026 |
| MEMD role | Removes motion / baseline drift / muscle artefact only. **Maternal–fetal separation is the PINN's job.** IMF count is data-determined (~10–14); "IMF 3–5" is a heuristic, actual selection is scored per recording (spectral coherence × in-band power, fetal band 20–70 Hz). | Team |
| PINN | 5 FC layers × 512 units; Tanh (layers 1–3), ReLU + Dropout(0.2) (layers 4–5). Predict potential V, enforce ∇²V = 0 via double autograd. Loss `L = 0.7·L_data + 0.3·L_physics`. **Default: train on ADFECGDB only, freeze, no fine-tuning of a trained PINN. Amended 2026-10-02 at Bhuvanesh's request: an experimental PINN may also train on NInFEA recordings (split by mother; NInFEA is training/validation, not zero-shot), judged on held-out NInFEA mothers and on CinC 2013, which is never trained on and is the zero-shot cohort.** | Spec §7.2 |
| Extraction | Default method is **hybrid** (`extraction.method: hybrid`): per window, maternal cancellation -> frozen PINN when the output passes the quality gate, otherwise FastICA BSS on the preprocessed windows. `pinn` (cancel -> PINN only) remains selectable. Confirmed by Bhuvanesh 2026-10-03; measured best on ADFECGDB, CinC 2013 and NInFEA (see engineering log). PINN numbers are still not to be reported until the coordinate scheme is signed off. | Bhuvanesh, 2026-10-03 |
| WGAN | **DROPPED** (unstable at ~90 recordings). Augmentation comes from the CVAE latent space instead. Do not scaffold a WGAN module. | Team |
| CVAE | **Contrastive** VAE (not Conditional). Input: 12-feature window vector. Latent dim 64. InfoNCE contrastive loss, τ = 0.07, positives = same-subject / same-GA windows. GA label is **not** fed as input (circular). | Spec §9.2, team |
| Transformer | Encoder, 4 heads, d_model = 256, d_k = 64, 3 layers. Input: sequence of latent z. Output: scalar CA. MAE loss, AdamW. | Spec §9.2 |
| Evaluation | Pooled LOSOCV over ADFECGDB (5) + gated NInFEA recordings (9), grouped by subject (amended 2026-10-02; was ADFECGDB LOSOCV + NInFEA zero-shot, which now applies to CinC 2013 only, and only where it has GA -- it has none). Report MAE in **days** + 95% bootstrap CI (1000 resamples, 2.5–97.5 pct). Mandatory MAE unit-verification script before any number is reported. | Spec §10 |
| Reproducibility | Fixed seeds everywhere, pinned `requirements.txt`, single entry point that reproduces every figure/number. | Spec §11 |

**Open questions (do not resolve unilaterally, expose as config):**

*PINN coordinate scheme.* §2 fixes the PINN's width/depth/activations/loss weights but
not its input coordinates or how V is read out as a fetal ECG. As of 2026-09-08 a
concrete scheme is implemented so stages 1-3 run end to end -- input
`[x, y, z, t, obs_0..obs_3]` (`pinn.input_dim: 8`), spatial-only Laplace residual at
interior collocation points, fECG read out from the potential channel `V` (measured
2026-09-08 on ADFECGDB, QRS F1: `readout: potential` scores 0.519 vs `readout: residual`
0.443, beating the MEMD input's own 0.501 -- the Laplace constraint acts as a spatial
denoiser, not a maternal-absorber). It is an ASSUMPTION, documented with its two open
sub-questions at the top of `src/fcac/pinn/coords.py` and in README "Open questions".
Do not report a number from it until the supervisor signs it off, and do not treat it
as LOCKED.

*Bandpass band.* -- RESOLVED 2026-10-03 by Bhuvanesh: **1-100 Hz** (the ICC 2026 paper's band). The
0.5-150 Hz spec preset stays in config as `spec_0_150` for ablations only.

---

## 3. Dataset facts

**ADFECGDB** (PhysioNet `adfecgdb/1.0.0`)
- 5 recordings r01, r04, r07, r08, r10 · EDF · 1000 Hz · 16-bit
- Channels: `Direct_1` (scalp, ground truth), `Abdomen_1..4` (model input)
- `.qrs` annotation files with fetal R-peak positions
- GA 38–41 weeks · 5 min each · loaded with `pyedflib`
- ICC 2026 EMD quality: r01/r04/r08 = high, r07 = low (expect trouble on r07)

**NInFEA** (PhysioNet `ninfea/1.0.0`)
- 60 recordings, 39 subjects · WFDB · 2048 Hz · 22-bit
- 34 channels: 24 abdominal unipolar + thoracic + non-physiological (`dc1–dc4`, `saw`, `sync`)
- Synchronised pulsed-wave Doppler → fetal beat timing reference
- GA 21–27 weeks · loaded with `wfdb`
- **GA coverage gap:** training is 38–41 wks, validation is 21–27 wks; 28–37 wks is unrepresented.
  Downstream code must never assume overlapping GA ranges.

**CinC 2013** (PhysioNet `challenge-2013/1.0.0`, set-a) — reinstated 2026-09-08, see §2
- 25 records a01–a25 · WFDB · 1000 Hz · 60 s each
- 4 abdominal channels `AECG1..AECG4` — already exactly the target 4 channels at 1000 Hz,
  so neither kurtosis selection nor resampling applies (unlike NInFEA)
- `.fqrs` annotations give fetal R-peak timing only — no scalp/amplitude ground truth,
  same as NInFEA
- No published per-record GA — `ga_days` stays `null`; any ageing/ΔAge work must skip
  this cohort rather than invent a GA
- 13 of 25 records carry strong 50 Hz mains (measured 2026-09-08, prominence 5x–231x
  over the local noise floor) — notch filter enabled for this dataset

---

## 4. Repository layout (build exactly this)

```
fetal-cardiac-ageing-clock/
├── CLAUDE.md                   ← this file
├── README.md
├── requirements.txt            ← pinned versions
├── pyproject.toml              ← `pip install -e .`, package name `fcac`
├── configs/
│   ├── default.yaml            ← every hyperparameter lives here, nothing hardcoded
│   ├── datasets/
│   │   ├── adfecgdb.yaml
│   │   ├── ninfea.yaml
│   │   └── cinc2013.yaml       ← reinstated 2026-09-08, see §2/§3
│   └── experiments/            ← overrides for ablations, one file per run
├── data/
│   ├── raw/                    ← gitignored; PhysioNet downloads
│   ├── interim/                ← gitignored; preprocessed .npz per recording
│   ├── memd/                   ← gitignored; cached IMFs (MEMD is slow)
│   └── processed/              ← gitignored; PINN outputs, features, latents
├── src/fcac/
│   ├── __init__.py
│   ├── config.py               ← load/merge YAML → dataclass; single `Config` object
│   ├── seed.py                 ← `set_seed(seed)` for numpy/torch/random
│   ├── io/
│   │   ├── adfecgdb.py         ← load EDF + .edf.qrs → Recording
│   │   ├── ninfea.py           ← load WFDB + Doppler → Recording
│   │   ├── cinc2013.py         ← load WFDB set-a + .fqrs → Recording (reinstated 2026-09-08)
│   │   └── types.py            ← `Recording` dataclass (see §5)
│   ├── preprocessing/
│   │   ├── filters.py          ← butter_bandpass, notch, resample (pure functions)
│   │   ├── channels.py         ← kurtosis_select, exclude_nonphys
│   │   ├── normalise.py        ← zscore
│   │   ├── windowing.py        ← segment(signal, win, stride)
│   │   └── pipeline.py         ← `preprocess(recording, cfg) -> PreprocessedRecording`
│   ├── mecg.py                 ← adaptive maternal-ECG template cancellation, MEMD → PINN
│   ├── quality.py              ← window/record quality gating (silent maternal-lock failure)
│   ├── memd/
│   │   ├── sift.py             ← vendored Hammersley-sequence multivariate EMD
│   │   ├── decompose.py        ← wraps memd/sift.py, disk cache keyed by recording+window
│   │   ├── select.py           ← imf_scores (coherence × in-band power), select_fetal_band
│   │   └── diagnostics.py      ← welch per IMF, Pan-Tompkins bpm check, score bar chart
│   ├── pinn/
│   │   ├── model.py            ← FC network per §2
│   │   ├── coords.py           ← input coordinate scheme + readout (ASSUMPTION, see §2)
│   │   ├── physics.py          ← laplacian(V, coords) via double autograd
│   │   ├── losses.py           ← L_data, L_physics, hybrid
│   │   ├── train.py            ← training loop, ADFECGDB only
│   │   └── infer.py            ← frozen zero-shot forward pass
│   ├── features/
│   │   ├── fiducials.py        ← P/Q/R/S/T detection
│   │   ├── morphological.py    ← QTc σ, P-dispersion, QRS mean/var/asymmetry
│   │   ├── dynamic.py          ← PRSA, MSE, ApEn, SampEn, DFA α, LF/HF, symbolic dynamics
│   │   └── extract.py          ← `extract_features(fecg) -> np.ndarray[12]`
│   ├── cvae/
│   │   ├── model.py            ← encoder/decoder, 12 → 64
│   │   ├── losses.py           ← reconstruction + KL + InfoNCE(τ=0.07)
│   │   ├── train.py
│   │   └── augment.py          ← sample latent-space neighbours (replaces WGAN)
│   ├── transformer/
│   │   ├── model.py            ← 4-head, d_model 256, 3 layers → scalar CA
│   │   ├── train.py            ← MAE loss, AdamW
│   │   └── infer.py
│   ├── evaluation/
│   │   ├── extraction.py       ← SNR (ADFECGDB only), QRS F1 ±50 ms, Bland–Altman R-R
│   │   ├── ageing.py           ← MAE, RMSE, Pearson r, bootstrap_ci
│   │   ├── losocv.py           ← leave-one-subject-out driver
│   │   ├── delta_age.py        ← ΔAge = CA − GA
│   │   └── verify_units.py     ← mandatory MAE unit-verification script
│   └── viz/
│       └── plots.py            ← shared matplotlib style, save to reports/figures/
├── scripts/
│   ├── 00_download.py
│   ├── 01_preprocess.py        ← --dataset {adfecgdb,ninfea}
│   ├── 02_memd.py
│   ├── 03_train_pinn.py
│   ├── 04_extract_fecg.py      ← zero-shot on ninfea
│   ├── 05_features.py
│   ├── 06_train_cvae.py
│   ├── 07_train_transformer.py
│   ├── 08_evaluate.py
│   └── reproduce_all.sh        ← the §11 single entry point
├── tests/                      ← pytest; one file per package, synthetic inputs
├── notebooks/                  ← exploration only, never imported by src
└── reports/
    ├── figures/
    └── engineering_log.md      ← every design decision → spec section it satisfies
```

---

## 5. Data contracts (tensor shapes at every arrow)

All arrays are `float32` numpy unless noted. Shapes must match these exactly; if a module needs
something different, change the contract here first.

```
Recording            .signal (C_raw, N_raw)  .fs_raw  .fetal_ref (timing or scalp)  .ga_days  .subject_id
PreprocessedRecording .windows (W, 4, 5000)  .fs=1000 .ref_windows (W, 5000) | None  .ga_days .subject_id
MEMDOutput           .imfs (W, K, 4, 5000)   .scores (W, K)   .selected (W, list[int])
FetalBandWindows     .x (W, 4, 5000)         ← sum of selected IMFs, PINN input
PINNOutput           .fecg (W, 5000)          ← single-channel isolated fetal ECG
FeatureVector        .f (W, 12)               ← order fixed in features/extract.py FEATURE_NAMES
Latent               .z (W, 64)
CAPrediction         .ca_days (W,)  or per-recording aggregate (R,)
```

GA and CA are stored in **days** everywhere internally. Convert to weeks only at display time.

---

## 6. Coding conventions

- Python 3.11, PyTorch 2.x, numpy, scipy, wfdb, pyedflib, MEMD_all, pyyaml, pytest.
- Every module exposes pure functions with type hints and a docstring stating **which spec stage
  (§5 table, stage 1–8) it implements**.
- No hardcoded numbers in model/preprocessing code — read from `Config`. Filter cutoffs, IMF band,
  window size, loss weights, latent dim, τ, heads, layers all live in `configs/default.yaml`.
- Caching: MEMD outputs are expensive → always cache to `data/memd/` keyed by
  `{dataset}/{record_id}/win{idx}.npz`; check cache before recomputing.
- Logging via `logging`, not print. Each script writes a JSON run summary to `reports/`.
- Every training script calls `set_seed(cfg.seed)` first and writes the resolved config alongside
  checkpoints.
- Tests use small synthetic signals (e.g. summed sinusoids + impulse train) so they run in seconds
  without PhysioNet data.
- Plots: verify preprocessing and MEMD output visually — time-domain + Welch spectrum — before
  any downstream stage consumes them.
- Never write code that touches `Direct_1` outside `io/`, `evaluation/`, or `memd/select.py`
  (where it is a scoring reference, not an input).

---

## 7. How the user works with you

- Bhuvanesh will give short, informal instructions per module ("now do the pinn physics loss"). Interpret
  intent; make targeted edits; do not reformat unrelated files.
- Scaffold **boilerplate first**: module skeletons, configs, data contracts, tests, scripts — with
  clear `TODO(<owner>)` markers where the user will supply model-specific logic. Do not invent
  scientific details (feature formulas, physics coordinates, loss variants) that aren't in §2 or
  the spec; leave a TODO and ask.
- When something in the spec PDF contradicts §2 of this file, §2 wins (it reflects later verbal
  instructions from the supervisor).
- Keep `reports/engineering_log.md` updated with one line per decision.
- Before claiming a stage works, run its test file and show the output.

---

## 8. Immediate scaffolding task

Build the full tree in §4 with:
1. `configs/default.yaml` containing every hyperparameter in §2 (both bandpass presets, notch flag
   per dataset, MEMD directions=64, fetal band 20–70 Hz, PINN/CVAE/Transformer dims, loss weights,
   seed, bootstrap settings).
2. `config.py` that loads default + dataset + optional experiment YAML into a frozen dataclass.
3. `io/types.py` with the dataclasses from §5.
4. Complete, working **preprocessing** package (filters, kurtosis selection, resample, z-score,
   windowing, pipeline) — this is already designed, implement it fully.
5. **MEMD** package: decompose wrapper with cache, scoring/selection per §2, diagnostics.
6. **PINN / CVAE / Transformer**: model classes with the exact dims in §2, loss stubs with
   signatures and TODOs, train/infer loops that run end-to-end on random tensors.
7. **Features**: `extract.py` returning a 12-vector with `FEATURE_NAMES` in spec §8 order; each
   feature a stub function with docstring + TODO.
8. **Evaluation**: bootstrap CI, MAE/RMSE/r, LOSOCV driver, `verify_units.py`, ΔAge.
9. `tests/` that pass on synthetic data.
10. `scripts/reproduce_all.sh` chaining scripts 01–08.

Do not download data. Do not run MEMD on real recordings. Stop after the scaffold and summarise
what has TODOs remaining.

---

## 9. Note on the legacy flat `src/` tree (historical)

Before the `src/fcac/` scaffold, the repo had a flat `src/preprocessing/`,
`src/extraction/`, `src/model/`, `src/features/`, `src/evaluation/` tree with mostly
empty stub files, plus two working modules (`src/preprocessing/adfecgdb.py` and
`src/preprocessing/ninfea.py`) that were ported into `src/fcac/io/` and
`src/fcac/preprocessing/`, fixing two bugs relative to the LOCKED decisions in §2
(notch applied to both datasets instead of NInFEA-only; NInFEA filtering happened
*after* resampling instead of before) along the way.

A second, untracked pair of top-level `cvae/` and `transformer/` folders later
appeared with a more complete CVAE/Transformer implementation (InfoNCE projection
head, GA-window positive pairing, KL warmup, CLS-token pooling, LOSOCV/zero-shot
wiring, Mahalanobis OOD gating) than the `src/fcac/cvae`/`src/fcac/transformer`
scaffold had at the time. That logic was ported into `src/fcac/cvae/` and
`src/fcac/transformer/` (see `reports/engineering_log.md`, 2026-09-28 entries), and
both the legacy flat tree and the top-level `cvae/`/`transformer/` folders were then
deleted. `src/fcac/` has been the sole authoritative package since; this section is
kept only as a historical record of where each piece of logic originally came from.
