# Fetal Cardiac Ageing Clock

A multi-cohort deep learning framework for non-invasive fetal gestational age and cardiac
maturity estimation. From noisy multi-channel maternal abdominal ECG: extract a clean fetal
ECG, compute 12 interpretable cardiac biomarkers, regress a Cardiac Age (CA), and report
ΔAge = CA − GA as a candidate IUGR screening index. Trained on ADFECGDB only; validated
zero-shot (no fine-tuning) on NInFEA and CinC 2013.

See `CLAUDE.md` for full architecture, locked decisions, and data contracts.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Layout

- `src/fcac/` -- the package (`io`, `preprocessing`, `mecg`, `quality`, `memd`, `pinn`,
  `features`, `cvae`, `transformer`, `evaluation`, `viz`).
- `configs/` -- all hyperparameters (`default.yaml` + per-dataset overrides).
- `scripts/` -- numbered pipeline stages, `00_download.py` through `08_evaluate.py`, plus
  `reproduce_all.sh` as the single entry point (assumes `data/raw/` is already populated).
- `tests/` -- pytest, synthetic inputs only, no PhysioNet data required.

Run tests with `pytest` from the repo root (a `conftest.py` puts `src/` on `sys.path` even
without an editable install).

## Open questions

- **Bandpass band** is not yet finalized: the ICC 2026 paper uses 1-100 Hz, the spec PDF
  says 0.5-150 Hz. `configs/default.yaml` defaults to the paper's 1-100 Hz
  (`preprocessing.bandpass_preset: paper_1_100`) and keeps 0.5-150 Hz available as
  `spec_0_150`, pending the supervisor's confirmation.
- **PINN input coordinate scheme** is an ASSUMPTION, not a confirmed decision: input
  `[x, y, z, t, obs_0..obs_3]` (8-dim), spatial-only Laplace residual at interior
  collocation points, fECG read out from the potential channel `V`. Full rationale and
  two open sub-questions (electrode geometry, residual-vs-potential readout) are
  documented at the top of `src/fcac/pinn/coords.py` and in `configs/default.yaml`'s
  `pinn:` section. Do not report a number from the PINN until the supervisor signs this
  off.

## Note on legacy code

An older, flat `src/preprocessing/`, `src/extraction/`, `src/model/`, `src/features/`,
`src/evaluation/` tree existed from before the `src/fcac/` scaffold. Its working logic
(ADFECGDB/NInFEA loaders, CVAE, transformer) has since been ported into `src/fcac/` and
the legacy tree was removed; `src/fcac/` is now the only package new or existing code
imports from.
