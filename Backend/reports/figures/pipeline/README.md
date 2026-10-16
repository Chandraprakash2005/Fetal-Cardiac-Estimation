# Pipeline walkthrough: input -> output per stage

One record (ADFECGDB `r08`, window 0), one figure per stage. Regenerate with:

```
.venv/bin/python scripts/demo_pipeline.py                        # adfecgdb r08
.venv/bin/python scripts/demo_pipeline.py --dataset ninfea --record 10   # no PINN figure (needs scalp ref)
```

Nothing is retrained. Stages 1-4 recompute from the real pipeline code and the cached
artefacts in `data/`; stages 5-6 load the saved checkpoints; stage 7 reads
`reports/08_evaluate.json`. Shapes follow CLAUDE.md section 5.

| # | Figure | Input | Output | What to point at |
|---|---|---|---|---|
| 1 | `01_preprocessing_adfecgdb_r08.png` | Raw abdominal channel, `(C_raw, N_raw)` @ 1000 Hz | `PreprocessedRecording.windows`, `(W, 4, 5000)` | Baseline wander removed; spectrum cut to 1-100 Hz. No notch for ADFECGDB (locked). |
| 2 | `02_memd_adfecgdb_r08_w0.png` | Preprocessed window `(4, 5000)` | IMFs `(K=7, 4, 5000)` -> fetal-band sum `(4, 5000)` | Per-IMF score; IMF 1 is the one selected. MEMD removes artefact only, it does not separate mother from fetus. |
| 3 | `03_pinn_adfecgdb_r08_w0.png` | Fetal-band window `(4, 5000)` | `PINNOutput.fecg`, `(5000,)` | Maternal part removed (row 2), PINN beats vs the scalp reference (rows 4-5). Cancellation is skipped on windows already at fetal rate. |
| 4 | `04_features_adfecgdb_r08_w0.png` | Isolated fECG window `(5000,)` | `FeatureVector.f`, `(12,)` | Bars are z-scores against all ADFECGDB+NInFEA windows; labels are raw values; grey = NaN. |
| 5 | `05_cvae_adfecgdb_r08_w0.png` | 12 features (standardised, NaN median-filled) | Latent `z`, `(64,)` | Reconstruction vs input shows the autoencoder works; it does not show the latent carries age information. |
| 6 | `06_transformer_adfecgdb_r08.png` | Latent sequence `(59, 64)` (windows x 64) | Scalar CA in days | Heat-map in, one number out, with DeltaAge = CA - GA. |
| 7 | `07_evaluation_pooled_losocv.png` | CA and GA per record, 64 held-out records | MAE (days) + 95% bootstrap CI, Pearson r, DeltaAge per record | Includes the unit-verification message (`verify_mae_units`). |

## Caveats that apply to this set

- **ADFECGDB GA is synthetic.** The five ADFECGDB gestational ages are seeded random values
  in 38-41 weeks (`configs/experiments/synthetic_ga_smoketest.yaml`). Figures 6 and 7 are
  stamped for this. Nothing involving ADFECGDB GA or CA may be reported until real labels
  are supplied.
- **Figure 6 is a train-set fit, not an evaluation.** The saved transformer was fitted on
  all records (`scripts/07_train_transformer.py`). Its prediction for r08 (279.2 d vs a
  275.0 d label) just matches `reports/07_train_transformer.json`. The held-out result is
  figure 7.
- **Figure 7 shows no age signal within NInFEA.** Pooled MAE is 22.6 d (95% CI 17.6-28.3)
  against 20.3 d for predicting the mean. Pooled r = 0.56, but r within NInFEA is about
  -0.02: the pooled value mostly reflects the two cohorts (21-27 wk and 38-41 wk) being far
  apart. It also comes from a run on 2026-10-03 and will change when script 08 is re-run.
- **Figure 3 is one window.** Windows already at fetal rate (r01, r08) look good because
  they never needed separating. On maternal-dominated records such as r04 the PINN still
  locks onto the maternal rate (see `reports/figures/separation_adfecgdb_r04_w0.png`).
  The default extractor is hybrid (PINN where the quality gate passes, otherwise FastICA);
  this figure shows the PINN-only path. PINN numbers are not reportable until the
  coordinate scheme is signed off (CLAUDE.md section 2).
- **Figure 4 uses 5 s windows,** so `prsa`, `mse`, `sampen`, `dfa_alpha` and `lf_hf_ratio`
  are NaN here. Record-level dynamic features are computed from pooled R-R intervals
  (`extract_features_record`); this figure shows the per-window values.
- **Figure 2 selects IMF 1 only,** not the spec's IMF 3-5. Selection is scored per recording
  (CLAUDE.md section 2); IMF 3-5 scored near zero here.
- The QRS markers in figure 4 come from the simplified Pan-Tompkins detector, a sanity check
  rather than the fiducial detector in `features/fiducials.py`.

## Related figures one level up (`reports/figures/`)

- `preprocessing_{adfecgdb_r01,ninfea_10,cinc2013_a01}.png` - before/after for each dataset
- `memd_scores_{adfecgdb,ninfea,cinc2013}.png` - spec 7.1 IMF-score bar charts
- `feature_distributions.png` - 12 features, ADFECGDB vs NInFEA (CinC 2013 has no features yet)
- `extraction_*.png`, `separation_*.png` - extraction chain and maternal/fetal split
