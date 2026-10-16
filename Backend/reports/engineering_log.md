# Engineering log

One line per decision, with the spec section it satisfies. Newest at the bottom.

- 2026-09-05: Scaffolded `src/fcac/` per CLAUDE.md §4; left the pre-existing flat
  `src/preprocessing/`, `src/extraction/`, `src/model/`, `src/features/`,
  `src/evaluation/` tree in place, untouched, at the user's request. -> CLAUDE.md §4, §8.
- 2026-09-05: Ported the working ADFECGDB/NInFEA loader+filter logic from the old
  `src/preprocessing/*.py` into `src/fcac/io/*.py` (loading only) and
  `src/fcac/preprocessing/*.py` (filtering/resampling/selection), fixing two bugs
  relative to the LOCKED decisions: (1) notch filter was applied to both datasets, now
  NInFEA-only; (2) NInFEA filtered after resampling, now filters at native rate first.
  -> CLAUDE.md §2 "Notch", "Resampling order".
- 2026-09-05: `configs/default.yaml` defaults `preprocessing.bandpass_preset` to
  `paper_1_100` (ICC 2026 paper's 1-100 Hz) with `spec_0_150` (0.5-150 Hz, spec PDF)
  kept as a named alternate preset -- open question, awaiting supervisor confirmation.
  -> CLAUDE.md §2 "Open question".
- 2026-09-05: PINN `input_dim`/`output_dim` (configs/default.yaml `pinn.input_dim: 4`,
  `pinn.output_dim: 1`) are placeholders; the actual input coordinate scheme for the
  potential-field PINN is not specified anywhere available and needs the supervisor's
  input before real training. -> CLAUDE.md §7 "do not invent ... physics coordinates".
- 2026-09-05: `src/fcac/features/{fiducials,morphological,dynamic}.py` are left as
  `NotImplementedError` stubs (12 functions covering FEATURE_NAMES) pending a chosen
  fiducial-detection algorithm and precise formulas (QTc correction, entropy
  parameters, LF/HF band edges for fetal HR, etc). -> CLAUDE.md §7 "leave a TODO and
  ask".
- 2026-09-05: `src/fcac/memd/decompose.py::_run_memd` raises `NotImplementedError` --
  no MEMD source is pinned yet (see `requirements.txt` TODO). Scoring/selection logic
  in `memd/select.py` is fully implemented and tested on synthetic signals.
  -> CLAUDE.md §2 "MEMD role".
- 2026-09-29: Ported a more advanced snapshot of this project (found untracked at
  `fetal-cardiac-ageing/`, extracted from a Downloads zip) into `src/fcac/` --
  `memd/sift.py` (vendored Hammersley-sequence MEMD, resolving the "MEMD source not
  pinned" blocker), `pinn/coords.py` + updated `pinn/{losses,physics,train,infer}.py`
  (the input-coordinate-scheme ASSUMPTION, documented, not LOCKED), updated
  `io/adfecgdb.py` (.edf.qrs annotation loading, verified against r01: 644 beats/300s =
  128.8 bpm), updated `memd/{decompose,select,diagnostics}.py`, updated
  `features/{fiducials,morphological,dynamic,extract}.py`, updated `viz/plots.py`, and
  two new modules: `mecg.py` (adaptive maternal-ECG cancellation -- measured finding:
  ADFECGDB r04/r07/r10 currently extract at the maternal rate, 82-93 bpm, without it)
  and `quality.py` (window/record quality gating against that silent failure mode).
  `configs/default.yaml` gained the corresponding `memd.abstain_fraction`, full `pinn:`
  section, `mecg_cancellation:`, `quality:`, and `features.{fiducials,morphological,
  dynamic}` sections; `cvae:`/`transformer:`/`evaluation:` sections were left untouched
  (out of scope, tracked separately). Ported scripts 00-04 to match (cinc2013 dataset
  choice, real fetal-band/PINN wiring in place of random placeholder tensors). Full
  suite: 112/112 passing (up from 47). -> CLAUDE.md §7 "keep engineering_log.md updated".
- 2026-09-29: Reinstated CinC 2013 as a second zero-shot validation cohort (train
  ADFECGDB only; validate NInFEA + CinC 2013) at Bhuvanesh's explicit request -- the
  ported code already treated it this way (`io/cinc2013.py`, `configs/datasets/
  cinc2013.yaml`) even though CLAUDE.md §2 still recorded it as dropped; updated §2/§3/§4
  and README to match. 25 records (a01-a25), WFDB, 4 abdominal channels already at the
  target 1000 Hz, `.fqrs` timing-only reference, no published GA, notch enabled (13/25
  records measured with strong 50 Hz mains). -> CLAUDE.md §2 "Datasets".
- 2026-09-28: Ported the top-level `cvae/` folder's working logic (InfoNCE projection
  head, GA-window positive pairing, KL warmup, AdamW mini-batch training, Mahalanobis
  OOD stats, latent diagnostics) into `src/fcac/cvae/`, then deleted the top-level
  folder -- `src/fcac/` is now the only place CVAE code lives. -> CLAUDE.md §4, §7.
- 2026-09-28: Ported the top-level `transformer/` folder's working logic (CLS-token
  pooling, padding-mask support, variable-length recording batching, LOSOCV +
  zero-shot-with-OOD-gating evaluation reusing `src/fcac/evaluation/`) into
  `src/fcac/transformer/`, then deleted the top-level folder. -> CLAUDE.md §4, §7.
- 2026-09-29: Git-hygiene correction. The `mecg.py`/`quality.py`/`pinn/coords.py`/
  `memd/sift.py`/`io/cinc2013.py` port (documented above, dated 2026-09-28) was left
  uncommitted when the cvae/transformer restructure plan started. Because `git add
  configs/default.yaml` and `git add README.md CLAUDE.md` stage a file's whole
  working-tree state, two of that plan's commits (`1adf620`, `3919c5d`) unintentionally
  swept in this already-uncommitted config/doc content -- including the CinC 2013 LOCKED
  §2 decision change -- under commit messages scoped to CVAE/transformer work. Caught by
  this plan's final review. The actual module files (`mecg.py` etc.) were committed only
  now, in this entry's commit, so `configs/default.yaml`/`CLAUDE.md`/`README.md` (already
  committed) and the code they describe are consistent as of this commit. No functional
  change; this is a historical-record correction only. -> CLAUDE.md §7 "keep
  engineering_log.md updated".
- 2026-09-29: Running scripts/01_preprocess.py on the real, correctly-pathed data
  (data/raw/{adfecgdb,ninfea,cinc2013}/) surfaced two real NInFEA channel-selection
  issues. (1) Recording 34's uni_abd9 electrode is dead (constant) -- scipy's kurtosis
  on a constant channel is NaN, and NaN sorted to the front of the existing
  `argsort(...)[::-1]`, so kurtosis_select was picking the dead channel as its #1
  choice. Fixed: non-finite kurtosis now ranks lowest (src/fcac/preprocessing/
  channels.py). (2) Recording 50 selects uni_abd24, a channel with legitimately high
  finite kurtosis (64.3) but a whole-recording std ~24,000x the other three selected
  channels -- almost certainly ADC railing/saturation producing spike artifacts that
  score high on pure kurtosis without being real signal. This is a known weakness of
  kurtosis-only bad-channel selection, not a code bug; CLAUDE.md §2 locks kurtosis-only
  selection with no amplitude-outlier guard, and per user decision this is left as-is
  -- `fcac.quality.assess_window`/`assess_record` (built for exactly this failure mode)
  is the intended downstream gate, not a preprocessing-stage fix. -> CLAUDE.md §2
  "Kurtosis channel selection".
- 2026-10-02: scripts/07 now trains the transformer on real CVAE latents (features -> CVAE posterior means, same median imputation as 06) instead of random tensors; refuses to run with null GA. ADFECGDB per-record GA is still unavailable, so `configs/experiments/synthetic_ga_smoketest.yaml` supplies seeded random 38-41 wk labels at user's request, stamped `synthetic_ga: true` in reports/07 and 08 -- pipeline check only, no number may be reported; swap for real GA in configs/datasets/adfecgdb.yaml when received. -> CLAUDE.md §2 Transformer, §7 (no fabricated values).
- 2026-10-02: At Bhuvanesh's request the CVAE and transformer now train on ADFECGDB + the 9 NInFEA recordings that pass quality gating (`pooled_training.datasets` in default.yaml), evaluated by pooled leave-one-subject-out CV grouped by mother (`dataset.subject_of_record` in ninfea.yaml, from Sulas 2021 Table 2 -- read via one WebFetch, NOT independently verified). The PINN stays ADFECGDB-only; CinC 2013 stays zero-shot. CLAUDE.md §2 Datasets/Evaluation amended accordingly; NInFEA is no longer a held-out set for CVAE/transformer, so there is no external validation until CinC 2013 has GA. First pooled run used synthetic ADFECGDB GA (pooled MAE 35.9 d, n=14) -- unreportable. -> CLAUDE.md §2 Datasets, Evaluation.
- 2026-10-02: scripts/08 now refits the CVAE (and its imputation medians) inside every LOSOCV fold on the training subjects only (`losocv_refit_cvae`), removing the shared-CVAE leakage; `--shared-cvae` keeps the old behaviour for comparison. Pooled synthetic-GA run: MAE 36.6 d refit vs 35.9 d shared (n=14) -- difference is small next to the CI, and both are unreportable (synthetic ADFECGDB GA). -> CLAUDE.md §2 Evaluation.
- 2026-10-02: CLAUDE.md §2 PINN amended (Bhuvanesh): an experimental PINN may train on ADFECGDB + NInFEA split by mother (scripts/pinn_ninfea_experiment.py). Result, 200 epochs, 2 folds vs the ADFECGDB-only PINN: NInFEA held-out gate pass rate 4.8% (15/315 windows, 3/60 usable records) vs 6.0% (19/315, 3/60); CinC 2013 QRS F1 0.380 vs 0.377; ADFECGDB (in-sample) 0.514 vs 0.510 -- no improvement, so the ADFECGDB-only PINN stays the default. Median extracted heart rate on NInFEA is ~58-61 bpm for both (maternal rate), i.e. the failure is maternal lock, not too little PINN training data. NInFEA QRS F1 is not scoreable (no Doppler timing on disk). -> CLAUDE.md §2 PINN.
- 2026-10-02: scripts/validate_cvae.py + fcac.evaluation.cvae_metrics validate the CVAE on held-out data (172 pooled windows, 14 subjects): held-out reconstruction R^2 median -0.01 (4/11 features beat the train mean), same-subject retrieval 0.24 vs chance 0.23 (raw features 0.33), kNN GA MAE 9.1 d vs 9.1 d for the training-mean baseline, 14-22 of 64 latent dims active. Not validated: inputs are unstandardised (large-scale features dominate the loss). -> CLAUDE.md §2 CVAE.
- 2026-10-02: Added an established-method alternative to the PINN: FastICA BSS (src/fcac/bss.py, `bss:` config, scikit-learn==1.9.1 pinned) with annotation-free fetal-component selection; compared in scripts/bss_experiment.py. QRS F1 CinC 2013 (zero-shot): BSS on preprocessed windows 0.519 vs cancel+PINN 0.440 vs PINN 0.377; usable records 22/25 vs 8/25. ADFECGDB (PINN in-sample): cancel+PINN 0.630 vs BSS 0.557. NInFEA gate pass 83% (55/60 usable, median HR 126) vs 6% (3/60) -- optimistic: BSS selection and the gate both reward a fetal-like rate, and gate-passing BSS windows score only F1 ~0.59 on ADFECGDB/CinC. NInFEA correctness is unverifiable (no Doppler timing on disk). -> CLAUDE.md §2 PINN (alternative extractor, not yet in the default pipeline).
- 2026-10-03: Hybrid extractor (per window: cancel+PINN output when it passes the quality gate, else FastICA BSS output; `arm_outputs` in scripts/bss_experiment.py): QRS F1 ADFECGDB 0.637 (in-sample for the PINN; cancel+PINN 0.630, BSS 0.557), CinC 2013 zero-shot 0.542 (BSS 0.519, cancel+PINN 0.440); NInFEA gate pass 84% / 55 of 60 usable (gate pass is optimistic for BSS). Best F1 of all arms on both labelled cohorts. -> CLAUDE.md §2 PINN / extraction.
- 2026-10-03: NInFEA fetal beat reference built from the dataset's pulsed-wave Doppler bitmaps (public: PhysioNet ninfea/1.0.0/pwd_images, 60 .bmp, ~545 MB; downloaded to data/raw/ninfea_pwd via `00_download.py --dataset ninfea_pwd`). src/fcac/io/ninfea_doppler.py ports the dataset's envelope_extraction.m (GPL-3, Ortu/Sulas/Pani) and adds peak detection (the dataset ships no beat annotations); 284 image columns/s on the ECG time origin (dataset viewer Interfaccia_dataset.m; image 1 duration 28.074 s vs ECG 28.071 s). Doppler peaks lag the R wave by an unknown offset, so this is a timing reference, not R-peak truth. -> CLAUDE.md §3 NInFEA, §2 Ground truth.
- 2026-10-03: All 60 NInFEA Doppler images downloaded and beat times extracted (60/60 ok, median 141 bpm, range 128-156; Doppler vs ECG duration agrees to <=66 ms, nearly all <5 ms). First NInFEA extraction scoring vs Doppler (scripts/eval_ninfea_doppler.py, windows with >=3 Doppler beats, global lag -62 ms estimated from BSS, concentration 0.15 vs ~0.10 unlocked): heart rate within 10 bpm of Doppler: hybrid 54%, BSS 52%, BSS-on-MEMD 16%, cancel+PINN 5%, PINN 2% (median error 9-10 bpm for BSS/hybrid vs 53-59 for the PINN arms); beat F1 @+/-50 ms 0.23 (BSS, hybrid) vs 0.17/0.14 (PINN arms), @+/-100 ms 0.43-0.44 vs 0.34/0.27. BSS gate-pass windows score F1 0.24 vs 0.19 for gate-fail, so the quality gate does not discriminate correctness on NInFEA. Hybrid is the best extractor on all three cohorts. -> CLAUDE.md §2 Ground truth, PINN/extraction.
- 2026-10-03: Hybrid extractor wired into the default pipeline (`extraction.method: hybrid`, src/fcac/extraction.py, scripts/04). Effect on usable data (gate pass): ADFECGDB 5/5 records, 153 -> 272 windows; NInFEA 3 -> 55 usable records, 9 -> 59 records with features, 19 -> 265 windows (NInFEA windows are mostly BSS output, 6% from the PINN; ~half have the right heart rate per the Doppler check). LOSOCV generalised to hold out all recordings of a mother together (several mothers now have >1 recording); CVAE/transformer trainers gained `device: auto` (GPU) and 08 gained `--workers N` for parallel folds (4 workers + GPU: ~35 min vs ~4 h on one CPU process).
- 2026-10-03: Pooled strict LOSOCV on hybrid features (64 recordings: 5 ADFECGDB with SYNTHETIC GA + 59 NInFEA with real GA; CVAE refit per fold): MAE 17.7 d (CI 14.2-22.1), r 0.68 vs predict-the-mean 20.6 d -- mostly cohort separation. NInFEA alone: MAE 17.6 d, r -0.11 vs 12.5 d for predicting the NInFEA mean, predictions range 144-286 d (true 147-194): no within-cohort age signal. ADFECGDB part unreportable (synthetic GA). -> CLAUDE.md §2 Evaluation.
- 2026-10-03: CVAE inputs are now standardised (`cvae.standardise_inputs: true`; training mean/std stored as `feat_mean`/`feat_std` buffers in the checkpoint; `ContrastiveVAE.standardise/destandardise`; constant features keep std 1). Validation on the hybrid-feature pool (537 windows, 44 subjects; scripts/validate_cvae.py, reports/validate_cvae_{nostd,std}.json), unscaled -> standardised: held-out per-feature R^2 (observed entries) median -0.01 -> 0.86, features beating the train mean 3/11 -> 11/11, active latent dims 19-39/64 -> 64/64 (no posterior collapse), same-subject retrieval 0.10 -> 0.14 (raw features 0.15, chance 0.06). Age information is still absent: kNN GA over 39 NInFEA mothers MAE 12.3 d (latent) / 11.1 d (raw features) / 11.4 d (predict the mean). Note latent_dim 64 > 12 input features, so high reconstruction R^2 shows the autoencoder works, not that the latent is informative. -> CLAUDE.md §2 CVAE.
- 2026-10-03: Within NInFEA (GA 147-194 d, a 6.7-week span) neither the Doppler fetal heart rate (Spearman rho -0.07 vs GA, 60 recordings) nor any of the 12 extracted features correlates significantly with GA (|rho| <= 0.2 for n >= 29; MSE/SampEn rho 0.42 but n = 14, p = 0.14). Within-cohort age prediction on NInFEA is therefore not supported by these features; the usable GA contrast is across cohorts (21-27 vs 38-41 wk), and the ADFECGDB GA labels are still synthetic.
- 2026-10-03: Bhuvanesh confirmed: bandpass stays 1-100 Hz (CLAUDE.md open question closed); hybrid (PINN + BSS) is the default extractor; the CVAE and transformer train on ADFECGDB + NInFEA (already amended 2026-10-02); PINN numbers are not to be reported until the coordinate scheme is approved. Added an "Extraction" row to CLAUDE.md §2. -> CLAUDE.md §2.
- 2026-10-03: Pooled strict LOSOCV rerun with the standardised CVAE (64 recordings, 5 ADFECGDB with SYNTHETIC GA + 59 NInFEA real GA): MAE 22.6 d (CI 17.6-28.3), r 0.56, vs 17.7 d unscaled and 20.6 d for predicting the mean. NInFEA alone: MAE 22.6 d vs 12.5 d predicting the NInFEA mean, within-cohort r -0.02, predictions 150-275 d (sd 25 vs true sd ~15). Standardising made the latent richer (64/64 active dims) but the transformer only fits more noise: consistent with there being no GA signal in the features within NInFEA. -> CLAUDE.md §2 Evaluation.
- 2026-10-03: scripts/analyse_feature_ga_signal.py -- do the 12 features track GA within NInFEA once wrong-heart-rate windows are removed (Doppler-oracle selection: detected rate within 10 bpm of the Doppler rate)? Of 315 NInFEA windows, 265 pass the gate, 170 have the right rate, 166 are both: the gate keeps 98% of right-rate windows but only 63% of the windows it keeps are right. Per-recording mean feature vs real GA (Spearman, 51 recordings, 166 windows): no feature significant after Bonferroni (p < 0.0045); best symbolic_dynamics rho -0.25 (p 0.08), qtc_sigma +0.15, p_dispersion +0.17; MSE/SampEn (n=12, mostly NaN at 5 s windows) rho +0.36, p 0.25 -- underpowered. Conclusion: noise from wrong-rate windows is not what hides the age signal; with ~50 recordings only |rho| >~ 0.4 would be detectable, so weak effects cannot be excluded but are too weak to build a predictor on. -> CLAUDE.md §2 CVAE/Evaluation.
- 2026-10-03: SNR/MSE vs the ADFECGDB scalp lead (scripts/eval_snr_adfecgdb.py; `gain_matched_snr_db` in evaluation/extraction.py, sign- and scale-free because PINN/BSS outputs are not trained to reproduce the reference amplitude; PINN is in-sample). Mean gain-matched SNR over all windows: PINN 0.43 dB, cancel+PINN 0.44, BSS 0.41, hybrid 0.45 (oracle best raw channel 0.33, best MEMD channel 0.57); unit-gain SNR (z-scored, sign-aligned, ICC 2026 form) -1.8 to -1.7 dB, MSE ~1.55; |r| ~0.22. Clean records r01/r08 ~0.9-1.2 dB, r04/r07/r10 ~0.0-0.15 dB. Allowing +/-40 ms shift raises it only to 0.55-0.62 dB (median best shift -2 ms), so timing offset is not the cause. SNR against the scalp lead does not separate the arms and is not comparable with the ICC 2026 CNN's 14.4 dB (trained to regress the scalp signal); beat-timing metrics (QRS F1, heart-rate agreement) are the informative ones. -> CLAUDE.md §2 Evaluation / §5 evaluation/extraction.py.
- 2026-10-03: Supervisor progress memo saved as reports/supervisor_summary_2026-10-03.html (also published as a private claude.ai page). Evidence files it cites are committed alongside: reports/bss_experiment.json, eval_ninfea_doppler.json, ninfea_doppler_beats.json, pinn_ninfea_experiment.json, 08_evaluate.json (ADFECGDB GA in that run is SYNTHETIC, stamped synthetic_ga: true -- not reportable), plus the already-tracked eval_snr_adfecgdb.json, analyse_feature_ga_signal.json and validate_cvae_{std,nostd}.json.
