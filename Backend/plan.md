# Open Inputs Needed — Fetal Cardiac Ageing Clock

Tracks what still needs to come from Bhuvanesh / the team / the supervisor / the spec PDF
before each module can move past its current scaffold/TODO state. See `CLAUDE.md` for the
locked decisions already made; this file is only the *open* ones.

Checked against the real downloaded data in `data/adfecgdb/` and `data/ninfea/` as of
2026-09-07 — items below are resolvable now confirmed as such.

## Already resolved — no input needed
- Preprocessing (filters, kurtosis selection, resampling, z-score, windowing) — fully
  implemented, works on the real data already on disk.
- ADFECGDB structure — `r01/r04/r07/r08/r10`, `.edf`+`.qrs`, matches `io/adfecgdb.py`.
- NInFEA record naming — files are just numbered `1.dat`/`1.hea` ... `60.dat`/`60.hea` with a
  `RECORDS` index; wire directly into `scripts/01_preprocess.py`, no need to ask.

## 1. Gestational Age labels — biggest blocker, needed before anything trains

**NInFEA: RESOLVED 2026-09-30.** Real per-record GA for all 60 recordings is now in
`configs/datasets/ninfea.yaml`, sourced from Table 3 of the dataset's own companion
paper (Sulas et al. 2021, Scientific Data, https://doi.org/10.1038/s41597-021-00811-3,
PMC7838287) — not on the PhysioNet page or in the `.hea` headers, only in that paper's
main text. Transcribed via automated fetch, not independently re-verified against the
published table — spot-check a few values before relying on it for anything reported.

**ADFECGDB: still blocked, and genuinely so.** Confirmed via real research (not just
"checked the headers"): PhysioNet's adfecgdb page has no metadata file (only the 5
`.edf`/`.edf.qrs` pairs + `RECORDS`/`ANNOTATORS`/`SHA256SUMS.txt`); the original 2012
paper (Jezewski et al., Biomed Eng/Biomed Tech) and the newer 2020 Scientific Data
paper that re-published these same 5 records as part of a larger set (Matonia et al.,
https://doi.org/10.1038/s41597-020-0538-z, PMC7316827) both state per-patient GA was
recorded, but neither paper's *main text* tables include it (their tables cover SNR
and FHR variability instead). The one lead — "a machine-accessible metadata file",
figshare DOI 10.6084/m9.figshare.12280007, and the actual dataset's figshare page
(springernature.figshare.com/.../10311029) — is blocked by figshare's bot protection
(403 Forbidden) from automated fetching. **Need:** someone opens either figshare link
in a real browser and checks whether it has a per-patient GA table/file. If not, the
paper's authors would need to be contacted directly. Until then, ADFECGDB stays at
the aggregate 38-41 week range only, `configs/datasets/adfecgdb.yaml` GA is still `null`.

**NInFEA Doppler ground truth: also newly found, not yet acted on.** The Doppler data
`load_ninfea_doppler_beats` has always needed is not missing/protected as previously
assumed — confirmed via PhysioNet page that it's openly downloadable (no credentialing),
just never fetched into `data/raw/ninfea/`. It's `pwd_images/` on the NInFEA project
page: pulsed-wave Doppler ultrasound sweeps saved as **bitmap images**
(1680x1050px, 60Hz), not a signal channel — so building real NInFEA ground truth means
downloading those images and writing an image-processing parser (tracing the Doppler
envelope in each frame to get beat timestamps), not a simple file-format parser. Real,
now-unblocked, but a materially bigger task than "read a file" — scoped separately.

## 2. MEMD (`src/fcac/memd/decompose.py`)
- **Need:** which MEMD implementation to vendor — e.g. Rehman & Mandic's `MEMD_all.py`, or a
  PyPI package. Need the source/repo/commit to pin (license matters, it gets committed).
- Everything else (directions=64, scoring, selection) is implemented; just needs the real
  `memd()` call wired in.

## 3. PINN (`src/fcac/pinn/model.py`, `physics.py`, `losses.py`)
- **Need:** physical meaning of the input coordinates. CLAUDE.md says "enforce ∇²V=0" but not
  what the coordinate axes are — time only? electrode (x,y,z) positions + time? Determines
  `input_dim`/`output_dim` (currently placeholders in `configs/default.yaml`) and what
  `L_data` regresses against (likely isolated fetal ECG amplitude, needs confirming).
  If spec PDF §7.2 has this detail, that answers it; otherwise ask the supervisor.

## 4. Features (`src/fcac/features/{fiducials,morphological,dynamic}.py`)
CLAUDE.md says feature order comes from spec §8 — sharing that section answers most of these
at once:
- Fiducial detection algorithm (P/Q/R/S/T) — Pan-Tompkins-derivative? Wavelet-based?
- QTc correction — Bazett vs. Fridericia vs. other
- P-dispersion, QRS asymmetry — exact formula
- PRSA — anchor/scale definition
- MSE, ApEn, SampEn — scale range / `m`, `r` parameters
- DFA-α — short-term vs. long-term fitting range
- LF/HF — band edges (adult 0.04-0.15 / 0.15-0.4 Hz don't apply to fetal HRV)
- Symbolic dynamics — alphabet size, word length, summary statistic

## 5. CVAE (`src/fcac/cvae/losses.py`)
- **Need:** relative weighting of recon/KL/InfoNCE (currently summed 1:1:1 — the PINN has an
  explicit 0.7/0.3, this doesn't). Minor — can tune empirically once real features exist;
  flagging in case the supervisor already specified one.
- **Need clarifying:** "same-GA windows" as a positive pair — exact GA match, or a tolerance
  bucket (days/weeks)?

## 6. Evaluation
- Nothing new needed — `verify_units.py`, `losocv.py`, bootstrap CI are done; they just need
  real upstream outputs to run against.

## 7. Open architectural question (already flagged in CLAUDE.md, not urgent)
- Bandpass preset: paper's 1-100 Hz (current default) vs. spec's 0.5-150 Hz — pending
  supervisor confirmation. Both exist as named presets already in `configs/default.yaml`.

---
**Fastest path to unblock the most modules at once:** GA metadata (#1) and spec PDF §7 (PINN)
+ §8 (features) — those two alone resolve the majority of the list above. Everything else can
keep proceeding with placeholders/random tensors until then, as it already does.
