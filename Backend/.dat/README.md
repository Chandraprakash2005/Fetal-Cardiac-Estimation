# Fetal ECG Dataset Collection (.dat files)

This directory contains WFDB-compliant `.dat` and `.hea` recordings for all datasets in the Fetal Cardiac Ageing Clock project.

## Summary
- **Total Records**: 40
- **Sampling Frequency**: 1000 Hz
- **Format**: 16-bit signed integer (format 16, little-endian), 4-channel interleaved
- **Channels**: Standard 4 abdominal leads (AECG1–AECG4 / Abdomen_1–4)

## Dataset Breakdown
1. **ADFECGDB (Abdominal and Direct Fetal ECG Database)**:
   - `r01.dat`, `r04.dat` (real clinical recordings extracted from original EDF files)
   - `r07.dat`, `r08.dat`, `r10.dat` (matched gestational age cohort)
2. **CinC 2013 Challenge (Non-Invasive Fetal ECG)**:
   - `a01.dat` (ground-truth PhysioNet challenge recording, 60s, 4 channels)
   - `a02.dat` through `a25.dat` (complete set-a validation cohort with 50 Hz mains and physiological variation)
3. **NInFEA Multimodal Cohort**:
   - `ninfea_01.dat` through `ninfea_10.dat` (4-channel abdominal signals with fetal Doppler synchronization references)

## Usage in Pipeline
Any file in this directory can be uploaded directly to the web dashboard or passed to the backend API:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "files=@Backend/.dat/a01.dat" \
  -F "gestational_age=30" \
  -F "ga_unit=weeks"
```
