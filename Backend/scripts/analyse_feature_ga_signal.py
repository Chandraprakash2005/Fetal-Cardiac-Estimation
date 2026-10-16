#!/usr/bin/env python3
"""scripts/analyse_feature_ga_signal.py

Diagnostic: do the 12 extracted features track gestational age within NInFEA once windows with
a WRONG fetal heart rate are removed?

About half of the hybrid extractor's NInFEA windows carry the wrong rate (maternal or noise), so
their features are noise and could be hiding an age signal. NInFEA's Doppler gives the true
fetal rate per window (scripts/ninfea_doppler_beats.py), so each window is labelled correct when
its detected heart rate is within `--tol-bpm` of the Doppler rate. Using the Doppler this way is an
ORACLE selection (not available at inference) -- the point is only to learn whether a signal
exists when the data are clean.

Reported per feature (Spearman, per-recording mean over the selected windows vs real GA):
  all_gated      the windows the pipeline keeps today (quality gate)
  gated_correct  gated AND right heart rate
  all_correct    right heart rate regardless of the gate (features recomputed for every window)
With ~40-60 recordings, |rho| must exceed roughly 0.4 to be significant after correcting for 11
features (Bonferroni, p < 0.0045); recordings of the same mother are not independent.

Usage: python scripts/analyse_feature_ga_signal.py [--tol-bpm 10]

Implements: CLAUDE.md §2 "CVAE"/"Evaluation" (feature diagnostics).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from fcac.config import load_config
from fcac.evaluation.doppler_ref import heart_rate_bpm
from fcac.features.extract import FEATURE_NAMES, extract_features
from fcac.memd.diagnostics import pan_tompkins_peaks

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol-bpm", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(dataset="ninfea")
    ga = cfg.dataset.ga_days
    cache = Path(cfg.dataset.doppler.cache_dir)
    trim = float(cfg.preprocessing.edge_trim_seconds)
    win_s = cfg.preprocessing.window.size_samples / float(cfg.preprocessing.target_fs_hz)
    stride_s = cfg.preprocessing.window.stride_samples / float(cfg.preprocessing.target_fs_hz)

    # per recording -> {"ga", "gated": rows, "gated_correct": rows, "all_correct": rows}
    per_rec = {}
    counts = {"windows": 0, "with_doppler": 0, "gated": 0, "correct": 0, "gated_correct": 0}
    for path in sorted(Path("data/processed/fecg/ninfea").glob("*.npz"), key=lambda p: int(p.stem)):
        rid = path.stem
        if not (cache / f"{rid}.npz").exists():
            continue
        doppler = np.load(cache / f"{rid}.npz")["beat_times_s"]
        z = np.load(path, allow_pickle=True)
        fecg, fs, gate = z["fecg"], float(z["fs"]), z["quality_mask"].astype(bool)
        entry = {"ga": float(ga[rid]), "gated": [], "gated_correct": [], "all_correct": []}
        for w in range(fecg.shape[0]):
            counts["windows"] += 1
            start = trim + w * stride_s
            ref = doppler[(doppler >= start) & (doppler < start + win_s)]
            hr_ref = heart_rate_bpm(ref)
            if not np.isfinite(hr_ref):
                continue
            counts["with_doppler"] += 1
            hr_det = heart_rate_bpm(start + pan_tompkins_peaks(fecg[w], fs) / fs)
            correct = bool(np.isfinite(hr_det) and abs(hr_det - hr_ref) <= args.tol_bpm)
            if not (gate[w] or correct):
                continue
            feats = extract_features(fecg[w], fs, cfg)
            counts["gated"] += int(gate[w]); counts["correct"] += int(correct)
            counts["gated_correct"] += int(gate[w] and correct)
            if gate[w]:
                entry["gated"].append(feats)
            if gate[w] and correct:
                entry["gated_correct"].append(feats)
            if correct:
                entry["all_correct"].append(feats)
        per_rec[rid] = entry
    print(f"windows with Doppler beats: {counts['with_doppler']} of {counts['windows']}; gated {counts['gated']}, "
          f"right heart rate {counts['correct']}, both {counts['gated_correct']}")

    results = {}
    for subset in ("all_gated", "gated_correct", "all_correct"):
        key = {"all_gated": "gated"}.get(subset, subset)
        recs = [(e["ga"], np.nanmean(np.stack(e[key]), axis=0), len(e[key])) for e in per_rec.values() if e[key]]
        G = np.array([r[0] for r in recs]); F = np.array([r[1] for r in recs]); n_win = sum(r[2] for r in recs)
        feat = {}
        for j, name in enumerate(FEATURE_NAMES):
            ok = np.isfinite(F[:, j])
            if ok.sum() >= 10 and np.nanstd(F[ok, j]) > 0:
                rho, p = spearmanr(G[ok], F[ok, j])
                feat[name] = {"rho": float(rho), "p": float(p), "n_recordings": int(ok.sum())}
        results[subset] = {"n_recordings": len(recs), "n_windows": int(n_win), "features": feat}
        sig = [k for k, v in feat.items() if v["p"] < 0.05 / max(1, len(FEATURE_NAMES) - 1)]
        print(f"\n[{subset}] {len(recs)} recordings, {n_win} windows; significant after Bonferroni: {sig or 'none'}")
        for name, v in feat.items():
            print(f"   {name:18s} rho={v['rho']:+.2f} p={v['p']:.3f} (n={v['n_recordings']})")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/analyse_feature_ga_signal.json", "w") as fh:
        json.dump({"tol_bpm": args.tol_bpm, "counts": counts, "results": results,
                   "timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)


if __name__ == "__main__":
    main()
