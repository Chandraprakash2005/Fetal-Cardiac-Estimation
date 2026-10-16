#!/usr/bin/env python3
"""scripts/eval_ninfea_doppler.py

Scores every extraction arm on NInFEA against the Doppler-derived fetal beat times
(scripts/ninfea_doppler_beats.py), the first reference NInFEA has had here.

A Doppler peak trails the ECG R wave by an unknown lag, so
  1. the lag is estimated ONCE, globally: the mode of (Doppler beat - detected R) time
     differences over the windows of the `--lag-arm` extractor (default bss_pre), then
  2. every arm is scored with the Doppler beats shifted back by that lag.
Reported per arm (windows with >= 3 Doppler beats):
  * beat F1 / precision / recall at +/-50 ms and +/-100 ms (lag-corrected)
  * heart-rate agreement (OFFSET-FREE): share of windows within 10 bpm of the Doppler rate
  * the same F1 split by quality-gate pass/fail -- how far the gate can be trusted on NInFEA
  * the lag histogram concentration per arm: a maternal-locked arm has no lag structure.
Arms: pinn, cancel+pinn, bss_pre, bss_memd, hybrid (see scripts/bss_experiment.py).

Usage: python scripts/eval_ninfea_doppler.py [--lag-arm bss_pre]

Implements: CLAUDE.md §2 "Ground truth" (NInFEA Doppler -> timing metrics only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bss_experiment as bx  # noqa: E402

from fcac.config import load_config  # noqa: E402
from fcac.evaluation.doppler_ref import estimate_lag, heart_rate_bpm, lag_differences  # noqa: E402
from fcac.evaluation.extraction import qrs_f1  # noqa: E402
from fcac.memd.diagnostics import pan_tompkins_peaks  # noqa: E402
from fcac.pinn.model import PINN  # noqa: E402
from fcac.quality import assess_window  # noqa: E402


def collect(cfg, cfg_pinn, model) -> list[dict]:
    """One row per (record, window, arm): detected beat times, gate verdict, Doppler beats in window."""
    cache = Path(cfg.dataset.doppler.cache_dir)
    trim = float(cfg.preprocessing.edge_trim_seconds)
    win_s = cfg.preprocessing.window.size_samples / float(cfg.preprocessing.target_fs_hz)
    stride_s = cfg.preprocessing.window.stride_samples / float(cfg.preprocessing.target_fs_hz)
    rows = []
    for band_path in sorted(Path("data/processed/fetal_band/ninfea").glob("*.npz"), key=lambda p: int(p.stem)):
        rid = band_path.stem
        if not (cache / f"{rid}.npz").exists():
            continue
        doppler = np.load(cache / f"{rid}.npz")["beat_times_s"]
        band = np.load(band_path, allow_pickle=True)
        x, fs = band["x"], float(band["fs"])
        pre = np.load(Path(cfg.dataset.interim_dir) / f"{rid}.npz", allow_pickle=True)["windows"]
        outputs, _ = bx.arm_outputs(x, pre, fs, cfg_pinn, model)
        for arm, fecg in outputs.items():
            for w in range(fecg.shape[0]):
                start = trim + w * stride_s
                ref = doppler[(doppler >= start) & (doppler < start + win_s)]
                det = start + pan_tompkins_peaks(fecg[w], fs) / fs
                rows.append({"record": rid, "arm": arm, "window": w, "start": start, "det_s": det, "ref_s": ref,
                             "gate": bool(assess_window(fecg[w], fs, cfg_pinn).acceptable)})
    return rows


def score(rows: list[dict], lag: float, fs: float, tol_ms: float) -> dict:
    out = {"f1": [], "precision": [], "recall": [], "hr_ok": [], "hr_err": [], "gate": []}
    for r in rows:
        if len(r["ref_s"]) < 3:
            continue
        ref_local = np.round((r["ref_s"] - lag - r["start"]) * fs)
        det_local = np.round((r["det_s"] - r["start"]) * fs)
        f1, p, rec = qrs_f1(det_local, ref_local, fs, tolerance_ms=tol_ms)
        hr_ref, hr_det = heart_rate_bpm(r["ref_s"]), heart_rate_bpm(r["det_s"])
        out["f1"].append(f1); out["precision"].append(p); out["recall"].append(rec)
        err = abs(hr_det - hr_ref) if np.isfinite(hr_det) else float("nan")
        out["hr_err"].append(err); out["hr_ok"].append(bool(np.isfinite(err) and err <= 10.0))
        out["gate"].append(r["gate"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lag-arm", default="bss_pre")
    args = parser.parse_args()

    cfg = load_config(dataset="ninfea")
    cfg_pinn = load_config(dataset="adfecgdb")
    model = PINN(cfg_pinn)
    model.load_state_dict(torch.load("data/processed/pinn/pinn.pt", map_location="cpu", weights_only=True))
    model.eval()
    fs = float(cfg.preprocessing.target_fs_hz)

    rows = collect(cfg, cfg_pinn, model)
    records = sorted({r["record"] for r in rows}, key=int)
    print(f"{len(records)} recordings with Doppler beats: {records}", flush=True)

    # Global lag from the chosen arm's windows (all of them: the lag is a property of the Doppler
    # vs R-wave physiology, so it must not depend on the gate).
    diffs = np.concatenate([lag_differences(r["det_s"], r["ref_s"]) for r in rows if r["arm"] == args.lag_arm]
                           or [np.empty(0)])
    lag, conc = estimate_lag(diffs)
    print(f"estimated Doppler lag from {args.lag_arm}: {lag * 1000:.0f} ms (concentration {conc:.2f}; "
          f"flat/unlocked ~0.10)", flush=True)

    results = {}
    for arm in ("pinn", "cancel+pinn", "bss_pre", "bss_memd", "hybrid"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        d_arm = np.concatenate([lag_differences(r["det_s"], r["ref_s"]) for r in arm_rows] or [np.empty(0)])
        own_lag, own_conc = estimate_lag(d_arm)
        s50, s100 = score(arm_rows, lag, fs, 50.0), score(arm_rows, lag, fs, 100.0)
        gate = np.array(s50["gate"], dtype=bool)
        f1 = np.array(s50["f1"])
        results[arm] = {
            "n_windows_scored": len(f1),
            "f1_50ms": float(f1.mean()), "precision_50ms": float(np.mean(s50["precision"])),
            "recall_50ms": float(np.mean(s50["recall"])), "f1_100ms": float(np.mean(s100["f1"])),
            "hr_within_10bpm": float(np.mean(s50["hr_ok"])),
            "median_hr_error_bpm": float(np.nanmedian(s50["hr_err"])),
            "f1_50ms_gate_pass": {"mean": float(f1[gate].mean()) if gate.any() else None, "n": int(gate.sum())},
            "f1_50ms_gate_fail": {"mean": float(f1[~gate].mean()) if (~gate).any() else None, "n": int((~gate).sum())},
            "own_lag_ms": None if not np.isfinite(own_lag) else own_lag * 1000, "own_lag_concentration": own_conc,
        }
        r = results[arm]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"  {arm:12s} F1@50 {r['f1_50ms']:.3f} @100 {r['f1_100ms']:.3f}  HR within 10 bpm {r['hr_within_10bpm']:.2f} "
              f"(median err {r['median_hr_error_bpm']:.0f})  F1 gate-pass {fmt(r['f1_50ms_gate_pass']['mean'])} "
              f"(n={r['f1_50ms_gate_pass']['n']}) / fail {fmt(r['f1_50ms_gate_fail']['mean'])}  "
              f"own lag {r['own_lag_ms'] if r['own_lag_ms'] is None else round(r['own_lag_ms'])} ms conc {own_conc:.2f}", flush=True)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/eval_ninfea_doppler.json", "w") as fh:
        json.dump({"records": records, "lag_arm": args.lag_arm, "lag_ms": lag * 1000, "lag_concentration": conc,
                   "results": results, "timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    print("Wrote reports/eval_ninfea_doppler.json")


if __name__ == "__main__":
    main()
