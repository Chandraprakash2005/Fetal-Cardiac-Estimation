#!/usr/bin/env python3
"""scripts/bss_experiment.py

Does an established BSS extractor (FastICA, src/fcac/bss.py) recover the fetal ECG better than
the PINN, especially on NInFEA where the PINN pipeline locks onto the maternal rhythm?

Arms (same windows, same scoring):
  pinn          existing ADFECGDB-only PINN on the MEMD fetal-band windows
  cancel+pinn   maternal cancellation (mecg) -> PINN: the pipeline scripts 04/05 run
  bss_pre       FastICA on the preprocessed abdominal windows (4 ch), fetal component picked
  bss_memd      FastICA on the MEMD fetal-band windows (4 ch), fetal component picked
  hybrid        per window: the cancel+pinn output if it passes the quality gate, else bss_pre

Metrics per cohort
  * quality-gate pass rate, median heart rate, usable records (all cohorts)
  * QRS F1 / precision / recall within +/-50 ms where reference beats exist
    (ADFECGDB .qrs, CinC 2013 .fqrs; NInFEA's Doppler timing is not on disk)
  * F1 of gate-passing vs gate-failing windows: how far the gate can be trusted as a proxy
    for correctness on NInFEA
CAVEAT: the BSS component selection and the gate both reward a plausible fetal rate, so BSS
gate pass-rates are optimistic; F1 on ADFECGDB/CinC (annotations never used in selection) is
the check.

Usage: python scripts/bss_experiment.py [--datasets adfecgdb cinc2013 ninfea]

Implements: CLAUDE.md §2 PINN/extraction (alternative-method comparison).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_zeroshot as ez  # noqa: E402

from fcac.bss import extract_fecg_bss  # noqa: E402
from fcac.config import load_config  # noqa: E402
from fcac.extraction import extract_with_method  # noqa: E402
from fcac.mecg import adaptive_cancel_windows  # noqa: E402
from fcac.pinn.infer import extract_fecg  # noqa: E402
from fcac.pinn.model import PINN  # noqa: E402
from fcac.quality import assess_window  # noqa: E402

logging.basicConfig(level=logging.WARNING)
ARMS = ("pinn", "cancel+pinn", "bss_pre", "bss_memd", "hybrid")


def arm_outputs(x: np.ndarray, pre: np.ndarray, fs: float, cfg, model: PINN):
    """Single-channel fECG (W, N) per arm for one record, plus the per-window mask of windows the
    hybrid took from cancel+PINN (those that pass the quality gate).

    x : MEMD fetal-band windows (W, 4, N); pre : preprocessed abdominal windows (W, 4, N)."""
    xc, _ = adaptive_cancel_windows(x, fs, cfg)
    outputs = {
        "pinn": extract_fecg(model, x, cfg).fecg,
        "cancel+pinn": extract_fecg(model, xc, cfg).fecg,
        "bss_pre": extract_fecg_bss(pre, fs, cfg)[0],
        "bss_memd": extract_fecg_bss(x, fs, cfg)[0],
    }
    outputs["hybrid"], use_pinn, _ = extract_with_method(x, pre, fs, cfg, model, method="hybrid")
    return outputs, use_pinn


def run_arms(dataset: str, model: PINN) -> dict:
    cfg = load_config(dataset=dataset)
    rows = {a: [] for a in ARMS}
    for band_path in sorted((Path("data/processed/fetal_band") / dataset).glob("*.npz")):
        rid = band_path.stem
        band = np.load(band_path, allow_pickle=True)
        x, fs = band["x"], float(band["fs"])
        pre = np.load(Path(cfg.dataset.interim_dir) / f"{rid}.npz", allow_pickle=True)["windows"]
        outputs, use_pinn = arm_outputs(x, pre, fs, cfg, model)
        peaks = ez.reference_peaks(dataset, rid, cfg)
        for arm, fecg in outputs.items():
            for w in range(fecg.shape[0]):
                q = assess_window(fecg[w], fs, cfg)
                row = {"record": rid, "window": w, "pass": bool(q.acceptable), "hr": q.heart_rate_bpm,
                       "from_pinn": bool(use_pinn[w])}
                if peaks is not None:
                    ref = ez.peaks_for_window(peaks, w, cfg, fs)
                    if ref.size >= 3:
                        row["f1"], row["precision"], row["recall"], _ = ez.score_signal(fecg[w], ref, fs, 50.0)
                rows[arm].append(row)
    return {arm: summarise(r, cfg) for arm, r in rows.items()}


def summarise(rows: list, cfg) -> dict:
    n = len(rows)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["record"], []).append(r)
    usable = sum(np.mean([x["pass"] for x in v]) >= cfg.quality.min_acceptable_window_fraction for v in by_rec.values())
    rec_hr = [np.nanmedian([x["hr"] for x in v]) for v in by_rec.values()]
    out = {"n_windows": n, "n_records": len(by_rec), "gate_pass_rate": float(np.mean([r["pass"] for r in rows])),
           "usable_records": int(usable), "median_record_hr_bpm": float(np.nanmedian(rec_hr))}
    out["share_from_pinn"] = float(np.mean([r["from_pinn"] for r in rows]))
    scored = [r for r in rows if "f1" in r]
    if scored:
        for k in ("f1", "precision", "recall"):
            out[f"mean_{k}"] = float(np.mean([r[k] for r in scored]))
        out["n_scored"] = len(scored)
        for label, flag in (("passing", True), ("failing", False)):
            sel = [r["f1"] for r in scored if r["pass"] == flag]
            out[f"f1_gate_{label}"] = {"mean": float(np.mean(sel)) if sel else None, "n": len(sel)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["adfecgdb", "cinc2013", "ninfea"])
    args = parser.parse_args()

    cfg = load_config(dataset="adfecgdb")
    model = PINN(cfg)
    model.load_state_dict(torch.load("data/processed/pinn/pinn.pt", map_location="cpu", weights_only=True))
    model.eval()

    results = {}
    for ds in args.datasets:
        print(f"== {ds}", flush=True)
        results[ds] = run_arms(ds, model)
        for arm, s in results[ds].items():
            f1 = f"F1 {s['mean_f1']:.3f} (gate-pass {s['f1_gate_passing']['mean'] if s['f1_gate_passing']['mean'] is None else round(s['f1_gate_passing']['mean'], 3)} / " \
                 f"gate-fail {round(s['f1_gate_failing']['mean'], 3) if s['f1_gate_failing']['mean'] is not None else None})  " if "mean_f1" in s else ""
            print(f"  {arm:12s} gate pass {s['gate_pass_rate']:.3f}  usable {s['usable_records']}/{s['n_records']}  "
                  f"median HR {s['median_record_hr_bpm']:.0f}  {f1}", flush=True)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/bss_experiment.json", "w") as fh:
        json.dump({"results": results, "timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    print("Wrote reports/bss_experiment.json")


if __name__ == "__main__":
    main()
