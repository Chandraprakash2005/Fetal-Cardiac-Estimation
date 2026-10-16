#!/usr/bin/env python3
"""scripts/accuracy_table.py

Per-record extraction accuracy table for one cohort: how much of the fetal ECG the
chain actually recovers, stage by stage.

Every column is fully unsupervised and single-output, so they are directly comparable:
each (C, N) stage is collapsed to one channel by PC1 before R-peak detection. (An
earlier "best of 4 channels" baseline needed the ground truth to pick the channel, which
flatters the baseline and is not something you could do at inference.)

  raw    : preprocessed abdominal window, PC1        -- no extraction at all
  MEMD   : stage 02 output, PC1                      -- noise removed, mother + baby
  PINN   : stage 03/04 output (cfg.pinn.readout)     -- the isolated fetal ECG

Metrics are QRS F1 / precision / recall against the cohort's reference fetal beat
annotations, matched within +/-50 ms, plus the detected heart rate against the
reference rate.

Usage:
    python scripts/accuracy_table.py --dataset cinc2013
    python scripts/accuracy_table.py --dataset adfecgdb --checkpoint data/processed/pinn/pinn.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from fcac.config import load_config
from fcac.evaluation.extraction import qrs_f1
from fcac.mecg import adaptive_cancel
from fcac.quality import assess_window
from fcac.memd.diagnostics import pan_tompkins_peaks
from fcac.pinn.coords import project_to_single_channel, readout, window_to_coords
from fcac.pinn.model import PINN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_DATASET = "adfecgdb"


def reference_peaks(dataset: str, record_id: str, cfg):
    if dataset == "adfecgdb":
        from fcac.io.adfecgdb import load_adfecgdb_qrs_annotations

        return load_adfecgdb_qrs_annotations(record_id, cfg)
    if dataset == "cinc2013":
        from fcac.io.cinc2013 import load_cinc2013_fqrs

        return load_cinc2013_fqrs(record_id, cfg)
    return None  # NInFEA: no annotations in the WFDB download


def bpm(peaks: np.ndarray, fs: float) -> float:
    if len(peaks) < 2:
        return float("nan")
    return 60.0 / (float(np.median(np.diff(peaks))) / fs)


def evaluate_record(record_id: str, dataset: str, model: PINN, cfg, tolerance_ms: float):
    band = np.load(
        Path("data/processed/fetal_band") / dataset / f"{record_id}.npz", allow_pickle=True
    )
    windows = band["x"]
    fs = float(band["fs"])
    raw = np.load(
        Path(cfg.dataset.interim_dir) / f"{record_id}.npz", allow_pickle=True
    )["windows"]

    peaks_raw = reference_peaks(dataset, record_id, cfg)
    if peaks_raw is None:
        return None

    trim = int(cfg.preprocessing.edge_trim_seconds * fs)
    stride = cfg.preprocessing.window.stride_samples
    size = cfg.preprocessing.window.size_samples

    acc = {k: [] for k in ("raw", "memd", "pinn")}
    prec, rec, hr_pinn, hr_ref, passed = [], [], [], [], []

    for w in range(windows.shape[0]):
        local = peaks_raw - (trim + w * stride)
        local = local[(local >= 0) & (local < size)]
        if local.size < 3:
            continue

        # Mirror scripts/04_extract_fecg.py exactly: adaptive maternal cancellation runs
        # before the PINN, so evaluating without it would not measure the real pipeline.
        window, _ = adaptive_cancel(windows[w], fs, cfg)
        coords, _ = window_to_coords(window, cfg)
        with torch.no_grad():
            v_pred = model(coords)
        fecg = readout(window, v_pred, cfg)

        for key, signal in (
            ("raw", project_to_single_channel(raw[w])),
            ("memd", project_to_single_channel(windows[w])),   # pre-cancellation stage 02
            ("pinn", fecg),
        ):
            f1, p, r = qrs_f1(pan_tompkins_peaks(signal, fs), local, fs, tolerance_ms)
            acc[key].append(f1)
            if key == "pinn":
                prec.append(p)
                rec.append(r)

        detected = pan_tompkins_peaks(fecg, fs)
        hr_pinn.append(bpm(detected, fs))
        hr_ref.append(bpm(local, fs))
        passed.append(assess_window(fecg, fs, cfg).acceptable)

    if not acc["pinn"]:
        return None
    return {
        "record_id": record_id,
        "n_windows": len(acc["pinn"]),
        "f1_raw": float(np.mean(acc["raw"])),
        "f1_memd": float(np.mean(acc["memd"])),
        "f1_pinn": float(np.mean(acc["pinn"])),
        "precision_pinn": float(np.mean(prec)),
        "recall_pinn": float(np.mean(rec)),
        "hr_pinn": float(np.nanmean(hr_pinn)),
        "hr_ref": float(np.nanmean(hr_ref)),
        "quality_pass_rate": float(np.mean(passed)) if passed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["adfecgdb", "ninfea", "cinc2013"])
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    args = parser.parse_args()

    cfg = load_config(dataset=args.dataset)
    model = PINN(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    paths = sorted((Path("data/processed/fetal_band") / args.dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No fetal-band windows for {args.dataset}; run scripts 01 and 02 first."
        )

    rows = []
    for path in paths:
        row = evaluate_record(path.stem, args.dataset, model, cfg, args.tolerance_ms)
        if row is not None:
            rows.append(row)
        logger.info("scored %s", path.stem)

    if not rows:
        print(f"\n{args.dataset}: no reference annotations on disk -- nothing to score.\n")
        return

    role = "TRAIN (in-sample)" if args.dataset == TRAIN_DATASET else "TEST (zero-shot)"
    print("\n" + "=" * 84)
    print(f"EXTRACTION ACCURACY -- {args.dataset}   [{role}]")
    print(f"PINN trained on {TRAIN_DATASET}; readout={cfg.pinn.readout}; "
          f"QRS F1 +/-{args.tolerance_ms:.0f} ms; all columns PC1-projected")
    print("=" * 84)
    header = (f"{'record':<8}{'wins':>5}{'raw':>8}{'MEMD':>8}{'PINN':>8}"
              f"{'prec':>8}{'recall':>8}{'HR_pinn':>9}{'HR_ref':>8}{'qual':>7}")
    print(header)
    print("-" * 84)
    for r in rows:
        flag = "" if r["quality_pass_rate"] >= cfg.quality.min_acceptable_window_fraction else "  GATED"
        print(f"{r['record_id']:<8}{r['n_windows']:>5}{r['f1_raw']:>8.3f}{r['f1_memd']:>8.3f}"
              f"{r['f1_pinn']:>8.3f}{r['precision_pinn']:>8.3f}{r['recall_pinn']:>8.3f}"
              f"{r['hr_pinn']:>9.1f}{r['hr_ref']:>8.1f}{r['quality_pass_rate']:>6.0%}{flag}")
    print("-" * 84)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in
            ("f1_raw", "f1_memd", "f1_pinn", "precision_pinn", "recall_pinn")}
    print(f"{'MEAN':<8}{sum(r['n_windows'] for r in rows):>5}{mean['f1_raw']:>8.3f}"
          f"{mean['f1_memd']:>8.3f}{mean['f1_pinn']:>8.3f}"
          f"{mean['precision_pinn']:>8.3f}{mean['recall_pinn']:>8.3f}")
    print("=" * 84)
    kept = [r for r in rows if r["quality_pass_rate"] >= cfg.quality.min_acceptable_window_fraction]
    if kept:
        print(f"{'KEPT':<8}{sum(r['n_windows'] for r in kept):>5}"
              f"{np.mean([r['f1_raw'] for r in kept]):>8.3f}"
              f"{np.mean([r['f1_memd'] for r in kept]):>8.3f}"
              f"{np.mean([r['f1_pinn'] for r in kept]):>8.3f}"
              "   <- records that PASS the quality gate")
    print("=" * 84)
    print(f"quality gate: {len(kept)}/{len(rows)} records usable "
          f"({', '.join(r['record_id'] for r in kept) if kept else 'none'}); "
          "GATED records are excluded downstream, not averaged in.")
    good = [r for r in rows if r["f1_pinn"] >= 0.8]
    print(f"records with PINN F1 >= 0.80: {len(good)}/{len(rows)}"
          + (f"  ({', '.join(r['record_id'] for r in good)})" if good else ""))
    print("=" * 84 + "\n")

    out = Path("reports") / f"accuracy_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(
            {"dataset": args.dataset, "role": role, "readout": cfg.pinn.readout,
             "train_dataset": TRAIN_DATASET, "tolerance_ms": args.tolerance_ms,
             "records": rows, "mean": mean,
             "timestamp": datetime.now(timezone.utc).isoformat()},
            fh, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
