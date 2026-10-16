#!/usr/bin/env python3
"""scripts/eval_zeroshot.py

Zero-shot transfer evaluation for the extraction chain: the PINN is trained on ONE
cohort (ADFECGDB) and scored on the OTHERS without any fine-tuning
(CLAUDE.md section 2 "PINN" -- LOCKED).

For each cohort it detects R-peaks in the PINN's isolated fECG, matches them against
that cohort's reference fetal beat annotations within +/-50 ms, and reports QRS F1 /
precision / recall plus heart-rate agreement. Two baselines are reported alongside, so
the numbers say what the PINN actually added rather than just how easy the cohort is:

  * best raw abdominal channel (what you get with no extraction at all)
  * best MEMD fetal-band channel (stage 02, before the PINN)

Both baselines are best-of-4-channels, not channel 0: on ADFECGDB r01 the per-channel
MEMD F1 is 0.219 / 0.920 / 0.913 / 0.955, so quoting ch0 would flatter the PINN.

Reference availability differs per cohort, and the script reports only what each one can
actually support:

  cohort     role         reference            scorable
  ---------  -----------  -------------------  --------------------------------
  adfecgdb   TRAIN        .edf.qrs             yes -- but IN-SAMPLE, not validation
  cinc2013   zero-shot    .fqrs                yes -- true held-out validation
  ninfea     zero-shot    Doppler beat timing  NO -- annotations not on disk

Usage:
    python scripts/eval_zeroshot.py
    python scripts/eval_zeroshot.py --datasets cinc2013 --tolerance-ms 50
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from fcac.config import load_config
from fcac.evaluation.extraction import qrs_f1
from fcac.memd.diagnostics import pan_tompkins_peaks
from fcac.pinn.infer import extract_fecg
from fcac.pinn.model import PINN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_DATASET = "adfecgdb"  # LOCKED (CLAUDE.md section 2)
DEFAULT_DATASETS = ("adfecgdb", "cinc2013", "ninfea")


def reference_peaks(dataset: str, record_id: str, cfg) -> Optional[np.ndarray]:
    """Fetal R-peak sample indices in the RAW recording's time base, or None.

    None means this cohort has no usable reference on disk -- extraction still runs, but
    nothing can be scored against it.
    """
    if dataset == "adfecgdb":
        from fcac.io.adfecgdb import load_adfecgdb_qrs_annotations

        return load_adfecgdb_qrs_annotations(record_id, cfg)
    if dataset == "cinc2013":
        from fcac.io.cinc2013 import load_cinc2013_fqrs

        return load_cinc2013_fqrs(record_id, cfg)
    # NInFEA's Doppler beat timing is not in the WFDB download (see module docstring).
    return None


def peaks_for_window(
    peaks_raw: np.ndarray, window_idx: int, cfg, fs: float
) -> np.ndarray:
    """Map whole-recording reference peaks into one window's local sample indices.

    Preprocessing trims `edge_trim_seconds` from the START of the recording and then cuts
    non-overlapping windows, so window w begins at trim + w*stride in the raw time base.
    This assumes the cohort's native rate equals the target rate (true for ADFECGDB and
    CinC 2013, both 1000 Hz); a resampled cohort would need the ratio folded in here.
    """
    trim = int(cfg.preprocessing.edge_trim_seconds * fs)
    stride = cfg.preprocessing.window.stride_samples
    size = cfg.preprocessing.window.size_samples

    start = trim + window_idx * stride
    local = peaks_raw - start
    return local[(local >= 0) & (local < size)]


def score_signal(signal: np.ndarray, ref_local: np.ndarray, fs: float, tolerance_ms: float):
    """Detect R-peaks in one 1-D signal and score them against that window's reference."""
    detected = pan_tompkins_peaks(signal, fs)
    f1, precision, recall = qrs_f1(detected, ref_local, fs, tolerance_ms=tolerance_ms)
    return f1, precision, recall, len(detected)


def score_best_channel(
    multichannel: np.ndarray, ref_local: np.ndarray, fs: float, tolerance_ms: float
):
    """Best F1 over a (C, N) window's channels.

    Scoring channel 0 alone badly understates these baselines: on ADFECGDB r01 the
    per-channel MEMD F1 runs 0.219 / 0.920 / 0.913 / 0.955, so ch0 is the WORST lead and
    reporting it would make the PINN look better than it is. The honest baseline for a
    method that gets to see all four channels is the best single channel.
    """
    return max(
        score_signal(multichannel[c], ref_local, fs, tolerance_ms)[0]
        for c in range(multichannel.shape[0])
    )


def evaluate_dataset(dataset: str, model: PINN, tolerance_ms: float) -> dict:
    cfg = load_config(dataset=dataset)

    band_dir = Path("data/processed/fetal_band") / dataset
    interim_dir = Path(cfg.dataset.interim_dir)
    paths = sorted(band_dir.glob("*.npz"))
    if not paths:
        logger.warning("%s: no fetal-band windows, skipping (run scripts 01/02)", dataset)
        return {"dataset": dataset, "status": "missing_fetal_band"}

    rows = []
    for path in paths:
        record_id = path.stem
        band = np.load(path, allow_pickle=True)
        windows = band["x"]
        fs = float(band["fs"])

        fecg = extract_fecg(model, windows, cfg).fecg
        raw = np.load(interim_dir / f"{record_id}.npz", allow_pickle=True)["windows"]

        peaks_raw = reference_peaks(dataset, record_id, cfg)
        if peaks_raw is None:
            rows.append({"record_id": record_id, "n_windows": int(fecg.shape[0])})
            continue

        for w in range(fecg.shape[0]):
            ref_local = peaks_for_window(peaks_raw, w, cfg, fs)
            if ref_local.size < 3:  # too few beats in this window to score meaningfully
                continue
            pinn = score_signal(fecg[w], ref_local, fs, tolerance_ms)
            band_best = score_best_channel(windows[w], ref_local, fs, tolerance_ms)
            raw_best = score_best_channel(raw[w], ref_local, fs, tolerance_ms)
            rows.append(
                {
                    "record_id": record_id,
                    "window": w,
                    "n_ref": int(ref_local.size),
                    "f1_pinn": pinn[0],
                    "precision_pinn": pinn[1],
                    "recall_pinn": pinn[2],
                    "f1_memd_band": band_best,
                    "f1_raw": raw_best,
                }
            )

    scored = [r for r in rows if "f1_pinn" in r]
    result = {
        "dataset": dataset,
        "role": "train (IN-SAMPLE)" if dataset == TRAIN_DATASET else "zero-shot",
        "n_records": len(paths),
        "n_windows_scored": len(scored),
    }
    if not scored:
        result["status"] = "no_reference_annotations"
        return result

    for key in ("f1_raw", "f1_memd_band", "f1_pinn", "precision_pinn", "recall_pinn"):
        values = np.array([r[key] for r in scored], dtype=float)
        result[f"mean_{key}"] = float(values.mean())
        result[f"std_{key}"] = float(values.std())
    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    args = parser.parse_args()

    cfg = load_config(dataset=TRAIN_DATASET)
    model = PINN(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)  # LOCKED: never fine-tuned

    results = [evaluate_dataset(d, model, args.tolerance_ms) for d in args.datasets]

    print("\n" + "=" * 78)
    print(f"ZERO-SHOT TRANSFER -- PINN trained on {TRAIN_DATASET}, never fine-tuned")
    print(f"QRS F1 within +/-{args.tolerance_ms:.0f} ms")
    print("=" * 78)
    print(f"{'cohort':<10} {'role':<18} {'windows':>8} {'raw':>7} {'MEMD':>7} {'PINN':>7}")
    print("-" * 78)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['dataset']:<10} {r.get('role',''):<18} {'-':>8}   {r.get('status')}")
            continue
        print(
            f"{r['dataset']:<10} {r['role']:<18} {r['n_windows_scored']:>8} "
            f"{r['mean_f1_raw']:>7.3f} {r['mean_f1_memd_band']:>7.3f} {r['mean_f1_pinn']:>7.3f}"
        )
    print("=" * 78)
    print("'raw' and 'MEMD' are BEST-OF-4-CHANNELS baselines; 'PINN' is the single")
    print("extracted fetal ECG. To be worth its place the PINN must beat the MEMD")
    print("column, since that is already available without it. adfecgdb is the training")
    print("cohort -- in-sample, not validation.")
    print("Only cohorts with reference annotations on disk can be scored.")
    print("=" * 78 + "\n")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_dataset": TRAIN_DATASET,
        "checkpoint": args.checkpoint,
        "tolerance_ms": args.tolerance_ms,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(reports_dir / "eval_zeroshot.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote reports/eval_zeroshot.json")


if __name__ == "__main__":
    main()
