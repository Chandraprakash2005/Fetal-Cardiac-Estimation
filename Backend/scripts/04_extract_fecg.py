#!/usr/bin/env python3
"""scripts/04_extract_fecg.py

Extracts the single-channel fetal ECG for a dataset's windows and writes it to
data/processed/fecg/<dataset>/<record_id>.npz. The method is `extraction.method`
(src/fcac/extraction.py, `--method` overrides): `pinn` runs maternal cancellation -> the frozen
PINN (trained by scripts/03_train_pinn.py on ADFECGDB); `hybrid` (default) keeps that output
per window when it passes the quality gate and otherwise uses a FastICA BSS extraction of the
preprocessed windows. The npz also records `from_pinn` per window.

LOCKED (CLAUDE.md section 2 "PINN"): the checkpoint is loaded read-only and the model is
never updated here -- NInFEA is zero-shot, never fine-tuned. `--dataset adfecgdb` is
also allowed, and is in-sample by construction (that is what the model trained on), so
treat those outputs as a sanity check rather than as validation.

Usage: python scripts/04_extract_fecg.py --dataset ninfea
                                         [--checkpoint data/processed/pinn/pinn.pt]

Implements: CLAUDE.md section 4 scripts/04_extract_fecg.py.
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
from fcac.extraction import extract_with_method
from fcac.pinn.model import PINN
from fcac.quality import assess_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(dataset: str, checkpoint: str, method: str | None = None) -> dict:
    cfg = load_config(dataset=dataset)
    method = method or cfg.extraction.method

    model = PINN(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    for param in model.parameters():
        # Belt and braces against an accidental optimizer step downstream -- the
        # no-fine-tuning rule is LOCKED.
        param.requires_grad_(False)

    in_dir = Path("data/processed/fetal_band") / dataset
    paths = sorted(in_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No fetal-band windows in {in_dir}. Run "
            f"scripts/02_memd.py --dataset {dataset} first."
        )

    out_dir = Path("data/processed/fecg") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        windows = data["x"]
        fs = float(data["fs"])
        logger.info("Extracting fECG for %s/%s (%d windows)", dataset, path.stem, windows.shape[0])

        # Maternal cancellation -> PINN, with the BSS fallback when `method` is hybrid
        # (src/fcac/extraction.py). Measured on ADFECGDB: cancellation lifts mean QRS F1 0.501 ->
        # 0.612 because it rescues maternal-locked windows without damaging clean ones.
        pre = np.load(Path(cfg.dataset.interim_dir) / f"{path.stem}.npz", allow_pickle=True)["windows"]
        fecg, from_pinn, cancelled = extract_with_method(windows, pre, fs, cfg, model, method)

        # Quality gate: the chain fails silently, so record which windows are trustworthy
        # rather than letting the feature stage compute biomarkers off a maternal lock.
        quality = assess_record(fecg, fs, cfg)

        np.savez_compressed(
            out_dir / f"{path.stem}.npz",
            fecg=fecg,
            from_pinn=from_pinn,
            fs=fs,
            ga_days=data["ga_days"],
            subject_id=str(data["subject_id"]),
            mecg_cancelled=cancelled,
            quality_mask=quality["acceptable_mask"],
        )
        logger.info(
            "  %s: %d/%d windows pass quality (%.0f%% cancelled, median %.0f bpm)%s",
            path.stem,
            quality["n_acceptable"],
            quality["n_windows"],
            100.0 * cancelled.mean(),
            quality["median_heart_rate_bpm"],
            "" if quality["usable"] else "  <- RECORD UNUSABLE",
        )
        records.append({
            "record_id": path.stem,
            "n_windows": int(fecg.shape[0]),
            "share_from_pinn": float(from_pinn.mean()),
            "n_acceptable": quality["n_acceptable"],
            "acceptable_fraction": quality["acceptable_fraction"],
            "usable": quality["usable"],
            "median_heart_rate_bpm": quality["median_heart_rate_bpm"],
            "mecg_cancelled_fraction": float(cancelled.mean()),
        })

    return {"dataset": dataset, "checkpoint": checkpoint, "method": method, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"], default="ninfea")
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    parser.add_argument("--method", choices=["pinn", "hybrid"], default=None, help="override extraction.method")
    args = parser.parse_args()

    summary = run(args.dataset, args.checkpoint, args.method)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"04_extract_fecg_{args.dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote reports/04_extract_fecg_%s.json", args.dataset)


if __name__ == "__main__":
    main()
