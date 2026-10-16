#!/usr/bin/env python3
"""scripts/02_memd.py

Runs MEMD decomposition + fetal-band IMF selection over the cached preprocessed windows
(data/interim/<dataset>/*.npz), caching raw IMFs to data/memd/ and writing the
fetal-band windows the PINN consumes to data/processed/fetal_band/<dataset>/.

MEMD costs roughly 1.5-4 s per (4, 5000) window, so a 5-minute ADFECGDB record is a few
minutes and the whole dataset is tens of minutes on first run. Re-runs hit the
data/memd/ cache (src/fcac/memd/decompose.py) and are fast.

What comes out of this stage still contains BOTH maternal and fetal complexes -- MEMD
only removes motion/baseline-drift/muscle artefact. Maternal-fetal separation is the
PINN's job (CLAUDE.md section 2 "MEMD role").

Requires scripts/01_preprocess.py to have run first.

Usage: python scripts/02_memd.py --dataset {adfecgdb,ninfea} [--limit-windows N]

Implements: CLAUDE.md section 4 scripts/02_memd.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fcac.config import load_config
from fcac.memd.decompose import decompose_recording
from fcac.memd.select import build_fetal_band_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(dataset: str, limit_windows: int | None = None) -> dict:
    cfg = load_config(dataset=dataset)
    interim_dir = Path(cfg.dataset.interim_dir)
    out_dir = Path("data/processed/fetal_band") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(interim_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(
            f"No preprocessed windows in {interim_dir}. Run "
            f"scripts/01_preprocess.py --dataset {dataset} first."
        )

    records = []
    for npz_path in npz_paths:
        record_id = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        windows = data["windows"]
        fs = float(data["fs"])
        if limit_windows is not None:
            windows = windows[:limit_windows]

        logger.info("MEMD %s/%s: %d windows", dataset, record_id, windows.shape[0])
        imfs_per_window = decompose_recording(windows, dataset, record_id, cfg)

        # ADFECGDB's filtered scalp reference scores the coherence term. Scoring only --
        # never a model input (CLAUDE.md section 6).
        ref_windows = data["ref_windows"]
        references = None
        if ref_windows.size > 0:
            references = [ref_windows[w] for w in range(windows.shape[0])]

        x, scores, selected = build_fetal_band_windows(
            imfs_per_window, fs, cfg, references=references
        )

        np.savez_compressed(
            out_dir / f"{record_id}.npz",
            x=x,
            fs=fs,
            scores=scores,
            selected=np.array(selected, dtype=object),
            ga_days=data["ga_days"],
            subject_id=str(data["subject_id"]),
        )

        k_per_window = [int(i.shape[0]) for i in imfs_per_window]
        records.append(
            {
                "record_id": record_id,
                "n_windows": int(x.shape[0]),
                "k_min": min(k_per_window),
                "k_max": max(k_per_window),
                "mean_n_selected": float(np.mean([len(s) for s in selected])),
            }
        )
        logger.info(
            "  %s: K=%d-%d, mean %.1f IMFs selected per window",
            record_id,
            records[-1]["k_min"],
            records[-1]["k_max"],
            records[-1]["mean_n_selected"],
        )

    return {"dataset": dataset, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"], required=True)
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="decompose only the first N windows per record (smoke-testing)",
    )
    args = parser.parse_args()

    summary = run(args.dataset, args.limit_windows)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"02_memd_{args.dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote reports/02_memd_%s.json", args.dataset)


if __name__ == "__main__":
    main()
