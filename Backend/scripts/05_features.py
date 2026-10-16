#!/usr/bin/env python3
"""scripts/05_features.py

Extracts the 12-feature vector (src/fcac/features/extract.py) for every quality-passing
fECG window from scripts/04_extract_fecg.py's output.

Only windows that passed src/fcac/quality.py's gate are used -- a maternal-locked
window would otherwise contribute 12 confident-looking numbers computed from the
mother's heartbeat, not the baby's (see quality.py's module docstring).

Usage: python scripts/05_features.py --dataset {adfecgdb,ninfea,cinc2013}

Implements: CLAUDE.md §4 scripts/05_features.py; CLAUDE.md §5 PINNOutput -> FeatureVector.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fcac.config import load_config
from fcac.features.extract import FEATURE_NAMES, extract_features_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"], required=True)
    args = parser.parse_args()

    cfg = load_config(dataset=args.dataset)

    in_dir = Path("data/processed/fecg") / args.dataset
    paths = sorted(in_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No extracted fECG in {in_dir}. Run scripts/04_extract_fecg.py "
            f"--dataset {args.dataset} first."
        )

    out_dir = Path("data/processed/features") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        fecg = data["fecg"]
        mask = data["quality_mask"]
        kept = fecg[mask]

        record_id = path.stem
        if kept.shape[0] == 0:
            logger.warning("%s: 0/%d windows pass quality, skipping", record_id, fecg.shape[0])
            continue

        features = extract_features_batch(kept, float(data["fs"]), cfg)  # (W_kept, 12)
        nan_rate = np.isnan(features).mean(axis=0)

        np.savez_compressed(
            out_dir / f"{record_id}.npz",
            x=features,
            ga_days=data["ga_days"],
            subject_id=str(data["subject_id"]),
        )
        records.append(
            {
                "record_id": record_id,
                "n_windows": int(kept.shape[0]),
                "nan_rate_by_feature": {
                    name: float(rate) for name, rate in zip(FEATURE_NAMES, nan_rate)
                },
            }
        )
        logger.info(
            "%s: %d windows -> features, mean NaN rate %.0f%%",
            record_id,
            kept.shape[0],
            float(nan_rate.mean()) * 100,
        )

    if not records:
        logger.error("No record had any quality-passing window; nothing written.")
        raise SystemExit(1)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "feature_names": FEATURE_NAMES,
        "records": records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(reports_dir / f"05_features_{args.dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", reports_dir / f"05_features_{args.dataset}.json")


if __name__ == "__main__":
    main()
