#!/usr/bin/env python3
"""scripts/01_preprocess.py

Runs src/fcac/preprocessing/pipeline.py::preprocess() over every record of one dataset
and caches the result to data/interim/<dataset>/<record_id>.npz.

TODO(bhuvanesh): NInFEA record ids below are a 1..n_recordings placeholder -- confirm
the actual PhysioNet NInFEA file-naming convention before relying on this.

Usage: python scripts/01_preprocess.py --dataset {adfecgdb,ninfea}

Implements: CLAUDE.md §4 scripts/01_preprocess.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fcac.config import load_config
from fcac.io.adfecgdb import load_adfecgdb_record
from fcac.io.cinc2013 import load_cinc2013_record
from fcac.io.ninfea import load_ninfea_record
from fcac.preprocessing.pipeline import preprocess
from fcac.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LOADERS = {
    "adfecgdb": load_adfecgdb_record,
    "ninfea": load_ninfea_record,
    "cinc2013": load_cinc2013_record,
}


def run(dataset: str) -> dict:
    cfg = load_config(dataset=dataset)
    set_seed(cfg.seed)
    loader = _LOADERS[dataset]

    if "record_ids" in cfg.dataset:
        record_ids = [str(r) for r in cfg.dataset.record_ids]
    else:
        # NInFEA records are numbered 1..60 and not listed individually in its config.
        record_ids = [str(i + 1) for i in range(cfg.dataset.n_recordings)]

    interim_dir = Path(cfg.dataset.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    for record_id in record_ids:
        logger.info("Preprocessing %s/%s", dataset, record_id)
        recording = loader(record_id, cfg)
        result = preprocess(recording, cfg)
        out_path = interim_dir / f"{record_id}.npz"
        np.savez_compressed(
            out_path,
            windows=result.windows,
            fs=result.fs,
            ref_windows=(
                result.ref_windows if result.ref_windows is not None else np.array([])
            ),
            ga_days=np.nan if result.ga_days is None else result.ga_days,
            subject_id=result.subject_id,
        )
        processed.append({"record_id": record_id, "n_windows": int(result.windows.shape[0])})

    return {"dataset": dataset, "records": processed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"], required=True)
    args = parser.parse_args()

    summary = run(args.dataset)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"01_preprocess_{args.dataset}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote reports/01_preprocess_%s.json", args.dataset)


if __name__ == "__main__":
    main()
