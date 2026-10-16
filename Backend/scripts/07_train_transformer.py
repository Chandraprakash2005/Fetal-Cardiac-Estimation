#!/usr/bin/env python3
"""scripts/07_train_transformer.py

Trains the Cardiac Age transformer (src/fcac/transformer/) on sequences of CVAE latents
from real pooled features -- ADFECGDB + the quality-gated NInFEA recordings, per
`pooled_training.datasets` (scripts/05_features.py -> scripts/06_train_cvae.py).

Each record is one subject (or one recording of one) with one GA (CLAUDE.md §3), so a record's window
latents form one sequence with one target. Feature NaNs are filled with the same
per-column medians scripts/06_train_cvae.py used (read from reports/06_train_cvae.json)
so the encoder sees exactly the distribution it was trained on.

GA comes from each dataset's `ga_days` config (ADFECGDB: cfg.dataset.ga_days). While real per-record GA is unavailable, run with
`--experiment synthetic_ga_smoketest`: those labels are random, the run is stamped
`synthetic_ga: true` in its summary, and no number from it may be reported. With null
GA and no experiment, this script refuses to train.

Usage: python scripts/07_train_transformer.py [--experiment synthetic_ga_smoketest]

Implements: CLAUDE.md §4 scripts/07_train_transformer.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from fcac.config import load_config
from fcac.transformer.evaluate import predict_recordings
from fcac.transformer.latents import load_cvae_and_fill_values, load_pooled_recordings
from fcac.transformer.train import train_transformer_on_recordings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/processed/transformer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg = load_config(dataset="adfecgdb", experiment=args.experiment)
    synthetic = bool(cfg["synthetic_ga"]) if "synthetic_ga" in cfg else False
    if synthetic:
        logger.warning("SYNTHETIC GA LABELS in use -- pipeline check only, do NOT report numbers.")

    cvae, fill_values = load_cvae_and_fill_values(cfg)
    datasets = list(cfg.pooled_training.datasets)
    cfgs = {d: (cfg if d == "adfecgdb" else load_config(dataset=d)) for d in datasets}
    adfecgdb_null = [r for r in cfg.dataset.record_ids if cfg.dataset.ga_days[r] is None]
    if adfecgdb_null:
        raise SystemExit(
            f"No GA label for ADFECGDB {adfecgdb_null} in configs/datasets/adfecgdb.yaml. Fill in "
            "the real LMP-dated GA, or pass --experiment synthetic_ga_smoketest for a pipeline check."
        )

    pooled = load_pooled_recordings(cfgs, cvae, fill_values)
    record_keys = [p["key"] for p in pooled]
    z_by_recording = [p["z"] for p in pooled]
    ga_days = torch.tensor([p["ga_days"] for p in pooled])
    logger.info("Training on %d pooled recordings: %s", len(pooled), dict(zip(record_keys, ga_days.tolist())))

    model = train_transformer_on_recordings(cfg, z_by_recording, ga_days)
    train_pred = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "transformer.pt")
    with open(OUT_DIR / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh)

    reports_dir = Path("reports")
    summary = {
        "synthetic_ga": synthetic,
        "experiment": args.experiment,
        "datasets": datasets,
        "record_ids": record_keys,
        "n_windows": {p["key"]: int(len(p["z"])) for p in pooled},
        "ga_days": dict(zip(record_keys, ga_days.tolist())),
        "train_predicted_ca_days": dict(zip(record_keys, train_pred.tolist())),
        "note": "train-set fit only, not an evaluation (see scripts/08_evaluate.py)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(reports_dir / "07_train_transformer.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", reports_dir / "07_train_transformer.json")


if __name__ == "__main__":
    main()
