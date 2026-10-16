#!/usr/bin/env python3
"""scripts/06_train_cvae.py

Trains the Contrastive VAE (src/fcac/cvae/) on real 12-feature window vectors from
scripts/05_features.py, pooled across every dataset in `pooled_training.datasets` (ADFECGDB + the NInFEA
recordings that passed quality gating; amended CLAUDE.md §2 Datasets, 2026-10-02). The PINN
is still ADFECGDB-only. Usage: python scripts/06_train_cvae.py [--experiment ...]

Real per-window features are sparsely populated (see scripts/05_features.py's report --
prsa/mse/sampen/dfa_alpha are frequently NaN because a 5 s window holds far fewer beats
than those measures need; lf_hf_ratio is ALWAYS NaN at this window size, structurally,
not as a data-quality accident -- it needs >=25 s to resolve its lowest frequency).
Rather than block training on that, each feature column's NaNs are imputed with that
column's own median over the real observed values in this run (a column with zero real
values, i.e. lf_hf_ratio today, is imputed with 0.0 -- a documented placeholder, not a
measurement). This is a data-hygiene choice, not a feature-formula invention, and is
reported in full in reports/06_train_cvae.json so it's never silently mistaken for
measured data.

GA labels are currently null for every ADFECGDB record (see plan.md item 1), so the
CVAE's "same_subject_or_ga_within_window" positive rule (CLAUDE.md §2 "CVAE") collapses
to same-subject-only here: NaN GA comparisons are never True, so build_positive_mask
falls back gracefully rather than raising.

Usage: python scripts/06_train_cvae.py

Implements: CLAUDE.md §4 scripts/06_train_cvae.py.
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
from fcac.cvae.train import train_cvae
from fcac.transformer.latents import impute_missing  # noqa: F401  (also used by visualize_cvae_latents.py)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_real_features(cfgs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every gated record's real per-window features across `cfgs` (dataset
    name -> Config) into one (N, 12) matrix, with matching (N,) namespaced subject_id and
    ga_days arrays (one row repeated per window). GA comes from each dataset's config
    (NaN when null), not the stale value baked into the features .npz. NInFEA windows are
    grouped by mother via `dataset.subject_of_record`."""
    x_parts, subject_parts, ga_parts = [], [], []
    for dataset, cfg in cfgs.items():
        in_dir = Path("data/processed/features") / dataset
        paths = sorted(in_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No features in {in_dir}. Run scripts/05_features.py --dataset {dataset} first."
            )
        for path in paths:
            record_id = path.stem
            x = np.load(path, allow_pickle=True)["x"]  # (W, 12)
            ga = cfg.dataset.ga_days[record_id] if record_id in cfg.dataset.ga_days else None
            if "subject_of_record" in cfg.dataset:
                subject = f"{dataset}:S{cfg.dataset.subject_of_record[record_id]}"
            else:
                subject = f"{dataset}:{record_id}"
            x_parts.append(x)
            subject_parts.append(np.full(x.shape[0], subject))
            ga_parts.append(np.full(x.shape[0], np.nan if ga is None else float(ga)))

    return np.concatenate(x_parts), np.concatenate(subject_parts), np.concatenate(ga_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg = load_config(dataset="adfecgdb", experiment=args.experiment)
    synthetic = bool(cfg["synthetic_ga"]) if "synthetic_ga" in cfg else False
    if synthetic:
        logger.warning("SYNTHETIC ADFECGDB GA in use -- pipeline check only, do NOT report numbers.")
    datasets = list(cfg.pooled_training.datasets)
    cfgs = {d: (cfg if d == "adfecgdb" else load_config(dataset=d)) for d in datasets}

    features, subject_ids, ga_days = load_real_features(cfgs)
    logger.info(
        "Loaded %d real windows from %d subjects across %s",
        len(features), len(set(subject_ids)), datasets,
    )

    imputed, impute_report = impute_missing(features)
    for name, r in impute_report.items():
        logger.info(
            "  %-18s real=%3d imputed=%3d (fill=%.4f)",
            name, r["n_real"], r["n_imputed"], r["fill_value"],
        )

    features_t = torch.from_numpy(imputed.astype(np.float32))
    model = train_cvae(cfg, features_t, subject_ids, ga_days)

    out_dir = Path("data/processed/cvae")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "cvae.pt")
    logger.info("Saved checkpoint to %s", out_dir / "cvae.pt")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "datasets": datasets,
        "synthetic_ga": synthetic,
        "experiment": args.experiment,
        "n_windows": int(len(features)),
        "n_subjects": len(set(subject_ids)),
        "imputation": impute_report,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(reports_dir / "06_train_cvae.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", reports_dir / "06_train_cvae.json")


if __name__ == "__main__":
    main()
