#!/usr/bin/env python3
"""scripts/08_evaluate.py

Evaluates the Cardiac Age transformer by POOLED leave-one-subject-out CV (CLAUDE.md §2
"Evaluation", amended 2026-10-02): every gated recording in `pooled_training.datasets`
(ADFECGDB + quality-gated NInFEA) is held out once, grouped by mother; the transformer
is re-trained on the rest per fold. By default the CVAE (and its imputation medians) is
refit inside every fold on the training subjects only, so the held-out recording shapes
nothing (fcac.transformer.evaluate.losocv_refit_cvae). `--shared-cvae` instead reuses the
single pooled CVAE from scripts/06 for every fold (faster, but leaky -- see the
LIMITATION in losocv_adfecgdb); use it only to measure how much the leakage matters.

Reports MAE in days + 95% bootstrap CI, RMSE, Pearson r and Delta-Age -- overall and per
cohort -- after the mandatory MAE unit check. NInFEA is no longer zero-shot here.

If ADFECGDB GA labels are synthetic (--experiment synthetic_ga_smoketest, or reports/07
or 06 say so) the summary is stamped `synthetic_ga: true` and NO number may be reported.

Usage: python scripts/08_evaluate.py [--experiment synthetic_ga_smoketest] [--shared-cvae] [--workers N]

Implements: CLAUDE.md §4 scripts/08_evaluate.py.
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
from fcac.evaluation.ageing import bootstrap_ci, mae, pearson_r, rmse
from fcac.evaluation.delta_age import delta_age_days
from fcac.evaluation.verify_units import verify_mae_units
from fcac.transformer.evaluate import losocv_adfecgdb, losocv_refit_cvae
from fcac.transformer.latents import (
    load_cvae_and_fill_values,
    load_pooled_recordings,
    load_recording_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYNTHETIC_REPORTS = [Path("reports/06_train_cvae.json"), Path("reports/07_train_transformer.json")]


def metric_block(cfg, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/RMSE with bootstrap CI (days) and Pearson r, after the unit check."""
    verify_mae_units(y_true, y_pred)
    mae_p, mae_lo, mae_hi = bootstrap_ci(y_true, y_pred, mae, cfg)
    rmse_p, rmse_lo, rmse_hi = bootstrap_ci(y_true, y_pred, rmse, cfg)
    r = pearson_r(y_true, y_pred) if len(y_true) > 2 and np.std(y_pred) > 0 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae_days": {"point": mae_p, "ci_low": mae_lo, "ci_high": mae_hi},
        "rmse_days": {"point": rmse_p, "ci_low": rmse_lo, "ci_high": rmse_hi},
        "pearson_r": r,
        "delta_age_days_mean": float(np.mean(delta_age_days(y_pred, y_true))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--shared-cvae", action="store_true", help="reuse one pooled CVAE (leaky)")
    parser.add_argument("--workers", type=int, default=1,
                        help="run LOSOCV folds in this many parallel processes (strict path only)")
    args = parser.parse_args()

    cfg = load_config(dataset="adfecgdb", experiment=args.experiment)
    datasets = list(cfg.pooled_training.datasets)
    cfgs = {d: (cfg if d == "adfecgdb" else load_config(dataset=d)) for d in datasets}

    synthetic_ga = bool(cfg["synthetic_ga"]) if "synthetic_ga" in cfg else False
    for report in SYNTHETIC_REPORTS:
        if report.exists():
            synthetic_ga = synthetic_ga or bool(json.loads(report.read_text()).get("synthetic_ga"))
    if synthetic_ga:
        logger.warning("SYNTHETIC ADFECGDB GA in use -- these numbers must NOT be reported.")

    adfecgdb_null = [r for r in cfg.dataset.record_ids if cfg.dataset.ga_days[r] is None]
    if adfecgdb_null:
        raise SystemExit(
            f"No ADFECGDB GA for {adfecgdb_null}. Fill configs/datasets/adfecgdb.yaml, or pass "
            "--experiment synthetic_ga_smoketest for a pipeline check."
        )

    if args.shared_cvae:
        cvae, fill_values = load_cvae_and_fill_values(cfg)
        pooled = load_pooled_recordings(cfgs, cvae, fill_values)
        subjects = [p["subject"] for p in pooled]
        ga = torch.tensor([p["ga_days"] for p in pooled])
        folds = losocv_adfecgdb(cfg, [p["z"] for p in pooled], ga, subjects)
    else:
        pooled = []
        for dataset, dcfg in cfgs.items():
            ids = list(dcfg.dataset.record_ids) if "record_ids" in dcfg.dataset else None
            for record_id, x in load_recording_features(dataset, ids).items():
                ga_value = dcfg.dataset.ga_days[record_id] if record_id in dcfg.dataset.ga_days else None
                if ga_value is None:
                    logger.warning("%s:%s has no GA label, skipping", dataset, record_id)
                    continue
                subject = (
                    f"{dataset}:S{dcfg.dataset.subject_of_record[record_id]}"
                    if "subject_of_record" in dcfg.dataset
                    else f"{dataset}:{record_id}"
                )
                pooled.append({"key": f"{dataset}:{record_id}", "dataset": dataset, "subject": subject,
                               "ga_days": float(ga_value), "x": x})
        folds = losocv_refit_cvae(cfg, pooled, workers=args.workers)

    rows = []
    for fold in folds:
        for r in fold["records"]:
            rec = pooled[r["index"]]
            rows.append({
                "record": rec["key"], "subject": rec["subject"], "dataset": rec["dataset"],
                "n_windows": int(len(rec["z"] if "z" in rec else rec["x"])),
                "ga_days": r["ga_days"], "ca_days": r["predicted_ca_days"],
                "delta_age_days": r["predicted_ca_days"] - r["ga_days"],
            })

    def block(selected):
        y = np.array([r["ga_days"] for r in selected])
        p = np.array([r["ca_days"] for r in selected])
        return metric_block(cfg, y, p)

    summary = {
        "synthetic_ga": synthetic_ga,
        "experiment": args.experiment,
        "datasets": datasets,
        "cvae_refit_per_fold": not args.shared_cvae,
        "folds": rows,
        "overall": block(rows),
        "per_dataset": {d: block([r for r in rows if r["dataset"] == d]) for d in datasets},
        "note": (
            "Pooled LOSOCV (all of a mother's recordings held out together) across ADFECGDB (38-41 wk) and gated NInFEA (22-27 wk): NInFEA is no "
            "longer zero-shot. CVAE refit per fold unless cvae_refit_per_fold is false."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/08_evaluate.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info(
        "Pooled LOSOCV MAE %.2f d (n=%d) %s", summary["overall"]["mae_days"]["point"],
        summary["overall"]["n"],
        {d: round(v["mae_days"]["point"], 2) for d, v in summary["per_dataset"].items()},
    )


if __name__ == "__main__":
    main()
