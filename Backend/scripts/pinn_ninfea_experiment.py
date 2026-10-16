#!/usr/bin/env python3
"""scripts/pinn_ninfea_experiment.py

Does adding NInFEA to PINN training help? (CLAUDE.md §2 "PINN", amended 2026-10-02.)

Arms
  A    the existing ADFECGDB-only PINN (data/processed/pinn/pinn.pt, never retrained here)
  B_k  PINN trained on ADFECGDB + NInFEA fold (1-k), k in {0, 1}. NInFEA mothers are split
       into two folds (`dataset.subject_of_record`, seeded shuffle) so every NInFEA
       recording is evaluated by a PINN that never saw its mother. ALL windows of the
       training-fold recordings are used -- no selection by quality, which would be
       circular since quality is measured on PINN output.

Metrics (identical pipeline for every arm)
  * NInFEA held-out mothers: quality-gate pass rate + median heart rate, through the same
    maternal cancellation + `assess_record` as scripts/04. NInFEA's Doppler beat timing is
    NOT on disk, so QRS F1 cannot be scored there; the gate checks that the rhythm is
    fetal-plausible (110-160 bpm, few RR artefacts), not that it is correct -- a proxy.
  * CinC 2013 (zero-shot, never trained on): QRS F1 against `.fqrs`, via eval_zeroshot.
  * ADFECGDB (training cohort for A and B, in-sample): QRS F1, to catch degradation.
B's CinC / ADFECGDB numbers are the mean over its two fold models.

Requires scripts/01+02 for adfecgdb, ninfea and cinc2013. Usage:
    python scripts/pinn_ninfea_experiment.py [--epochs N] [--skip-train]

Implements: CLAUDE.md §2 "PINN" (amended experiment).
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
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import eval_zeroshot  # noqa: E402  (sibling script reused for QRS F1 scoring)

from fcac.config import Config, load_config  # noqa: E402
from fcac.mecg import adaptive_cancel_windows  # noqa: E402
from fcac.pinn.infer import extract_fecg  # noqa: E402
from fcac.pinn.model import PINN  # noqa: E402
from fcac.pinn.train import concat_fetal_band_windows, train_pinn_on_windows  # noqa: E402
from fcac.quality import assess_record  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/processed/pinn_ninfea_exp")
BASELINE_CKPT = Path("data/processed/pinn/pinn.pt")


def load_band(dataset: str) -> dict[str, np.ndarray]:
    in_dir = Path("data/processed/fetal_band") / dataset
    paths = sorted(in_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No fetal-band windows in {in_dir}; run scripts 01/02 first")
    return {p.stem: np.load(p, allow_pickle=True)["x"] for p in paths}


def split_ninfea_by_mother(cfg_ninfea: Config, record_ids: list[str], n_folds: int = 2) -> list[list[str]]:
    """Deterministic seeded split of recordings into `n_folds`, keeping each mother whole."""
    subject_of = {r: cfg_ninfea.dataset.subject_of_record[r] for r in record_ids}
    subjects = sorted(set(subject_of.values()))
    order = np.random.default_rng(cfg_ninfea.seed).permutation(len(subjects))
    fold_of_subject = {subjects[i]: k % n_folds for k, i in enumerate(order)}
    folds = [[] for _ in range(n_folds)]
    for r in record_ids:
        folds[fold_of_subject[subject_of[r]]].append(r)
    return folds


def load_model(cfg: Config, ckpt: Path) -> PINN:
    model = PINN(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def gate_stats(model: PINN, cfg_ninfea: Config, band: dict[str, np.ndarray], records: list[str]) -> dict:
    """Per-record quality-gate pass rate and median HR, mirroring scripts/04."""
    per_record = {}
    for r in records:
        windows, cancelled = adaptive_cancel_windows(band[r], 1000.0, cfg_ninfea)
        q = assess_record(extract_fecg(model, windows, cfg_ninfea).fecg, 1000.0, cfg_ninfea)
        per_record[r] = {
            "n_windows": q["n_windows"], "n_acceptable": q["n_acceptable"],
            "acceptable_fraction": q["acceptable_fraction"], "usable": bool(q["usable"]),
            "median_hr_bpm": q["median_heart_rate_bpm"],
        }
    n = sum(v["n_windows"] for v in per_record.values())
    k = sum(v["n_acceptable"] for v in per_record.values())
    return {"per_record": per_record, "n_windows": n, "n_acceptable": k,
            "window_pass_rate": k / n if n else float("nan"),
            "n_records": len(per_record), "n_usable_records": sum(v["usable"] for v in per_record.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="override pinn.epochs for B arms")
    parser.add_argument("--skip-train", action="store_true", help="reuse existing B checkpoints")
    parser.add_argument("--train-only", type=int, default=None, metavar="K",
                        help="train just fold K's B model and exit (run the folds in parallel, "
                             "then rerun with --skip-train to evaluate)")
    args = parser.parse_args()

    cfg = load_config(dataset="adfecgdb")
    cfg_n = load_config(dataset="ninfea")
    cfg_train = cfg
    if args.epochs is not None:
        data = cfg.to_dict()
        data["pinn"]["epochs"] = args.epochs
        cfg_train = Config(data)

    ad_band = load_band("adfecgdb")
    nin_band = load_band("ninfea")
    folds = split_ninfea_by_mother(cfg_n, sorted(nin_band))
    logger.info("NInFEA folds by mother: %s", {k: len(v) for k, v in enumerate(folds)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = {}
    for k in range(len(folds)):
        ckpt = OUT_DIR / f"pinn_B{k}.pt"
        ckpts[k] = ckpt
        if (args.skip_train and ckpt.exists()) or (args.train_only is not None and k != args.train_only):
            continue
        train_records = [r for j, f in enumerate(folds) if j != k for r in f]
        windows = concat_fetal_band_windows(
            list(ad_band.values()) + [nin_band[r] for r in train_records]
        )
        logger.info("Training B%d on %d windows (ADFECGDB %d + NInFEA %d recordings)",
                    k, windows.shape[0], sum(len(v) for v in ad_band.values()), len(train_records))
        model = train_pinn_on_windows(cfg_train, windows)
        torch.save(model.state_dict(), ckpt)
        with open(OUT_DIR / f"pinn_B{k}.config.yaml", "w") as fh:
            yaml.safe_dump({"cfg": cfg_train.to_dict(), "ninfea_train_records": train_records}, fh)

    if args.train_only is not None:
        return

    arm_models = {"A": [(None, load_model(cfg, BASELINE_CKPT))],
                  "B": [(k, load_model(cfg, ckpts[k])) for k in ckpts]}

    # --- NInFEA held-out mothers: gate pass rate on the same records for A and B ---
    nin = {"A": {"per_record": {}}, "B": {"per_record": {}}}
    for arm, models in arm_models.items():
        for k, model in models:
            held_out = folds[k] if k is not None else sorted(nin_band)
            for r, v in gate_stats(model, cfg_n, nin_band, held_out)["per_record"].items():
                nin[arm]["per_record"][r] = v
    # A scored on every record; restrict to the records B scored held-out (all of them).
    for arm in nin:
        pr = nin[arm]["per_record"]
        n = sum(v["n_windows"] for v in pr.values())
        k_ok = sum(v["n_acceptable"] for v in pr.values())
        nin[arm].update(window_pass_rate=k_ok / n, n_windows=n, n_acceptable=k_ok,
                        n_usable_records=sum(v["usable"] for v in pr.values()), n_records=len(pr))
    better = sum(nin["B"]["per_record"][r]["n_acceptable"] > nin["A"]["per_record"][r]["n_acceptable"]
                 for r in nin["A"]["per_record"])
    worse = sum(nin["B"]["per_record"][r]["n_acceptable"] < nin["A"]["per_record"][r]["n_acceptable"]
                for r in nin["A"]["per_record"])

    # --- QRS F1 on CinC 2013 (zero-shot) and ADFECGDB (in-sample) ---
    f1 = {}
    for dataset in ("cinc2013", "adfecgdb"):
        f1[dataset] = {}
        for arm, models in arm_models.items():
            results = [eval_zeroshot.evaluate_dataset(dataset, m, 50.0) for _, m in models]
            ok = [x for x in results if x.get("status") == "ok"]
            f1[dataset][arm] = (
                {"mean_f1_pinn": float(np.mean([x["mean_f1_pinn"] for x in ok])),
                 "mean_f1_memd_band": float(np.mean([x["mean_f1_memd_band"] for x in ok])),
                 "n_windows_scored": ok[0]["n_windows_scored"]} if ok else {"status": "missing"}
            )

    summary = {
        "ninfea_heldout_gate": {"A": {k: v for k, v in nin["A"].items() if k != "per_record"},
                                "B": {k: v for k, v in nin["B"].items() if k != "per_record"},
                                "records_B_better": int(better), "records_B_worse": int(worse),
                                "per_record": {r: {"A": nin["A"]["per_record"][r], "B": nin["B"]["per_record"][r]}
                                               for r in nin["A"]["per_record"]}},
        "qrs_f1": f1,
        "folds": {str(k): v for k, v in enumerate(folds)},
        "epochs_B": cfg_train.pinn.epochs,
        "note": "NInFEA metric is a quality-gate proxy (no Doppler timing on disk); CinC 2013 and "
                "ADFECGDB are QRS F1. A is ADFECGDB-only; B is the mean of two NInFEA-pooled folds.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/pinn_ninfea_experiment.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    g = summary["ninfea_heldout_gate"]
    print("\nNInFEA held-out mothers (gate proxy):")
    for arm in ("A", "B"):
        print(f"  {arm}: window pass rate {g[arm]['window_pass_rate']:.3f} "
              f"({g[arm]['n_acceptable']}/{g[arm]['n_windows']}), usable records "
              f"{g[arm]['n_usable_records']}/{g[arm]['n_records']}")
    print(f"  records where B passes more windows: {better}, fewer: {worse}")
    print("QRS F1 (PINN output):")
    for d in f1:
        print(f"  {d}: A {f1[d]['A'].get('mean_f1_pinn', float('nan')):.3f}  "
              f"B {f1[d]['B'].get('mean_f1_pinn', float('nan')):.3f}  "
              f"(MEMD band {f1[d]['A'].get('mean_f1_memd_band', float('nan')):.3f})")


if __name__ == "__main__":
    main()
