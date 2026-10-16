#!/usr/bin/env python3
"""scripts/03_train_pinn.py

Trains the PINN on the ADFECGDB MEMD fetal-band windows written by scripts/02_memd.py,
then freezes it. Default (CLAUDE.md section 2 "PINN", amended 2026-10-02): ADFECGDB only and
no fine-tuning -- scripts/04_extract_fecg.py loads this checkpoint read-only. The NInFEA-pooled
variant is scripts/pinn_ninfea_experiment.py and writes its own checkpoints.

The coordinate scheme this trains on is an ASSUMPTION pending supervisor sign-off; see
the header of src/fcac/pinn/coords.py and the OPEN QUESTION block in
configs/default.yaml before reporting any number from the resulting model.

Requires scripts/02_memd.py --dataset adfecgdb to have run first.

Usage: python scripts/03_train_pinn.py [--epochs N]

Implements: CLAUDE.md section 4 scripts/03_train_pinn.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from fcac.config import Config, load_config
from fcac.pinn.train import concat_fetal_band_windows, train_pinn_on_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_fetal_band(dataset: str) -> tuple[np.ndarray, list[str]]:
    """Concatenate every record's fetal-band windows into one (W, C, N) array."""
    in_dir = Path("data/processed/fetal_band") / dataset
    paths = sorted(in_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No fetal-band windows in {in_dir}. Run "
            f"scripts/02_memd.py --dataset {dataset} first."
        )
    per_record = [np.load(p, allow_pickle=True)["x"] for p in paths]
    return concat_fetal_band_windows(per_record), [p.stem for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="override pinn.epochs")
    args = parser.parse_args()

    cfg_dataset = "adfecgdb"  # LOCKED: the PINN never trains on anything else
    cfg = load_config(dataset=cfg_dataset)
    if args.epochs is not None:
        data = cfg.to_dict()
        data["pinn"]["epochs"] = args.epochs
        cfg = Config(data)

    windows, record_ids = load_fetal_band(cfg_dataset)
    logger.info(
        "Training PINN on %d windows from %d records: %s",
        windows.shape[0],
        len(record_ids),
        ", ".join(record_ids),
    )

    model = train_pinn_on_windows(cfg, windows)

    out_dir = Path("data/processed/pinn")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pinn.pt")
    # Resolved config alongside the checkpoint (CLAUDE.md section 6).
    with open(out_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    logger.info("Saved checkpoint to %s", out_dir / "pinn.pt")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "03_train_pinn.json", "w") as fh:
        json.dump(
            {
                "dataset": cfg_dataset,
                "records": record_ids,
                "n_windows": int(windows.shape[0]),
                "epochs": cfg.pinn.epochs,
                "readout": cfg.pinn.readout,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
