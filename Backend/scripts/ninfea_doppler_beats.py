#!/usr/bin/env python3
"""scripts/ninfea_doppler_beats.py

Extracts fetal beat timing from NInFEA's pulsed-wave Doppler bitmaps for every recording and
caches it (data/interim/ninfea_doppler/<n>.npz: beat_times_s, x_up, x_down), so the beat
reference can score extraction on NInFEA (see scripts/eval_ninfea_doppler.py).

Needs data/raw/ninfea_pwd/pwd_images/<n>.bmp (PhysioNet ninfea/1.0.0/pwd_images, ~545 MB;
`python scripts/00_download.py --dataset ninfea_pwd`). Images that are not downloaded yet are
skipped, so the script can be re-run as the download completes.

Per recording it also cross-checks the Doppler image duration (columns / fs_doppler) against
the ECG record length -- a mismatch would mean the shared time origin is wrong.

Usage: python scripts/ninfea_doppler_beats.py

Implements: CLAUDE.md §3 NInFEA Doppler beat timing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wfdb

from fcac.config import load_config
from fcac.io.ninfea_doppler import beat_times_from_envelope, doppler_envelope, load_bmp_gray

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config(dataset="ninfea")
    d = cfg.dataset.doppler
    images_dir, cache_dir = Path(d.images_dir), Path(d.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for n in range(1, int(cfg.dataset.n_recordings) + 1):
        bmp = images_dir / f"{n}.bmp"
        if not bmp.exists():
            logger.warning("%d: %s not downloaded yet, skipping", n, bmp)
            continue
        try:
            x_up, x_down, axis = doppler_envelope(load_bmp_gray(bmp))
        except Exception as exc:  # a corrupt/partial image must not stop the batch
            logger.warning("%d: envelope extraction failed (%s)", n, exc)
            records.append({"record_id": str(n), "status": f"failed: {exc}"})
            continue
        times, info = beat_times_from_envelope(x_up, x_down, cfg)
        header = wfdb.rdheader(str(Path(cfg.dataset.raw_dir) / str(n)))
        ecg_s = header.sig_len / header.fs
        np.savez_compressed(cache_dir / f"{n}.npz", beat_times_s=times, x_up=x_up, x_down=x_down)
        records.append({
            "record_id": str(n), "status": "ok", **info, "ecg_duration_s": float(ecg_s),
            "duration_mismatch_s": float(info["duration_s"] - ecg_s),
        })
        logger.info("%d: %d beats, %.0f bpm, Doppler %.3f s vs ECG %.3f s",
                    n, info["n_beats"], info["median_rate_bpm"], info["duration_s"], ecg_s)

    ok = [r for r in records if r["status"] == "ok"]
    summary = {
        "n_ok": len(ok),
        "median_rate_bpm": float(np.nanmedian([r["median_rate_bpm"] for r in ok])) if ok else None,
        "max_abs_duration_mismatch_s": float(max(abs(r["duration_mismatch_s"]) for r in ok)) if ok else None,
        "records": records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/ninfea_doppler_beats.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote reports/ninfea_doppler_beats.json (%d ok)", len(ok))


if __name__ == "__main__":
    main()
