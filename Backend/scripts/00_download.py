#!/usr/bin/env python3
"""scripts/00_download.py

Downloads ADFECGDB, NInFEA, and CinC 2013 from PhysioNet into data/raw/<dataset>/.

Not run as part of this scaffold (CLAUDE.md §8: "Do not download data").

TODO(bhuvanesh): ADFECGDB is distributed as .edf files, which wfdb.dl_database may not
fetch correctly (it expects native WFDB .dat/.hea records) -- verify against the real
PhysioNet ADFECGDB file listing before relying on this for that dataset.

Usage: python scripts/00_download.py --dataset {adfecgdb,ninfea,cinc2013,ninfea_pwd}

Implements: CLAUDE.md §4 scripts/00_download.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import wfdb

from fcac.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ninfea_pwd() -> None:
    """NInFEA's pulsed-wave Doppler bitmaps (60 files, ~545 MB) and the dataset's MATLAB code.
    They are not WFDB records, so wfdb.dl_database does not fetch them."""
    import urllib.request

    cfg = load_config(dataset="ninfea")
    base = f"https://physionet.org/files/{cfg.dataset.physionet_id}"
    root = Path(cfg.dataset.doppler.images_dir).parent
    targets = [(f"code/{f}", root / "code" / f) for f in
               ("envelope_extraction.m", "Interfaccia_dataset.m", "ReadBinaryFile.m")]
    targets += [(f"pwd_images/{n}.bmp", Path(cfg.dataset.doppler.images_dir) / f"{n}.bmp")
                for n in range(1, int(cfg.dataset.n_recordings) + 1)]
    for rel, dest in targets:
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", rel)
        urllib.request.urlretrieve(f"{base}/{rel}", dest)


def download(dataset: str) -> None:
    if dataset == "ninfea_pwd":
        return download_ninfea_pwd()
    cfg = load_config(dataset=dataset)
    raw_dir = Path(cfg.dataset.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (%s) into %s", dataset, cfg.dataset.physionet_id, raw_dir)
    wfdb.dl_database(cfg.dataset.physionet_id, str(raw_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013", "ninfea_pwd"], required=True)
    args = parser.parse_args()
    download(args.dataset)


if __name__ == "__main__":
    main()
