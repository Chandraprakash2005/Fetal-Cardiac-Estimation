"""src/fcac/io/ninfea_doppler.py

Fetal beat timing from NInFEA's pulsed-wave Doppler (PWD) bitmaps. The dataset ships the
Doppler only as long .bmp images (pwd_images/<n>.bmp, one image column per 1/284 s, same time
origin as the ECG -- see `ninfea_doppler.fs_doppler` and the dataset's own viewer
Interfaccia_dataset.m) plus a MATLAB envelope extractor (envelope_extraction.m, Ortu, Sulas
& Pani 2019, GPL-3). `doppler_envelope` is a Python port of that extractor; the beat
detector on top of it is ours -- the dataset provides no beat annotations.

A Doppler peak marks blood flow, not the ECG R wave: it lags the R wave by an
electromechanical delay plus vessel-depth latency, so beat times from here are a TIMING
reference with an unknown constant offset, not R-peak ground truth (CLAUDE.md §2 "Ground
truth": timing metrics only).

Implements: CLAUDE.md §3 NInFEA "Synchronised pulsed-wave Doppler -> fetal beat timing".
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage, signal as sp_signal

from fcac.config import Config

Image.MAX_IMAGE_PIXELS = None  # the stitched PWD images exceed PIL's default safety limit


def load_bmp_gray(path: str | Path) -> np.ndarray:
    """Load a bitmap as uint8 grayscale (ITU-R 601 luma, the same weights as MATLAB rgb2gray)."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def median_otsu_threshold(gray: np.ndarray, k: int = 5) -> int:
    """2-D Otsu threshold on the (pixel value, k x k median) histogram -- port of
    `median_otsu2d` in envelope_extraction.m. Returns the 1-based threshold level used there."""
    half = k // 2
    med = ndimage.median_filter(gray, size=k, mode="reflect")
    f = gray[half:-half, half:-half].astype(np.int64)
    g = med[half:-half, half:-half].astype(np.int64)
    p = np.bincount((f * 256 + g).ravel(), minlength=256 * 256).reshape(256, 256).astype(np.float64)
    p /= p.sum()

    idx = np.arange(1, 257, dtype=np.float64)
    cum = lambda a: a.cumsum(axis=0).cumsum(axis=1)  # noqa: E731
    A, AI, AJ = cum(p), cum(p * idx[:, None]), cum(p * idx[None, :])
    tot_i, tot_j = AI[-1, -1], AJ[-1, -1]

    s = np.arange(1, 256)[:, None]
    t = np.arange(1, 256)[None, :]
    def block(X):  # sum over rows 1..s, cols 1..t  and the complementary (s+1.., t+1..) block
        head = X[s - 1, t - 1]
        tail = X[-1, -1] - X[s - 1, -1] - X[-1, t - 1] + X[s - 1, t - 1]
        return head, tail
    w0, w1 = block(A)
    i0, i1 = block(AI)
    j0, j1 = block(AJ)
    with np.errstate(divide="ignore", invalid="ignore"):
        trace = (w0 * ((i0 / w0 - tot_i) ** 2 + (j0 / w0 - tot_j) ** 2)
                 + w1 * ((i1 / w1 - tot_i) ** 2 + (j1 / w1 - tot_j) ** 2))
    trace = np.nan_to_num(trace, nan=-np.inf)
    t_best = int(np.unravel_index(np.argmax(trace), trace.shape)[1]) + 1  # loop value 1..255
    return t_best + 1


def doppler_envelope(gray: np.ndarray, min_blob_px: int = 70) -> Tuple[np.ndarray, np.ndarray, int]:
    """Upper and lower envelope of a PWD image, in pixels from the zero-velocity line.

    Port of envelope_extraction.m: blank the white x-axis line, binarise with the median-Otsu
    threshold, drop connected blobs under `min_blob_px` (4-connectivity), then per column take
    the extreme lit pixel above (x_up >= 0) and below (x_down <= 0) the axis.

    Returns (x_up (W,), x_down (W,), axis_row).
    """
    height, width = gray.shape
    gray = gray.copy()
    axis_rows = np.where(gray.sum(axis=1, dtype=np.int64) == 255 * width)[0]
    if axis_rows.size == 0:
        raise ValueError("no fully white zero-velocity line found in the image")
    axis = int(axis_rows[0])
    gray[axis_rows, :] = 0

    threshold = median_otsu_threshold(gray)
    binary = gray.astype(np.float64) / 255.0 > threshold / 255.0
    labels, n = ndimage.label(binary)  # default structure: 4-connectivity
    if n:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_blob_px
        keep[0] = False
        binary = keep[labels]

    up, down = binary[:axis, :], binary[axis + 1:, :]
    rows_up = np.arange(axis)[:, None]
    # topmost lit pixel above the axis; axis itself when none
    first_up = np.where(up.any(axis=0), np.argmax(up, axis=0), axis)
    # lowest lit pixel below the axis (1-based distance from the axis); 0 when none
    last_down = np.where(down.any(axis=0), down.shape[0] - np.argmax(down[::-1], axis=0), 0)
    del rows_up
    return (axis - first_up).astype(np.float32), (-last_down).astype(np.float32), axis


def beat_times_from_envelope(
    x_up: np.ndarray, x_down: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, dict]:
    """Doppler cardiac-cycle peak times (seconds) from the envelope.

    The signal is the envelope's total vertical extent (x_up - x_down), lightly smoothed;
    peaks must be at least `min_beat_interval_s` apart and stand out by `prominence_frac` of
    the extent's robust range. Returns (times_s, diagnostics)."""
    d = cfg.dataset.doppler
    fs = float(d.fs_doppler)
    extent = (x_up - x_down).astype(np.float64)
    win = max(1, int(round(d.smooth_s * fs)))
    smooth = np.convolve(extent, np.ones(win) / win, mode="same")
    lo, hi = np.percentile(smooth, [10, 90])
    prominence = max(float(d.prominence_frac) * (hi - lo), 1e-6)
    peaks, _ = sp_signal.find_peaks(
        smooth, distance=max(1, int(round(d.min_beat_interval_s * fs))), prominence=prominence
    )
    times = peaks / fs
    rate = float(60.0 / np.median(np.diff(times))) if len(times) >= 3 else float("nan")
    return times, {"n_beats": int(len(times)), "median_rate_bpm": rate, "duration_s": len(extent) / fs}
