#!/usr/bin/env python3
"""scripts/eval_snr_adfecgdb.py

SNR and MSE of every extraction arm against ADFECGDB's `Direct_1` scalp channel (CLAUDE.md §2
"Ground truth": ADFECGDB only; NInFEA and CinC 2013 have no amplitude reference).

Two SNRs per window (ICC 2026 eq. 3 form, 10*log10(sum y^2 / sum (y - y_hat)^2)):
  * unit-gain  : extracted and reference both z-scored per window, extracted sign-aligned to the
                 reference (BSS has an arbitrary sign). Closest to the paper's definition.
  * gain-matched: the best scalar gain per window (fcac.evaluation.extraction.gain_matched_snr_db).
Both measure waveform agreement with the scalp ECG; neither extractor is trained to reproduce
the reference amplitude, unlike the ICC 2026 CNN (14.4 dB), so those figures are not comparable.
Note the abdominal fetal ECG has a different morphology from the scalp lead, which caps SNR even
for a perfect extraction. Oracle baselines (best channel per window, picked WITH the reference)
are shown for context only.

The PINN was trained on ADFECGDB, so its numbers here are in-sample.

Usage: python scripts/eval_snr_adfecgdb.py

Implements: CLAUDE.md §2 "Evaluation" / §5 evaluation/extraction.py (SNR).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bss_experiment as bx  # noqa: E402

from fcac.config import load_config  # noqa: E402
from fcac.evaluation.extraction import gain_matched_snr_db, mse  # noqa: E402
from fcac.pinn.model import PINN  # noqa: E402
from fcac.quality import assess_window  # noqa: E402

ARMS = ("pinn", "cancel+pinn", "bss_pre", "hybrid")


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def best_lag_snr(extracted: np.ndarray, ref: np.ndarray, max_lag: int = 40) -> tuple[float, int]:
    """Gain-matched SNR maximised over a +/- `max_lag` sample shift (ms at 1 kHz) of the extracted
    signal: how much of the gap to the scalp lead is just a few ms of timing offset between the
    abdominal and scalp fetal R waves. Returns (SNR dB, best lag)."""
    x, y = zscore(extracted), zscore(ref)
    n = len(x)
    best_r, best_l = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        a, b = (x[lag:], y[: n - lag]) if lag >= 0 else (x[: n + lag], y[-lag:])
        r = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if abs(r) > abs(best_r):
            best_r, best_l = r, lag
    return float(-10 * np.log10(max(1e-12, 1 - best_r**2))), best_l


def unit_gain(extracted: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """(SNR dB, MSE) with both z-scored and the extracted sign-aligned to the reference."""
    x, y = zscore(extracted), zscore(ref)
    if np.dot(x, y) < 0:
        x = -x
    noise = float(np.sum((y - x) ** 2))
    return (float(10 * np.log10(np.sum(y ** 2) / noise)) if noise > 0 else float("inf")), mse(x, y)


def main() -> None:
    cfg = load_config(dataset="adfecgdb")
    model = PINN(cfg)
    model.load_state_dict(torch.load("data/processed/pinn/pinn.pt", map_location="cpu", weights_only=True))
    model.eval()

    rows = []  # one per (record, arm, window)
    for band_path in sorted(Path("data/processed/fetal_band/adfecgdb").glob("*.npz")):
        rid = band_path.stem
        band = np.load(band_path, allow_pickle=True)
        x, fs = band["x"], float(band["fs"])
        interim = np.load(Path(cfg.dataset.interim_dir) / f"{rid}.npz", allow_pickle=True)
        pre, ref = interim["windows"], interim["ref_windows"]
        outputs, _ = bx.arm_outputs(x, pre, fs, cfg, model)
        for w in range(len(ref)):
            gate_cache = {}
            for arm in ARMS:
                f = outputs[arm][w]
                snr_g, gain, r = gain_matched_snr_db(f, ref[w])
                snr_u, mse_u = unit_gain(f, ref[w])
                snr_l, lag_l = best_lag_snr(f, ref[w])
                gate_cache[arm] = bool(assess_window(f, fs, cfg).acceptable)
                rows.append({"record": rid, "arm": arm, "snr_gain": snr_g, "snr_unit": snr_u, "mse_unit": mse_u,
                             "abs_r": abs(r), "gate": gate_cache[arm],
                             "snr_lag": snr_l, "lag": lag_l})
            # oracle baselines, context only: best single channel picked using the reference
            for name, chans in (("raw_best_channel (oracle)", pre[w]), ("memd_best_channel (oracle)", x[w])):
                best = max((gain_matched_snr_db(c, ref[w]) for c in chans), key=lambda t: t[0])
                rows.append({"record": rid, "arm": name, "snr_gain": best[0], "snr_unit": float("nan"),
                             "mse_unit": float("nan"), "abs_r": abs(best[2]), "gate": True,
                             "snr_lag": float("nan"), "lag": 0})

    arms = list(dict.fromkeys(r["arm"] for r in rows))
    records = sorted({r["record"] for r in rows})
    summary = {}
    print(f"{'arm':28s} {'gain-matched SNR dB':>20s} {'unit-gain SNR dB':>17s} {'unit-gain MSE':>14s} {'|r|':>6s}   (mean over windows; median in brackets)")
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        g, u, m, c = (np.array([r[k] for r in sel], dtype=float) for k in ("snr_gain", "snr_unit", "mse_unit", "abs_r"))
        gate = np.array([r["gate"] for r in sel])
        summary[arm] = {
            "n_windows": len(sel),
            "snr_gain_mean": float(np.nanmean(g)), "snr_gain_median": float(np.nanmedian(g)),
            "snr_unit_mean": float(np.nanmean(u)) if np.isfinite(u).any() else None,
            "snr_unit_median": float(np.nanmedian(u)) if np.isfinite(u).any() else None,
            "mse_unit_mean": float(np.nanmean(m)) if np.isfinite(m).any() else None,
            "abs_r_mean": float(np.nanmean(c)),
            "snr_lag_mean": float(np.nanmean([r["snr_lag"] for r in sel])) if arm in ARMS else None,
            "median_best_lag_ms": float(np.median([r["lag"] for r in sel])) if arm in ARMS else None,
            "snr_gain_mean_gate_pass": float(np.nanmean(g[gate])) if gate.any() else None,
            "snr_gain_mean_gate_fail": float(np.nanmean(g[~gate])) if (~gate).any() else None,
            "per_record_snr_gain_mean": {rid: float(np.nanmean([r["snr_gain"] for r in sel if r["record"] == rid])) for rid in records},
        }
        s = summary[arm]
        uf = f"{s['snr_unit_mean']:.2f} ({s['snr_unit_median']:.2f})" if s["snr_unit_mean"] is not None else "n/a"
        mf = f"{s['mse_unit_mean']:.3f}" if s["mse_unit_mean"] is not None else "n/a"
        print(f"{arm:28s} {s['snr_gain_mean']:>12.2f} ({s['snr_gain_median']:.2f}) {uf:>17s} {mf:>14s} {s['abs_r_mean']:>6.2f}")
    print("\ngain-matched SNR allowing a +/-40 ms shift (mean dB) and the median best shift:")
    for arm in ARMS:
        print(f"  {arm:14s} {summary[arm]['snr_lag_mean']:.2f} dB   median best shift {summary[arm]['median_best_lag_ms']:+.0f} ms")
    print("\nper-record gain-matched SNR (dB, mean over windows):")
    print(f"{'arm':28s} " + " ".join(f"{rid:>6s}" for rid in records))
    for arm in arms:
        print(f"{arm:28s} " + " ".join(f"{summary[arm]['per_record_snr_gain_mean'][rid]:>6.2f}" for rid in records))

    Path("reports").mkdir(exist_ok=True)
    with open("reports/eval_snr_adfecgdb.json", "w") as fh:
        json.dump({"summary": summary, "timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)


if __name__ == "__main__":
    main()
