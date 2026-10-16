#!/usr/bin/env python3
"""scripts/eval_bland_altman.py

R-R interval agreement (Bland-Altman) between the extracted fetal ECG and the per-cohort
timing reference -- the metric in spec section 7.3 that no other script computes:

    ADFECGDB  fetal R peaks from the .qrs annotations  (PINN is IN-SAMPLE here)
    CinC 2013 fetal R peaks from the .fqrs annotations (zero-shot)
    NInFEA    Doppler beat times (src/fcac/io/ninfea_doppler.py) (zero-shot, timing only)

For every arm (pinn, cancel+pinn, bss_pre, bss_memd, hybrid -- see scripts/bss_experiment.py)
and every window with >= 3 reference beats:

  1. detect R peaks in the extracted fECG (Pan-Tompkins, as everywhere else);
  2. match detections to reference beats one-to-one within +/- `--tol-ms`;
  3. keep an interval only when two CONSECUTIVE reference beats are matched to two CONSECUTIVE
     detections (no missed or spurious beat in between), giving a (detected, reference) R-R
     pair in ms;
  4. Bland-Altman on the pairs: bias = mean(detected - reference), limits of agreement
     = bias +/- 1.96 SD (fcac.evaluation.extraction.bland_altman_rr).

A Doppler peak trails the R wave by a roughly constant lag. A constant lag cancels in an
INTERVAL, so R-R agreement does not depend on it; it only matters for step 2 (matching),
where the global lag from scripts/eval_ninfea_doppler.py's estimator is applied
(`--lag-arm`, default bss_pre).

Read the numbers with these in mind:
  * Pairs from one recording are correlated, so the pooled limits of agreement are too
    narrow. The script also reports (a) a CLUSTER bootstrap (resampling recordings) 95% CI
    on the bias and on the limits, and (b) a per-recording Bland-Altman on mean R-R.
  * Only intervals between two matched beats count, so a failed extraction (maternal rate,
    few matches) contributes FEW pairs, not bad ones. Always read `coverage` and `n_pairs`
    next to the bias/limits: agreement among matched beats is not extraction quality.
  * Matching tolerance is 50 ms for the ECG-annotated cohorts (the locked F1 tolerance) and
    100 ms for Doppler, whose peak shape is wider (override with --tol-ms).

Writes reports/eval_bland_altman.json and reports/figures/bland_altman_<dataset>.png.

Usage:
    .venv/bin/python scripts/eval_bland_altman.py
    .venv/bin/python scripts/eval_bland_altman.py --datasets ninfea cinc2013

Implements: spec section 7.3 (R-R timing agreement, Bland-Altman); CLAUDE.md section 2
"Ground truth" (NInFEA Doppler -> timing metrics only).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bss_experiment as bx  # noqa: E402
import eval_zeroshot as ez  # noqa: E402

from fcac.config import load_config  # noqa: E402
from fcac.evaluation.doppler_ref import estimate_lag, lag_differences  # noqa: E402
from fcac.evaluation.extraction import bland_altman_rr  # noqa: E402
from fcac.memd.diagnostics import pan_tompkins_peaks  # noqa: E402
from fcac.pinn.model import PINN  # noqa: E402
from fcac.quality import assess_window  # noqa: E402
from fcac.viz.plots import apply_style, save_figure  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARMS = ("pinn", "cancel+pinn", "bss_pre", "bss_memd", "hybrid")
DEFAULT_TOL_MS = {"adfecgdb": 50.0, "cinc2013": 50.0, "ninfea": 100.0}
MIN_REF_BEATS = 3


def reference_times_s(dataset: str, record: str, cfg, fs: float) -> np.ndarray | None:
    """Reference beat times in seconds on the raw recording's time base, or None."""
    if dataset == "ninfea":
        path = Path(cfg.dataset.doppler.cache_dir) / f"{record}.npz"
        return np.load(path)["beat_times_s"] if path.exists() else None
    peaks = ez.reference_peaks(dataset, record, cfg)
    return None if peaks is None else np.sort(peaks) / fs


def match_rr_pairs(det_s: np.ndarray, ref_s: np.ndarray, tol_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(detected R-R, reference R-R) in ms for consecutive matched beats; see module docstring."""
    det_s, ref_s = np.sort(det_s), np.sort(ref_s)
    if len(det_s) < 2 or len(ref_s) < 2:
        return np.empty(0), np.empty(0)
    taken: dict[int, int] = {}                      # reference index -> detection index
    used: set[int] = set()
    for i, r in enumerate(ref_s):
        d = np.abs(det_s - r)
        for j in np.argsort(d):
            if d[j] > tol_s:
                break
            if int(j) not in used:
                taken[i] = int(j)
                used.add(int(j))
                break
    det_rr, ref_rr = [], []
    for i in range(len(ref_s) - 1):
        if i in taken and (i + 1) in taken and taken[i + 1] == taken[i] + 1:
            det_rr.append((det_s[taken[i + 1]] - det_s[taken[i]]) * 1000.0)
            ref_rr.append((ref_s[i + 1] - ref_s[i]) * 1000.0)
    return np.asarray(det_rr), np.asarray(ref_rr)


def collect(dataset: str, model: PINN, cfg_pinn) -> list[dict]:
    """One row per (record, arm, window): detected beat times, reference beats in the window, gate."""
    cfg = load_config(dataset=dataset)
    trim = float(cfg.preprocessing.edge_trim_seconds)
    fs = float(cfg.preprocessing.target_fs_hz)
    win_s = cfg.preprocessing.window.size_samples / fs
    stride_s = cfg.preprocessing.window.stride_samples / fs
    rows = []
    for band_path in sorted((Path("data/processed/fetal_band") / dataset).glob("*.npz")):
        record = band_path.stem
        ref_all = reference_times_s(dataset, record, cfg, fs)
        if ref_all is None:
            continue
        band = np.load(band_path, allow_pickle=True)
        pre = np.load(Path(cfg.dataset.interim_dir) / f"{record}.npz", allow_pickle=True)["windows"]
        outputs, _ = bx.arm_outputs(band["x"], pre, float(band["fs"]), cfg_pinn, model)
        for arm, fecg in outputs.items():
            for w in range(fecg.shape[0]):
                start = trim + w * stride_s
                rows.append({
                    "record": record, "arm": arm, "window": w, "start": start,
                    "det_s": start + pan_tompkins_peaks(fecg[w], fs) / fs,
                    "ref_s": ref_all[(ref_all >= start) & (ref_all < start + win_s)],
                    "gate": bool(assess_window(fecg[w], fs, cfg_pinn).acceptable),
                })
    return rows


def ba_stats(det_rr: np.ndarray, ref_rr: np.ndarray) -> dict:
    if len(det_rr) < 2:
        return {"n_pairs": int(len(det_rr)), "bias_ms": None, "loa_low_ms": None, "loa_high_ms": None}
    bias, lo, hi = bland_altman_rr(det_rr, ref_rr)
    return {"n_pairs": int(len(det_rr)), "bias_ms": bias, "loa_low_ms": lo, "loa_high_ms": hi,
            "sd_diff_ms": float(np.std(det_rr - ref_rr)),
            "mean_abs_diff_ms": float(np.mean(np.abs(det_rr - ref_rr)))}


def cluster_bootstrap(per_record: dict[str, tuple[np.ndarray, np.ndarray]], n_resamples: int, seed: int) -> dict:
    """95% CI of bias and of the limits of agreement, resampling RECORDINGS (not pairs)."""
    records = [r for r, (d, _) in per_record.items() if len(d) >= 2]
    if len(records) < 3:
        return {}
    rng = np.random.default_rng(seed)
    bias, lo, hi = [], [], []
    for _ in range(n_resamples):
        pick = rng.choice(records, size=len(records), replace=True)
        d = np.concatenate([per_record[r][0] for r in pick])
        g = np.concatenate([per_record[r][1] for r in pick])
        b, l, h = bland_altman_rr(d, g)
        bias.append(b); lo.append(l); hi.append(h)
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"bias_ci95_ms": pct(bias), "loa_low_ci95_ms": pct(lo), "loa_high_ci95_ms": pct(hi),
            "n_records_resampled": len(records)}


def score_arm(arm_rows: list[dict], lag_s: float, tol_s: float, n_resamples: int, seed: int, gate_only: bool) -> dict:
    per_record: dict[str, list] = {}
    n_ref_intervals = 0
    for r in arm_rows:
        if len(r["ref_s"]) < MIN_REF_BEATS or (gate_only and not r["gate"]):
            continue
        ref = r["ref_s"] - lag_s
        det_rr, ref_rr = match_rr_pairs(r["det_s"], ref, tol_s)
        n_ref_intervals += len(ref) - 1
        per_record.setdefault(r["record"], [[], []])
        per_record[r["record"]][0].append(det_rr)
        per_record[r["record"]][1].append(ref_rr)
    cat = {k: (np.concatenate(v[0]) if v[0] else np.empty(0), np.concatenate(v[1]) if v[1] else np.empty(0))
           for k, v in per_record.items()}
    det_all = np.concatenate([d for d, _ in cat.values()]) if cat else np.empty(0)
    ref_all = np.concatenate([g for _, g in cat.values()]) if cat else np.empty(0)
    out = ba_stats(det_all, ref_all)
    out["n_records_scored"] = len(cat)
    out["n_records_with_pairs"] = sum(len(d) >= 1 for d, _ in cat.values())
    out["n_reference_intervals"] = int(n_ref_intervals)
    out["coverage"] = float(len(det_all) / n_ref_intervals) if n_ref_intervals else None
    out.update(cluster_bootstrap(cat, n_resamples, seed))
    # per-recording Bland-Altman on mean R-R (one point per recording: no within-recording correlation)
    means = [(d.mean(), g.mean()) for d, g in cat.values() if len(d) >= 3]
    if len(means) >= 3:
        d_m, g_m = np.array([m[0] for m in means]), np.array([m[1] for m in means])
        out["per_record_mean_rr"] = ba_stats(d_m, g_m) | {"n_records": len(means)}
    out["_pairs"] = (det_all, ref_all)
    return out


def plot_dataset(dataset: str, results: dict) -> None:
    arms = [a for a in ARMS if results.get(a, {}).get("all_windows", {}).get("_pairs") is not None]
    apply_style()
    fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 4), sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms):
        s = results[arm]["all_windows"]
        det, ref = s["_pairs"]
        if len(det) < 2:
            ax.text(0.5, 0.5, f"{len(det)} pairs", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.scatter((det + ref) / 2, det - ref, s=8, alpha=0.4)
            ax.axhline(s["bias_ms"], color="k", lw=1, label=f"bias {s['bias_ms']:+.0f} ms")
            for v in (s["loa_low_ms"], s["loa_high_ms"]):
                ax.axhline(v, color="tab:red", ls="--", lw=0.9)
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{arm}\nn={s['n_pairs']} pairs, cov {s['coverage'] if s['coverage'] is None else round(s['coverage'], 2)}",
                     fontsize=9)
        ax.set_xlabel("mean R-R (ms)")
    axes[0][0].set_ylabel("detected - reference R-R (ms)")
    fig.suptitle(f"Bland-Altman R-R agreement -- {dataset} (red dashes = 95% limits of agreement, pooled)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = save_figure(fig, f"bland_altman_{dataset}")
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["adfecgdb", "cinc2013", "ninfea"],
                        choices=["adfecgdb", "cinc2013", "ninfea"])
    parser.add_argument("--tol-ms", type=float, default=None,
                        help="matching tolerance (default 50 ms ECG cohorts, 100 ms Doppler)")
    parser.add_argument("--lag-arm", default="bss_pre", help="NInFEA only: arm used to estimate the Doppler lag")
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    args = parser.parse_args()

    base = load_config()
    n_resamples, seed = base.evaluation.bootstrap.n_resamples, base.seed
    cfg_pinn = load_config(dataset="adfecgdb")
    model = PINN(cfg_pinn)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    # Merge into an existing report so datasets can be (re)run one at a time -- a full run is slow.
    out_path = Path("reports/eval_bland_altman.json")
    report: dict = {"min_reference_beats": MIN_REF_BEATS, "datasets": {}}
    if out_path.exists():
        report["datasets"] = json.load(open(out_path)).get("datasets", {})
    for dataset in args.datasets:
        tol_ms = args.tol_ms if args.tol_ms is not None else DEFAULT_TOL_MS[dataset]
        rows = collect(dataset, model, cfg_pinn)
        if not rows:
            logger.warning("%s: no windows with a reference on disk -- skipped", dataset)
            continue
        lag_s = 0.0
        entry: dict = {"tolerance_ms": tol_ms, "lag_ms": 0.0,
                       "n_records": len({r["record"] for r in rows}),
                       "role": {"adfecgdb": "train (PINN in-sample)", "cinc2013": "zero-shot",
                                "ninfea": "zero-shot for the PINN; timing reference only"}[dataset]}
        if dataset == "ninfea":
            diffs = np.concatenate([lag_differences(r["det_s"], r["ref_s"]) for r in rows
                                    if r["arm"] == args.lag_arm and len(r["ref_s"])] or [np.empty(0)])
            lag_s, conc = estimate_lag(diffs)
            lag_s = 0.0 if not np.isfinite(lag_s) else lag_s
            entry["lag_ms"], entry["lag_arm"], entry["lag_concentration"] = lag_s * 1000, args.lag_arm, conc
            logger.info("ninfea: Doppler lag %.0f ms (concentration %.2f, flat ~0.10)", lag_s * 1000, conc)

        results = {}
        for arm in ARMS:
            arm_rows = [r for r in rows if r["arm"] == arm]
            results[arm] = {
                "all_windows": score_arm(arm_rows, lag_s, tol_ms / 1000, n_resamples, seed, gate_only=False),
                "gate_pass_windows": score_arm(arm_rows, lag_s, tol_ms / 1000, n_resamples, seed, gate_only=True),
            }
            s = results[arm]["all_windows"]
            if s["bias_ms"] is None:
                logger.info("%s %-12s only %d pairs -- no Bland-Altman", dataset, arm, s["n_pairs"])
            else:
                logger.info("%s %-12s n=%5d pairs (%d rec, coverage %.2f)  bias %+6.1f ms  LoA [%+.0f, %+.0f] ms",
                            dataset, arm, s["n_pairs"], s["n_records_with_pairs"], s["coverage"] or 0.0,
                            s["bias_ms"], s["loa_low_ms"], s["loa_high_ms"])
        plot_dataset(dataset, results)
        for arm in results:
            for variant in results[arm].values():
                variant.pop("_pairs", None)
        entry["arms"] = results
        report["datasets"][dataset] = entry

    Path("reports").mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report | {"timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    logger.info("wrote reports/eval_bland_altman.json")


if __name__ == "__main__":
    main()
