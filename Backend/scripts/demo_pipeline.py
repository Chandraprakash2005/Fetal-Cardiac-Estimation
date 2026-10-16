#!/usr/bin/env python3
"""scripts/demo_pipeline.py

Input -> output walkthrough of every pipeline stage for ONE record, one figure per stage in
reports/figures/pipeline/:

  01_preprocessing  raw channel                 -> filtered / resampled / z-scored window
  02_memd           preprocessed window         -> IMFs (selected ones highlighted) + fetal band
  03_pinn           maternal-cancelled band     -> PINN-isolated fetal ECG (vs scalp reference)
  04_features       isolated fECG window        -> the 12 biomarkers
  05_cvae           12 features                 -> 64-d latent z (+ reconstruction)
  06_transformer    sequence of latents z       -> scalar cardiac age (CA)
  07_evaluation     CA vs GA over all records   -> MAE / CI / r / DeltaAge (LOSOCV)

Stages 1-3 and 4 are recomputed from the real pipeline code and cached artefacts under
data/; stages 5-6 use the saved checkpoints; stage 7 reads reports/08_evaluate.json.
Nothing is retrained.

Honesty notes baked into the figures:
  * ADFECGDB gestational ages are currently SYNTHETIC (random 38-41 wk, see the engineering
    log). Any figure that shows ADFECGDB GA/CA is stamped accordingly.
  * Stage 6 uses the checkpoint fitted on ALL records (scripts/07), so its prediction for a
    training record is a train-set fit, not an evaluation. Stage 7 is the held-out result.

Usage:
    .venv/bin/python scripts/demo_pipeline.py                       # adfecgdb r08, window 0
    .venv/bin/python scripts/demo_pipeline.py --dataset ninfea --record 10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from fcac.config import load_config
from fcac.evaluation.ageing import bootstrap_ci, mae, pearson_r
from fcac.evaluation.verify_units import verify_mae_units
from fcac.features.extract import FEATURE_NAMES
from fcac.memd.diagnostics import pan_tompkins_peaks
from fcac.pinn.coords import project_to_single_channel
from fcac.transformer.dataset import pad_batch
from fcac.transformer.latents import load_cvae_and_fill_values
from fcac.transformer.model import CardiacAgeTransformer
from fcac.viz.plots import apply_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("reports/figures/pipeline")
SYNTH = "ADFECGDB GA is SYNTHETIC (random 38-41 wk) -- not reportable"


def _save(fig, name: str) -> Path:
    path = save_figure(fig, name, figures_dir=OUT_DIR)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _norm(x):
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


# --------------------------------------------------------------------------------------
def stage_preprocessing_and_pinn(dataset: str, record: str, window: int) -> None:
    """Stages 1 and 3 reuse the dedicated scripts so the figures stay identical."""
    import plot_preprocessing
    import plot_separation

    plot_preprocessing.plot_one(dataset, record)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"reports/figures/preprocessing_{dataset}_{record}.png",
                OUT_DIR / f"01_preprocessing_{dataset}_{record}.png")
    if dataset == "adfecgdb":  # separation figure needs the scalp reference row
        plot_separation.plot_separation(dataset, record, window, "data/processed/pinn/pinn.pt")
        shutil.copy(f"reports/figures/separation_{dataset}_{record}_w{window}.png",
                    OUT_DIR / f"03_pinn_{dataset}_{record}_w{window}.png")
    else:
        logger.warning("03_pinn: the separation figure needs a scalp reference; skipped for %s",
                       dataset)


# --------------------------------------------------------------------------------------
def stage_memd(dataset: str, record: str, window: int) -> None:
    cfg = load_config(dataset=dataset)
    interim = np.load(Path(cfg.dataset.interim_dir) / f"{record}.npz", allow_pickle=True)
    fs = float(interim["fs"])
    x_in = interim["windows"][window]                               # (4, N)
    imfs = np.load(Path("data/memd") / dataset / record / f"win{window}.npz")["imfs"]  # (K,4,N)
    band_file = np.load(Path("data/processed/fetal_band") / dataset / f"{record}.npz",
                        allow_pickle=True)
    band = band_file["x"][window]
    selected = [int(k) for k in np.atleast_1d(band_file["selected"][window])]
    scores = band_file["scores"][window]
    K = imfs.shape[0]
    t = np.arange(x_in.shape[1]) / fs

    apply_style()
    fig, axes = plt.subplots(K + 2, 1, figsize=(12, 1.15 * (K + 2) + 1), sharex=True)
    axes[0].plot(t, x_in[0], color="k", lw=0.7)
    axes[0].set_title(f"INPUT: preprocessed window, channel 0, shape {x_in.shape} @ {fs:.0f} Hz",
                      fontsize=10, loc="left")
    for k in range(K):
        ax = axes[k + 1]
        chosen = k in selected
        ax.plot(t, imfs[k, 0], color="tab:orange" if chosen else "0.45", lw=0.7)
        if chosen:
            ax.set_facecolor("#fff1e0")
        ax.set_ylabel(f"IMF {k + 1}", rotation=0, labelpad=28, fontsize=8)
        ax.text(1.005, 0.5, f"score {scores[k]:.3g}" + ("  <- selected" if chosen else ""),
                transform=ax.transAxes, fontsize=7, va="center")
    axes[-1].plot(t, band[0], color="tab:blue", lw=0.8)
    axes[-1].set_title(f"OUTPUT: fetal-band reconstruction (sum of selected IMFs "
                       f"{[s + 1 for s in selected]}), shape {band.shape}", fontsize=10, loc="left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"MEMD -- {dataset}/{record}, window {window}: {K} IMFs from 4-channel input "
                 "(channel 0 shown)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    _save(fig, f"02_memd_{dataset}_{record}_w{window}")


# --------------------------------------------------------------------------------------
def stage_features(dataset: str, record: str, window: int) -> tuple[int, np.ndarray]:
    cfg = load_config(dataset=dataset)
    fecg_file = np.load(Path("data/processed/fecg") / dataset / f"{record}.npz", allow_pickle=True)
    fs = float(fecg_file["fs"])
    feat_all = np.load(Path("data/processed/features") / dataset / f"{record}.npz",
                       allow_pickle=True)["x"]
    # Window `window` of the record may have been gated out; features hold only gated windows.
    mask = fecg_file["quality_mask"].astype(bool)
    gated = np.where(mask)[0]
    if window not in gated:
        window = int(gated[0])
        logger.warning("features: requested window failed the quality gate; using window %d", window)
    fecg = fecg_file["fecg"][window]
    row = feat_all[int(np.where(gated == window)[0][0])]

    pool = np.concatenate([np.load(f, allow_pickle=True)["x"]
                           for ds in ("adfecgdb", "ninfea")
                           for f in sorted((Path("data/processed/features") / ds).glob("*.npz"))])
    mu, sd = np.nanmean(pool, 0), np.nanstd(pool, 0)
    sd[sd == 0] = 1.0
    z = (row - mu) / sd

    t = np.arange(fecg.size) / fs
    peaks = pan_tompkins_peaks(fecg, fs)
    apply_style()
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [1, 1.4]})
    a.plot(t, _norm(fecg), color="tab:blue", lw=0.8)
    a.plot(t[peaks], _norm(fecg)[peaks], "rv", ms=6, label=f"QRS detections, Pan-Tompkins ({len(peaks)})")
    a.legend(loc="upper right", fontsize=8)
    a.set_xlabel("time (s)")
    a.set_title(f"INPUT: isolated fetal ECG window, shape {fecg.shape} "
                f"({'PINN' if bool(fecg_file['from_pinn'][window]) else 'BSS fallback'} output)",
                loc="left", fontsize=10)
    colours = ["tab:gray" if not np.isfinite(v) else ("tab:green" if v >= 0 else "tab:purple")
               for v in z]
    b.bar(range(12), np.nan_to_num(z), color=colours)
    b.set_xticks(range(12))
    b.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
    b.axhline(0, color="k", lw=0.6)
    for i, (v, zi) in enumerate(zip(row, z)):
        b.text(i, (0 if not np.isfinite(zi) else zi) + (0.08 if not np.isfinite(zi) or zi >= 0 else -0.08),
               "NaN" if not np.isfinite(v) else f"{v:.3g}", ha="center",
               va="bottom" if not np.isfinite(zi) or zi >= 0 else "top", fontsize=8)
    b.set_ylabel("z-score vs pooled windows")
    b.set_title("OUTPUT: 12-feature vector (bar = z-score against all ADFECGDB+NInFEA windows, "
                "label = raw value, grey = NaN)", loc="left", fontsize=10)
    fig.suptitle(f"Feature extraction -- {dataset}/{record}, window {window}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, f"04_features_{dataset}_{record}_w{window}")
    return window, row


# --------------------------------------------------------------------------------------
def stage_cvae(dataset: str, record: str, row: np.ndarray, window: int) -> None:
    cfg = load_config(dataset=dataset)
    cvae, fill = load_cvae_and_fill_values(cfg)
    x = np.where(np.isnan(row), fill, row).astype(np.float32)
    with torch.no_grad():
        xs = cvae.standardise(torch.from_numpy(x)[None])
        out = cvae(xs)
    x_in = xs[0].numpy()
    x_hat = out["x_hat"][0].numpy()
    mu = out["mu"][0].numpy()

    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    idx = np.arange(12)
    a.bar(idx - 0.2, x_in, 0.4, label="input (standardised, NaN -> median fill)", color="tab:green")
    a.bar(idx + 0.2, x_hat, 0.4, label="CVAE reconstruction", color="tab:purple")
    a.set_xticks(idx)
    a.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right", fontsize=8)
    a.legend(fontsize=8)
    a.set_title("INPUT 12-d feature vector  ->  decoder reconstruction", fontsize=10, loc="left")
    b.bar(np.arange(mu.size), mu, color="tab:blue")
    b.set_xlabel("latent dimension")
    b.set_title(f"OUTPUT: 64-d latent z (posterior mean); {int((np.abs(mu) > 0.1).sum())} dims "
                "with |z| > 0.1", fontsize=10, loc="left")
    fig.suptitle(f"Contrastive VAE -- {dataset}/{record}, window {window}  "
                 f"(reconstruction MSE {np.mean((x_in - x_hat) ** 2):.3f} in standardised units)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, f"05_cvae_{dataset}_{record}_w{window}")


# --------------------------------------------------------------------------------------
def stage_transformer(dataset: str, record: str) -> None:
    cfg = load_config(dataset=dataset)
    cvae, fill = load_cvae_and_fill_values(cfg)
    feats = np.load(Path("data/processed/features") / dataset / f"{record}.npz",
                    allow_pickle=True)["x"].astype(np.float32)
    feats = np.where(np.isnan(feats), fill, feats).astype(np.float32)
    with torch.no_grad():
        z = cvae(cvae.standardise(torch.from_numpy(feats)))["mu"]          # (W, 64)

    model = CardiacAgeTransformer(cfg)
    model.load_state_dict(torch.load("data/processed/transformer/transformer.pt",
                                     map_location="cpu"))
    model.eval()
    crop = cfg.transformer.crop_len
    with torch.no_grad():
        chunks = [z[i:i + crop] for i in range(0, len(z), crop)]
        zp, mask = pad_batch(chunks)
        ca = float((model(zp, mask).mean() * model.target_std + model.target_mean).item())

    rep = json.load(open("reports/07_train_transformer.json"))
    ga = rep["ga_days"].get(f"{dataset}:{record}")
    synthetic = bool(rep.get("synthetic_ga")) and dataset == "adfecgdb"

    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    im = a.imshow(z.numpy().T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    a.set_xlabel("window index (5 s each)")
    a.set_ylabel("latent dim")
    a.set_title(f"INPUT: latent sequence z, shape {tuple(z.shape)} (windows x 64), "
                f"processed in chunks of {crop}", fontsize=10, loc="left")
    fig.colorbar(im, ax=a, fraction=0.03)
    bars = [("CA predicted", ca, "tab:blue")]
    if ga is not None:
        bars.insert(0, ("GA label", ga, "tab:gray"))
    b.bar([x[0] for x in bars], [x[1] for x in bars], color=[x[2] for x in bars])
    for i, (_, v, _) in enumerate(bars):
        b.text(i, v, f"{v:.1f} d\n({v / 7:.1f} wk)", ha="center", va="bottom", fontsize=9)
    b.set_ylim(0, max(v for _, v, _ in bars) * 1.25)
    b.set_ylabel("days")
    delta = f"  DeltaAge = CA - GA = {ca - ga:+.1f} d" if ga is not None else ""
    b.set_title("OUTPUT: scalar cardiac age" + delta, fontsize=10, loc="left")
    foot = ("TRAIN-SET FIT: this checkpoint was fitted on this record (not an evaluation; see "
            "figure 07 for held-out results)")
    if synthetic:
        foot += f"\n{SYNTH}"
    fig.text(0.5, 0.01, foot, ha="center", fontsize=9, color="firebrick")
    fig.suptitle(f"Transformer -- {dataset}/{record}", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, f"06_transformer_{dataset}_{record}")


# --------------------------------------------------------------------------------------
def stage_evaluation() -> None:
    d = json.load(open("reports/08_evaluate.json"))
    rows = []
    for f in d["folds"]:
        for r in f.get("records", [f]):
            ga = r.get("ga_days")
            ca = r.get("predicted_ca_days", r.get("ca_days"))
            if ga is not None and ca is not None:
                rows.append((f["dataset"], ga, ca))
    ds = np.array([r[0] for r in rows])
    ga = np.array([r[1] for r in rows], dtype=float)
    ca = np.array([r[2] for r in rows], dtype=float)
    cfg = load_config()

    buf = StringIO()
    with redirect_stdout(buf):
        verify_mae_units(ga, ca)
    unit_msg = buf.getvalue().strip()
    m, lo, hi = bootstrap_ci(ga, ca, mae, cfg)
    r = pearson_r(ga, ca)
    base = mae(ga, np.full_like(ga, ga.mean()))

    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    colours = {"adfecgdb": "tab:blue", "ninfea": "tab:orange"}
    for name in np.unique(ds):
        s = ds == name
        a.scatter(ga[s], ca[s], s=22, alpha=0.8, color=colours.get(name, "k"),
                  label=f"{name} (n={s.sum()})")
    lim = [min(ga.min(), ca.min()) - 10, max(ga.max(), ca.max()) + 10]
    a.plot(lim, lim, "k--", lw=0.8, label="CA = GA")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("GA label (days)"); a.set_ylabel("predicted CA (days, held-out)")
    a.legend(fontsize=8, loc="upper left")
    nin = ds == "ninfea"
    r_nin = pearson_r(ga[nin], ca[nin]) if nin.sum() > 2 else float("nan")
    a.set_title(f"MAE {m:.1f} d (95% CI {lo:.1f}-{hi:.1f}), Pearson r {r:.2f} pooled, "
                f"{r_nin:.2f} within NInFEA\n"
                f"predict-the-mean baseline MAE {base:.1f} d "
                "(pooled r mostly reflects the two cohorts being far apart)",
                fontsize=9, loc="left")
    delta = ca - ga
    order = np.argsort(ga)
    b.bar(np.arange(len(delta)), delta[order], color=[colours.get(x, "k") for x in ds[order]])
    b.axhline(0, color="k", lw=0.6)
    b.set_xlabel("records sorted by GA"); b.set_ylabel("DeltaAge = CA - GA (days)")
    b.set_title("DeltaAge per record", fontsize=10, loc="left")
    foot = unit_msg
    if d.get("synthetic_ga"):
        foot += f"   |   {SYNTH}"
    fig.text(0.5, 0.01, foot, ha="center", fontsize=9,
             color="firebrick" if d.get("synthetic_ga") else "k")
    fig.suptitle(f"Evaluation -- pooled leave-one-subject-out CV, n={len(ga)} records "
                 f"({d.get('experiment', '')})", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, "07_evaluation_pooled_losocv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="adfecgdb", choices=["adfecgdb", "ninfea"])
    parser.add_argument("--record", default=None)
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()
    record = args.record or {"adfecgdb": "r08", "ninfea": "10"}[args.dataset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage_preprocessing_and_pinn(args.dataset, record, args.window)
    stage_memd(args.dataset, record, args.window)
    window, row = stage_features(args.dataset, record, args.window)
    stage_cvae(args.dataset, record, row, window)
    stage_transformer(args.dataset, record)
    stage_evaluation()


if __name__ == "__main__":
    main()
