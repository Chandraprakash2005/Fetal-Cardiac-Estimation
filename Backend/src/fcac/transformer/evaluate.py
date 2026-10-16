"""src/fcac/transformer/evaluate.py

LOSOCV over ADFECGDB and zero-shot evaluation on NInFEA for the Cardiac Age
transformer (CLAUDE.md §2 "Evaluation"). Wraps the generic fold driver in
fcac.evaluation.losocv so the fold-indexing logic has one implementation shared with
every other LOSOCV consumer, rather than being re-written per stage.

Implements: CLAUDE.md §2 "Evaluation" (LOSOCV, zero-shot, bootstrap CI, ΔAge).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from fcac.config import Config
from fcac.evaluation.ageing import bootstrap_ci, mae, rmse
from fcac.evaluation.delta_age import delta_age_days
from fcac.evaluation.losocv import losocv_folds, run_losocv
from fcac.evaluation.ood import OODStats, mahalanobis_distance
from fcac.cvae.train import train_cvae
from fcac.transformer.dataset import pad_batch
from fcac.transformer.latents import encode_features, impute_missing
from fcac.transformer.model import CardiacAgeTransformer
from fcac.transformer.train import train_transformer_on_recordings


@torch.no_grad()
def predict_recordings(
    model: CardiacAgeTransformer, z_by_recording: List[torch.Tensor], crop_len: int
) -> torch.Tensor:
    """One CA prediction per recording (real days), averaged over non-overlapping
    crop_len chunks so a recording longer than max_seq_len is still fully covered.
    Un-scales the model's raw output using `model.target_mean`/`target_std`, set by
    `train_transformer_on_recordings` -- see model.py docstring for why."""
    model.eval()
    predictions = []
    for z in z_by_recording:
        chunks = [z[i : i + crop_len] for i in range(0, len(z), crop_len)]
        z_padded, pad_mask = pad_batch(chunks)
        scaled_prediction = model(z_padded, pad_mask).mean()
        predictions.append(scaled_prediction * model.target_std + model.target_mean)
    return torch.stack(predictions)


def _fold_predictions(test_idx, predicted, ga) -> Dict[str, Any]:
    """Per-fold result for a held-out subject with one or more recordings: `records` lists every
    held-out recording ({index, predicted_ca_days, ga_days}); when there is exactly one, its
    `predicted_ca_days`/`ga_days` are also exposed at the top level."""
    records = [
        {"index": int(i), "predicted_ca_days": float(p), "ga_days": float(g)}
        for i, p, g in zip(test_idx, predicted, ga)
    ]
    out: Dict[str, Any] = {"records": records}
    if len(records) == 1:
        out["predicted_ca_days"] = records[0]["predicted_ca_days"]
        out["ga_days"] = records[0]["ga_days"]
    return out


def losocv_adfecgdb(
    cfg: Config,
    z_by_recording: List[torch.Tensor],
    ga_days: torch.Tensor,
    subject_ids: List[str],
) -> List[Dict[str, Any]]:
    """CLAUDE.md §2 "Evaluation": LOSOCV; each fold holds out ALL recordings of one subject (a
    mother can have several), so no held-out mother is seen in training.

    LIMITATION (final review, 2026-09-29): this LOSOCV only re-trains and re-evaluates
    the transformer per fold. `z_by_recording` (CVAE latents) is precomputed once by
    `scripts/06_train_cvae.py` on the full ADFECGDB set before this function is ever
    called, and the CVAE's InfoNCE positive pairing includes cross-subject pairs
    within `cvae.contrastive.ga_window_days` of each other -- so a held-out
    recording's own GA label can have shaped its latent geometry through those
    cross-subject positives, before the fold that claims to hold it out even starts.
    This matches the original (pre-port) design, which also trained one CVAE on all
    of ADFECGDB and reused its latents for every transformer fold; it was not
    introduced by this plan. A stricter LOSOCV would refit the CVAE inside each fold
    on the training subjects only. Left as-is pending an explicit decision on whether
    the cost (5x CVAE training instead of 1x) is worth it."""

    def train_fn(train_idx):
        z_train = [z_by_recording[i] for i in train_idx]
        ga_train = ga_days[torch.as_tensor(train_idx)]
        return train_transformer_on_recordings(cfg, z_train, ga_train)

    def eval_fn(model, test_idx):
        z_test = [z_by_recording[i] for i in test_idx]
        pred = predict_recordings(model, z_test, cfg.transformer.crop_len)
        target = ga_days[torch.as_tensor(test_idx)]
        return _fold_predictions(test_idx, pred.tolist(), target.tolist())

    return run_losocv(subject_ids, train_fn, eval_fn)


def zero_shot_ninfea(
    cfg: Config,
    model: CardiacAgeTransformer,
    z_by_recording: List[torch.Tensor],
    ga_days: torch.Tensor,
    ood_stats: OODStats,
) -> Dict[str, Any]:
    """CLAUDE.md §2 "Evaluation": NInFEA, zero-shot, no fine-tuning. CLAUDE.md §3
    "GA coverage gap": training is 38-41 wks, NInFEA is 21-27 wks, so every zero-shot
    recording is flagged against the ADFECGDB-fitted OOD threshold rather than
    silently treated as in-distribution.

    `ood_stats` must be fit on per-WINDOW latents (e.g. `fit_ood(torch.cat(z_train))`),
    not per-recording means -- ADFECGDB has only 5 recordings, far fewer than the
    64-dim latent space, so a mean-per-recording fit is rank-deficient and either
    flags everything or nothing depending on how it happens to land (final review,
    2026-09-29). Each zero-shot recording is scored the same way: the MEDIAN distance
    of its own windows, not the distance of one pre-averaged latent -- averaging
    windows first collapses exactly the geometry the detector needs."""
    predicted = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)
    ood_distance = torch.stack(
        [mahalanobis_distance(z, ood_stats).median() for z in z_by_recording]
    )

    mae_point, ci_low, ci_high = bootstrap_ci(ga_days.numpy(), predicted.numpy(), mae, cfg)
    return {
        "predicted_ca_days": predicted.tolist(),
        "ga_days": ga_days.tolist(),
        "delta_age_days": delta_age_days(predicted.numpy(), ga_days.numpy()).tolist(),
        "mae_days": mae_point,
        "mae_ci95": [ci_low, ci_high],
        "rmse_days": rmse(ga_days.numpy(), predicted.numpy()),
        "ood_distance": ood_distance.tolist(),
        "ood_flag": (ood_distance > ood_stats.threshold).tolist(),
    }


def _refit_fold(cfg: Config, records: List[Dict[str, Any]], train_idx, test_idx) -> Dict[str, Any]:
    """One strict LOSOCV fold: fit imputation medians, the CVAE and the transformer on the
    training records only, then predict the held-out subject's recordings."""
    train = [records[int(i)] for i in train_idx]
    x = np.concatenate([r["x"] for r in train])
    subject_ids = np.concatenate([np.full(len(r["x"]), r["subject"]) for r in train])
    ga = np.concatenate([np.full(len(r["x"]), r["ga_days"], dtype=np.float64) for r in train])
    imputed, report = impute_missing(x)
    cvae = train_cvae(cfg, torch.from_numpy(imputed.astype(np.float32)), subject_ids, ga)
    fill = np.array([v["fill_value"] for v in report.values()], dtype=np.float32)
    z_train = [encode_features(cvae, r["x"], fill) for r in train]
    ga_train = torch.tensor([r["ga_days"] for r in train], dtype=torch.float32)
    model = train_transformer_on_recordings(cfg, z_train, ga_train)

    test = [records[int(i)] for i in test_idx]
    z_test = [encode_features(cvae, r["x"], fill) for r in test]
    pred = predict_recordings(model, z_test, cfg.transformer.crop_len)
    return _fold_predictions(test_idx, pred.tolist(), [r["ga_days"] for r in test])


def _refit_fold_worker(args) -> Dict[str, Any]:
    """Process-pool entry point (spawned workers cannot share the parent's Config object, so it
    travels as a dict)."""
    cfg_dict, records, train_idx, test_idx, n_threads = args
    torch.set_num_threads(n_threads)
    return _refit_fold(Config(cfg_dict), records, train_idx, test_idx)


def losocv_refit_cvae(
    cfg: Config, records: List[Dict[str, Any]], workers: int = 1
) -> List[Dict[str, Any]]:
    """Strict leave-one-subject-out: inside every fold the CVAE, its NaN-imputation
    medians AND the transformer are all fitted on the training subjects only, so the
    held-out recording influences nothing (unlike `losocv_adfecgdb`, which reuses one
    CVAE fit on every recording -- see its LIMITATION).

    records : dicts with `x` (W, 12 raw features, NaNs intact), `subject` (str; a subject may
              have several records, all held out together), `ga_days` (float, required).
    workers : folds are independent, so `workers` > 1 runs that many in parallel worker
              processes (CPU threads are split between them; each fold is seeded, so results do
              not depend on `workers`). Fold order is preserved.
    Returns run_losocv-style fold dicts: held_out_subject and `records` (see _fold_predictions)."""
    folds = losocv_folds([r["subject"] for r in records])
    if workers <= 1:
        return [
            {"held_out_subject": f["held_out_subject"],
             **_refit_fold(cfg, records, f["train_idx"], f["test_idx"])}
            for f in folds
        ]

    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_threads = max(1, (os.cpu_count() or workers) // workers)
    jobs = [(cfg.to_dict(), records, f["train_idx"], f["test_idx"], n_threads) for f in folds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_refit_fold_worker, jobs))
    return [{"held_out_subject": f["held_out_subject"], **r} for f, r in zip(folds, results)]
