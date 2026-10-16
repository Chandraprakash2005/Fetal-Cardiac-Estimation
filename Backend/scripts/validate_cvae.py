#!/usr/bin/env python3
"""scripts/validate_cvae.py

Validates the Contrastive VAE (CLAUDE.md §2 "CVAE") on HELD-OUT data, each metric against
an explicit baseline. Uses the same pooled windows scripts/06 trains on (ADFECGDB + the
quality-gated NInFEA recordings) and the same hyperparameters; a fresh CVAE is fitted per
split, with NaN-imputation medians taken from that split's training windows only.

Two protocols
  A. Within-subject window hold-out (5 seeded repeats, ~20% of each subject's windows):
       * held-out loss terms (reconstruction, KL, InfoNCE) next to the training ones
       * same-subject nearest-neighbour retrieval -- the direct test of the InfoNCE term --
         against chance and against raw standardised features
  B. Leave-one-subject-out (every subject held out once):
       * reconstruction R^2 per feature on OBSERVED entries only (imputed values excluded:
         scoring them scores a constant we inserted), vs predicting the training mean
       * active latent units (posterior-collapse check)
       * does the latent hold age information? kNN GA regression on held-out NInFEA mothers
         (real GA labels) from the latent vs raw features vs the training-mean baseline.
         ADFECGDB GA is not available (or synthetic), so it is never used as a target.

Usage: python scripts/validate_cvae.py [--repeats 5]

Implements: CLAUDE.md §2 "CVAE" (validation).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fcac.config import Config, load_config
from fcac.cvae.losses import info_nce_loss, kl_divergence
from fcac.cvae.model import ContrastiveVAE
from fcac.cvae.train import build_positive_mask, train_cvae
from fcac.evaluation.cvae_metrics import (
    active_units, knn_regress, r2_from_sums, recon_sums, same_subject_nn,
)
from fcac.features.extract import FEATURE_NAMES
from fcac.transformer.latents import impute_missing, load_recording_features

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pooled(cfg):
    X, subj, ga, dset = [], [], [], []
    for dataset in cfg.pooled_training.datasets:
        dcfg = cfg if dataset == "adfecgdb" else load_config(dataset=dataset)
        ids = list(dcfg.dataset.record_ids) if "record_ids" in dcfg.dataset else None
        for rid, x in load_recording_features(dataset, ids).items():
            g = dcfg.dataset.ga_days[rid] if rid in dcfg.dataset.ga_days else None
            s = (f"{dataset}:S{dcfg.dataset.subject_of_record[rid]}"
                 if "subject_of_record" in dcfg.dataset else f"{dataset}:{rid}")
            X.append(x); subj += [s] * len(x); dset += [dataset] * len(x)
            ga += [np.nan if g is None else float(g)] * len(x)
    return np.concatenate(X), np.array(subj), np.array(ga), np.array(dset)


class Fitted:
    """A CVAE fitted on a set of windows, with that set's imputation medians and mean."""

    def __init__(self, cfg, X, subj, ga):
        imputed, report = impute_missing(X)
        self.fill = np.array([v["fill_value"] for v in report.values()], dtype=np.float32)
        self.train_mean = np.nanmean(X, axis=0)
        self.train_mean = np.where(np.isnan(self.train_mean), 0.0, self.train_mean)
        self.model = train_cvae(cfg, torch.from_numpy(imputed.astype(np.float32)), subj, ga)
        self.model.eval()
        self.cfg = cfg
        # Fixed per-feature scale for config-independent reconstruction errors.
        std = imputed.std(axis=0)
        self.train_std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    @torch.no_grad()
    def forward(self, X):
        x = torch.from_numpy(np.where(np.isnan(X), self.fill, X).astype(np.float32))
        out = self.model(self.model.standardise(x))
        out["x_hat"] = self.model.destandardise(out["x_hat"])  # back in raw feature units
        return x, out

    def terms(self, X, subj, ga, with_nce=True):
        x, out = self.forward(X)
        res = {"recon_mse_raw": float(F.mse_loss(out["x_hat"], x)),
               # error in units of the training std of each feature, comparable across configs
               "recon_mse_std": float((((out["x_hat"] - x).numpy() / self.train_std) ** 2).mean()),
               "kl": float(kl_divergence(out["mu"], out["logvar"]))}
        if with_nce:
            mask = build_positive_mask(subj, ga, self.cfg.cvae.contrastive.ga_window_days)
            res["infonce"] = float(info_nce_loss(out["p"], mask, self.cfg.cvae.contrastive.temperature))
        return res


def mean_sd(values):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std()), "n": int(len(v))} if len(v) else None


def protocol_a(cfg, X, subj, ga, repeats):
    rows = []
    for rep in range(repeats):
        rng = np.random.default_rng(cfg.seed + rep)
        test = np.zeros(len(X), dtype=bool)
        for s in np.unique(subj):
            idx = np.where(subj == s)[0]
            if len(idx) >= 2:
                test[rng.choice(idx, max(1, int(round(0.2 * len(idx)))), replace=False)] = True
        tr, te = ~test, test
        fit = Fitted(cfg, X[tr], subj[tr], ga[tr])
        row = {"train": fit.terms(X[tr], subj[tr], ga[tr]),
               "heldout": fit.terms(X[te], subj[te], ga[te])}
        _, out_tr = fit.forward(X[tr]); _, out_te = fit.forward(X[te])
        acc, chance, n = same_subject_nn(out_te["mu"].numpy(), subj[te], out_tr["mu"].numpy(), subj[tr])
        raw_tr = np.where(np.isnan(X[tr]), fit.fill, X[tr]); raw_te = np.where(np.isnan(X[te]), fit.fill, X[te])
        raw_acc, _, _ = same_subject_nn(raw_te, subj[te], raw_tr, subj[tr])
        row.update(retrieval_latent=acc, retrieval_raw=raw_acc, retrieval_chance=chance, n_queries=n,
                   active_units=active_units(out_tr["mu"].numpy()))
        rows.append(row)
        print(f"  A repeat {rep}: retrieval latent {acc:.3f} raw {raw_acc:.3f} chance {chance:.3f} (n={n})", flush=True)
    out = {k: mean_sd([r[k] for r in rows]) for k in
           ("retrieval_latent", "retrieval_raw", "retrieval_chance", "active_units")}
    for split in ("train", "heldout"):
        out[split] = {t: mean_sd([r[split][t] for r in rows]) for t in rows[0][split]}
    out["n_repeats"] = repeats
    return out


def protocol_b(cfg, X, subj, ga, dset):
    subjects = np.unique(subj)
    sse = np.zeros(X.shape[1]); sst = np.zeros(X.shape[1]); nobs = np.zeros(X.shape[1])
    active, ga_rows = [], []
    for s in subjects:
        te = subj == s; tr = ~te
        fit = Fitted(cfg, X[tr], subj[tr], ga[tr])
        x, out = fit.forward(X[te])
        a, b, c = recon_sums(X[te], out["x_hat"].numpy(), fit.train_mean)
        sse += a; sst += b; nobs += c
        _, out_tr = fit.forward(X[tr])
        active.append(active_units(out_tr["mu"].numpy()))
        if dset[te][0] == "ninfea" and np.isfinite(ga[te]).all():
            ref = tr & (dset == "ninfea") & np.isfinite(ga)
            if ref.sum() >= 3:
                _, out_ref = fit.forward(X[ref])
                z_pred = knn_regress(out["mu"].numpy(), out_ref["mu"].numpy(), ga[ref]).mean()
                raw_ref = np.where(np.isnan(X[ref]), fit.fill, X[ref]); raw_te = np.where(np.isnan(X[te]), fit.fill, X[te])
                raw_pred = knn_regress(raw_te, raw_ref, ga[ref]).mean()
                ga_rows.append({"subject": str(s), "true": float(ga[te][0]), "latent": float(z_pred),
                                "raw": float(raw_pred), "mean_baseline": float(ga[ref].mean())})
        print(f"  B held out {s}: active units {active[-1]}", flush=True)
    r2 = r2_from_sums(sse, sst, nobs)
    result = {"recon_r2": {n: (None if np.isnan(v) else float(v)) for n, v in zip(FEATURE_NAMES, r2)},
              "recon_r2_median": float(np.nanmedian(r2)),
              "n_features_better_than_mean": int(np.nansum(r2 > 0)), "n_features_scored": int(np.isfinite(r2).sum()),
              "active_units": mean_sd(active), "latent_dim": cfg.cvae.latent_dim}
    if ga_rows:
        t = np.array([r["true"] for r in ga_rows])
        result["ga_knn"] = {"n_mothers": len(ga_rows),
                            "mae_days": {k: float(np.abs(np.array([r[k] for r in ga_rows]) - t).mean())
                                         for k in ("latent", "raw", "mean_baseline")},
                            "per_mother": ga_rows}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--standardise", choices=["true", "false"], default=None,
                        help="override cvae.standardise_inputs")
    args = parser.parse_args()
    cfg = load_config(dataset="adfecgdb")
    if args.standardise is not None:
        data = cfg.to_dict()
        data["cvae"]["standardise_inputs"] = args.standardise == "true"
        cfg = Config(data)
    tag = "std" if cfg.cvae.standardise_inputs else "nostd"
    X, subj, ga, dset = load_pooled(cfg)
    print(f"{len(X)} windows, {len(np.unique(subj))} subjects, "
          f"{int(np.isfinite(ga).sum())} windows with a GA label", flush=True)

    print("Protocol A: within-subject window hold-out", flush=True)
    a = protocol_a(cfg, X, subj, ga, args.repeats)
    print("Protocol B: leave-one-subject-out", flush=True)
    b = protocol_b(cfg, X, subj, ga, dset)

    summary = {"standardise_inputs": bool(cfg.cvae.standardise_inputs), "n_windows": int(len(X)), "n_subjects": int(len(np.unique(subj))),
               "protocol_a_window_holdout": a, "protocol_b_loso": b,
               "timestamp": datetime.now(timezone.utc).isoformat()}
    Path("reports").mkdir(exist_ok=True)
    out_path = f"reports/validate_cvae_{tag}.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
