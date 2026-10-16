"""tests/test_cvae.py

End-to-end tests on random tensors (CLAUDE.md §8 items 6/9).

Implements: CLAUDE.md §8 item 9.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fcac.config import Config, load_config
from fcac.cvae.augment import sample_latent_neighbours
from fcac.cvae.losses import (
    hybrid_loss,
    info_nce_loss,
    kl_divergence,
    kl_warmup_beta,
    reconstruction_loss,
)
from fcac.cvae.model import ContrastiveVAE
from fcac.cvae.train import build_positive_mask, train_cvae


def _with_epochs(cfg: Config, epochs: int) -> Config:
    data = cfg.to_dict()
    data["cvae"]["epochs"] = epochs
    return Config(data)


def test_cvae_forward_shapes():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    x = torch.randn(16, cfg.cvae.input_dim)
    out = model(x)
    assert out["x_hat"].shape == x.shape
    assert out["mu"].shape == (16, cfg.cvae.latent_dim)
    assert out["logvar"].shape == (16, cfg.cvae.latent_dim)
    assert out["z"].shape == (16, cfg.cvae.latent_dim)
    assert out["p"].shape == (16, cfg.cvae.proj_dim)
    assert torch.allclose(out["p"].norm(dim=-1), torch.ones(16), atol=1e-5)


def test_cvae_eval_mode_is_deterministic():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    model.eval()
    x = torch.randn(4, cfg.cvae.input_dim)
    out1, out2 = model(x), model(x)
    assert torch.equal(out1["z"], out2["z"])  # eval: z = mu, no sampling noise


def test_build_positive_mask_same_subject_or_ga_within_window():
    subject_ids = np.array(["a", "a", "b", "b"])
    ga_days = np.array([280.0, 284.0, 280.0, 300.0])
    mask = build_positive_mask(subject_ids, ga_days, ga_window_days=7.0)
    assert bool(mask[0, 1])  # same subject
    assert bool(mask[0, 2])  # different subject, |ΔGA|=0 <= 7
    assert not bool(mask[1, 3])  # different subject, |ΔGA|=20 > 7


def test_losses_are_finite():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    x = torch.randn(8, cfg.cvae.input_dim)
    out = model(x)

    assert torch.isfinite(reconstruction_loss(out["x_hat"], x))
    assert torch.isfinite(kl_divergence(out["mu"], out["logvar"]))

    positive_mask = torch.eye(8, dtype=torch.bool).roll(1, dims=1)
    assert torch.isfinite(
        info_nce_loss(out["p"], positive_mask, cfg.cvae.contrastive.temperature)
    )
    assert torch.isfinite(hybrid_loss(out, x, positive_mask, epoch=0, cfg=cfg))


def test_info_nce_loss_is_zero_not_nan_with_no_positives():
    p = F.normalize(torch.randn(6, 8), dim=-1)
    no_positives = torch.zeros(6, 6, dtype=torch.bool)
    loss = info_nce_loss(p, no_positives, temperature=0.07)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_kl_warmup_beta_ramps_linearly_then_holds():
    cfg = load_config()
    assert kl_warmup_beta(0, cfg) == pytest.approx(0.0)
    half = cfg.cvae.kl_warmup_epochs // 2
    assert kl_warmup_beta(half, cfg) == pytest.approx(cfg.cvae.beta_kl * half / cfg.cvae.kl_warmup_epochs)
    assert kl_warmup_beta(cfg.cvae.kl_warmup_epochs * 10, cfg) == pytest.approx(cfg.cvae.beta_kl)


def test_train_cvae_runs_end_to_end_on_random_tensors():
    cfg = _with_epochs(load_config(), epochs=2)
    n = 16
    features = torch.randn(n, cfg.cvae.input_dim)
    subject_ids = np.array([f"s{i % 4}" for i in range(n)])
    ga_days = np.full(n, 280.0)

    model = train_cvae(cfg, features, subject_ids, ga_days)
    out = model(features)
    assert torch.isfinite(out["x_hat"]).all()


def test_sample_latent_neighbours_shape():
    cfg = load_config()
    model = ContrastiveVAE(cfg)
    z = torch.randn(4, cfg.cvae.latent_dim)
    aug = sample_latent_neighbours(model, z, n_samples=3)
    assert aug.shape == (4, 3, cfg.cvae.input_dim)


def test_spearman_corr_perfect_rank_agreement():
    from fcac.cvae.diagnostics import spearman_corr

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    assert spearman_corr(a, b) == pytest.approx(1.0)


def test_latent_distance_vs_ga_gap_nan_when_all_same_ga():
    from fcac.cvae.diagnostics import latent_distance_vs_ga_gap

    z = torch.randn(10, 4)
    ga = torch.full((10,), 280.0)
    result = latent_distance_vs_ga_gap(z, ga)
    assert result != result  # NaN != NaN


def test_ridge_regression_loro_mae_days_recovers_linear_signal():
    from fcac.cvae.diagnostics import ridge_regression_loro_mae_days

    torch.manual_seed(0)
    n_per_subject, n_subjects = 20, 4
    subject_idx = torch.arange(n_subjects).repeat_interleave(n_per_subject)
    ga = 280.0 + subject_idx.float() * 5.0 + torch.randn(n_per_subject * n_subjects) * 0.01
    z = torch.stack([ga, torch.randn(n_per_subject * n_subjects)], dim=1)

    mae_days = ridge_regression_loro_mae_days(z, ga, subject_idx)
    assert mae_days < 5.0


def test_standardise_roundtrip_and_train_cvae_fits_the_scaler():
    cfg = load_config()
    data = cfg.to_dict()
    data["cvae"]["epochs"] = 1
    cfg = type(cfg)(data)
    gen = torch.Generator().manual_seed(0)
    scales = torch.tensor([1000.0, 1.0, 0.01] + [1.0] * (cfg.cvae.input_dim - 3))
    features = torch.randn(40, cfg.cvae.input_dim, generator=gen) * scales + 5.0
    model = train_cvae(cfg, features, np.array(["a", "b"] * 20), np.full(40, np.nan))
    assert torch.allclose(model.feat_mean, features.mean(0), atol=1e-4)
    assert torch.allclose(model.feat_std, features.std(0, unbiased=False), rtol=1e-4)
    assert torch.allclose(model.destandardise(model.standardise(features)), features, rtol=1e-4, atol=1e-3)
    assert model.standardise(features).std(0, unbiased=False).max() < 1.01


def test_constant_feature_keeps_unit_std_instead_of_dividing_by_zero():
    cfg = load_config()
    data = cfg.to_dict()
    data["cvae"]["epochs"] = 1
    cfg = type(cfg)(data)
    features = torch.randn(30, cfg.cvae.input_dim)
    features[:, 4] = 7.0
    model = train_cvae(cfg, features, np.array(["a", "b", "c"] * 10), np.full(30, np.nan))
    assert model.feat_std[4] == 1.0 and torch.isfinite(model.standardise(features)).all()


def test_scaler_can_be_disabled_by_config():
    cfg = load_config()
    data = cfg.to_dict()
    data["cvae"]["epochs"] = 1
    data["cvae"]["standardise_inputs"] = False
    cfg = type(cfg)(data)
    model = train_cvae(cfg, torch.randn(20, cfg.cvae.input_dim) * 50, np.array(["a", "b"] * 10), np.full(20, np.nan))
    assert torch.all(model.feat_mean == 0) and torch.all(model.feat_std == 1)
