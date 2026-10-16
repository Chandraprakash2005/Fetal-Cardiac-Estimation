"""tests/test_transformer_evaluate.py

Synthetic-data tests for src/fcac/transformer/evaluate.py (CLAUDE.md §8 item 9,
CLAUDE.md §2 "Evaluation").
"""

import numpy as np
import pytest
import torch

from fcac.config import Config, load_config
from fcac.evaluation.ood import fit_ood
from fcac.transformer.evaluate import losocv_adfecgdb, zero_shot_ninfea
from fcac.transformer.train import train_transformer_on_recordings


def _with_epochs(cfg: Config, epochs: int) -> Config:
    data = cfg.to_dict()
    data["transformer"]["epochs"] = epochs
    return Config(data)


def test_losocv_adfecgdb_produces_one_prediction_per_recording():
    cfg = _with_epochs(load_config(), epochs=2)
    n_recordings = 5
    z_by_recording = [torch.randn(6, cfg.cvae.latent_dim) for _ in range(n_recordings)]
    ga_days = torch.tensor([266.0, 273.0, 280.0, 287.0, 294.0])
    subject_ids = ["r01", "r04", "r07", "r08", "r10"]

    results = losocv_adfecgdb(cfg, z_by_recording, ga_days, subject_ids)
    assert len(results) == n_recordings
    assert {r["held_out_subject"] for r in results} == set(subject_ids)
    for r in results:
        assert "predicted_ca_days" in r and "ga_days" in r


def test_train_transformer_on_recordings_converges_on_day_scale_targets():
    """Reproduces a real bug found in final review: without target standardisation,
    the transformer's output head starts near 0 and cannot reach real GA targets
    (~266-301 days) within a normal training budget, so it collapses to predicting
    roughly the same value for every input regardless of the latent -- LOSOCV/
    zero-shot MAE was observed at ~243 days on real-scale targets. This pins that the
    model actually learns the signal present in the latent, not merely that it runs
    without error."""
    cfg = _with_epochs(load_config(), epochs=80)
    torch.manual_seed(0)
    ga = torch.tensor([266.0, 273.0, 280.0, 287.0, 294.0, 301.0])
    seq_len = 5
    z_by_recording = []
    for g in ga:
        signal = (g - ga.mean()).expand(seq_len, 1)
        noise = torch.randn(seq_len, cfg.cvae.latent_dim - 1) * 0.01
        z_by_recording.append(torch.cat([signal, noise], dim=1))

    model = train_transformer_on_recordings(cfg, z_by_recording, ga)
    from fcac.transformer.evaluate import predict_recordings

    preds = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)

    mae = (preds - ga).abs().mean().item()
    predict_mean_mae = (ga - ga.mean()).abs().mean().item()  # floor a collapsed model hits
    assert mae < predict_mean_mae, (
        f"MAE {mae:.1f} should beat the predict-the-mean floor {predict_mean_mae:.1f} "
        "if the model is using the latent signal rather than collapsing to a constant"
    )


def test_zero_shot_ninfea_reports_mae_and_ood_flags():
    cfg = _with_epochs(load_config(), epochs=2)
    # 20 windows/recording so the concatenated fit (N=100) exceeds the 64-dim latent
    # space -- fitting on 5 pre-averaged recording means (N=5 << D=64) is what the
    # final review found flags every in-distribution point as OOD.
    z_train = [torch.randn(20, cfg.cvae.latent_dim) for _ in range(5)]
    ga_train = torch.tensor([266.0, 273.0, 280.0, 287.0, 294.0])
    model = train_transformer_on_recordings(cfg, z_train, ga_train)
    ood_stats = fit_ood(torch.cat(z_train))

    z_zero_shot = [torch.randn(4, cfg.cvae.latent_dim) for _ in range(3)]
    ga_zero_shot = torch.tensor([147.0, 154.0, 161.0])  # 21-27 wks, CLAUDE.md §3

    report = zero_shot_ninfea(cfg, model, z_zero_shot, ga_zero_shot, ood_stats)
    assert len(report["predicted_ca_days"]) == 3
    assert "mae_days" in report and "mae_ci95" in report
    assert len(report["ood_flag"]) == 3


def test_zero_shot_ninfea_does_not_flag_in_distribution_recordings():
    """Final review: fitting/scoring OOD at the wrong granularity either flags
    everything or nothing. Exercises zero_shot_ninfea's real per-window-median
    aggregation (not just the underlying fit_ood/mahalanobis_distance primitives,
    already covered in tests/test_evaluation.py) -- a held-out recording drawn from
    the same distribution as training should not be flagged; a recording far outside
    it should. Uses an untrained model: ood_flag never depends on the model's
    predictions, so training one here would only slow the test down.

    Fits on N=1000 window-level latents (D=64): a covariance needs many more points
    than dimensions to be well-conditioned enough for a stable Mahalanobis distance
    -- N just above D, as fit_ood's minimum guard allows, is numerically valid but
    still statistically noisy (confirmed by hand: N=100 flagged an in-distribution
    point)."""
    from fcac.transformer.model import CardiacAgeTransformer

    torch.manual_seed(0)
    cfg = load_config()
    latent_dim = cfg.cvae.latent_dim
    z_train = torch.cat([torch.randn(200, latent_dim) for _ in range(5)])
    ood_stats = fit_ood(z_train, percentile=99.0)
    model = CardiacAgeTransformer(cfg)

    in_distribution = torch.randn(20, latent_dim)  # same distribution as z_train
    far_out = torch.full((20, latent_dim), 50.0)
    ga_zero_shot = torch.tensor([150.0, 155.0])

    report = zero_shot_ninfea(cfg, model, [in_distribution, far_out], ga_zero_shot, ood_stats)
    assert report["ood_flag"] == [False, True]


def test_predict_recordings_chunks_a_recording_longer_than_max_seq_len():
    """Final review flagged this as untested: a recording must be split into
    crop_len-sized chunks and averaged, not fed through the transformer whole (which
    would hit the ValueError CardiacAgeTransformer.forward raises above max_seq_len).
    Uses an untrained model -- only the chunking/averaging plumbing is under test."""
    from fcac.transformer.evaluate import predict_recordings
    from fcac.transformer.model import CardiacAgeTransformer

    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    long_recording = torch.randn(3 * cfg.transformer.max_seq_len + 7, cfg.cvae.latent_dim)

    predictions = predict_recordings(model, [long_recording], cfg.transformer.crop_len)
    assert predictions.shape == (1,)
    assert torch.isfinite(predictions).all()


def test_train_transformer_on_recordings_handles_batch_size_larger_than_dataset():
    """Final review flagged this as untested: with few recordings and
    repeats_per_epoch=1, RecordingDataset can be shorter than transformer.batch_size,
    and the DataLoader must not raise or silently train on zero batches. A single
    recording also exercises target_std's N=1 edge case (see train.py's
    `unbiased=False` comment) -- assert finite, not just non-None, so a NaN-poisoned
    model doesn't pass silently."""
    cfg = load_config()
    data = cfg.to_dict()
    data["transformer"]["epochs"] = 2
    data["transformer"]["repeats_per_epoch"] = 1  # dataset len == n_recordings == 1
    cfg = Config(data)
    assert cfg.transformer.batch_size > 1  # the case under test: batch_size > len(dataset)

    z_by_recording = [torch.randn(6, cfg.cvae.latent_dim)]
    ca_days_target = torch.tensor([280.0])

    model = train_transformer_on_recordings(cfg, z_by_recording, ca_days_target)
    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_losocv_refit_cvae_refits_inside_each_fold_and_scores_each_subject_once():
    from fcac.transformer.evaluate import losocv_refit_cvae

    data = _with_epochs(load_config(), epochs=1).to_dict()
    data["cvae"]["epochs"] = 1
    cfg = Config(data)
    gen = torch.Generator().manual_seed(0)
    records = []
    for i, ga in enumerate([266.0, 273.0, 280.0, 287.0]):
        x = torch.randn(6, cfg.cvae.input_dim, generator=gen).numpy()
        x[0, 3] = float("nan")  # imputation must cope with NaNs
        records.append({"x": x, "subject": f"s{i}", "ga_days": ga})

    results = losocv_refit_cvae(cfg, records)
    assert {r["held_out_subject"] for r in results} == {"s0", "s1", "s2", "s3"}
    for r in results:
        assert torch.isfinite(torch.tensor(r["predicted_ca_days"]))


def test_losocv_holds_out_every_recording_of_a_repeated_subject_together():
    from fcac.transformer.evaluate import losocv_refit_cvae

    data = _with_epochs(load_config(), epochs=1).to_dict()
    data["cvae"]["epochs"] = 1
    cfg = Config(data)
    gen = torch.Generator().manual_seed(0)
    layout = [("a", 266.0), ("a", 270.0), ("b", 280.0), ("c", 287.0), ("c", 289.0)]
    records = [{"x": torch.randn(6, cfg.cvae.input_dim, generator=gen).numpy(), "subject": s, "ga_days": g}
               for s, g in layout]

    results = losocv_refit_cvae(cfg, records)
    assert len(results) == 3  # one fold per subject
    by_subject = {r["held_out_subject"]: r for r in results}
    assert [x["index"] for x in by_subject["a"]["records"]] == [0, 1]
    assert [x["index"] for x in by_subject["c"]["records"]] == [3, 4]
    assert "predicted_ca_days" in by_subject["b"]  # single-record folds keep the scalar keys
    assert "predicted_ca_days" not in by_subject["a"]


def _tiny_cfg(device="cpu"):
    data = load_config().to_dict()
    data["transformer"]["epochs"] = 1
    data["cvae"]["epochs"] = 1
    data["transformer"]["device"] = device
    data["cvae"]["device"] = device
    return Config(data)


def _records(cfg, layout):
    gen = torch.Generator().manual_seed(0)
    return [{"x": torch.randn(6, cfg.cvae.input_dim, generator=gen).numpy(), "subject": s, "ga_days": g}
            for s, g in layout]


def test_parallel_folds_match_sequential_folds():
    from fcac.transformer.evaluate import losocv_refit_cvae

    cfg = _tiny_cfg()
    records = _records(cfg, [("a", 266.0), ("a", 270.0), ("b", 280.0), ("c", 287.0)])
    seq = losocv_refit_cvae(cfg, records, workers=1)
    par = losocv_refit_cvae(cfg, records, workers=2)
    assert [r["held_out_subject"] for r in seq] == [r["held_out_subject"] for r in par]
    for a, b in zip(seq, par):
        for ra, rb in zip(a["records"], b["records"]):
            assert ra["index"] == rb["index"]
            assert ra["predicted_ca_days"] == pytest.approx(rb["predicted_ca_days"], abs=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_training_returns_cpu_models_and_finite_predictions():
    from fcac.cvae.train import train_cvae
    from fcac.transformer.train import train_transformer_on_recordings

    cfg = _tiny_cfg(device="cuda")
    z = [torch.randn(5, cfg.cvae.latent_dim) for _ in range(4)]
    model = train_transformer_on_recordings(cfg, z, torch.tensor([266.0, 273.0, 280.0, 287.0]))
    assert next(model.parameters()).device.type == "cpu" and model.target_mean.device.type == "cpu"
    cvae = train_cvae(cfg, torch.randn(12, cfg.cvae.input_dim), np.array(["a", "a", "b"] * 4),
                      np.full(12, np.nan))
    assert next(cvae.parameters()).device.type == "cpu"
