"""tests/test_transformer.py

End-to-end tests on random tensors (CLAUDE.md §8 items 6/9).

Implements: CLAUDE.md §8 item 9.
"""

import torch

from fcac.config import Config, load_config
from fcac.transformer.infer import infer
from fcac.transformer.model import CardiacAgeTransformer
from fcac.transformer.train import train_transformer


def _with_epochs(cfg: Config, epochs: int) -> Config:
    data = cfg.to_dict()
    data["transformer"]["epochs"] = epochs
    return Config(data)


def test_transformer_forward_shape():
    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    z_seq = torch.randn(4, 8, cfg.cvae.latent_dim)
    ca = model(z_seq)
    assert ca.shape == (4,)


def test_transformer_rejects_inconsistent_head_config():
    cfg = load_config()
    data = cfg.to_dict()
    data["transformer"]["d_k"] = 999
    bad_cfg = Config(data)
    try:
        CardiacAgeTransformer(bad_cfg)
        assert False, "expected ValueError for inconsistent d_model/n_heads/d_k"
    except ValueError:
        pass


def test_train_transformer_runs_end_to_end_on_random_tensors():
    cfg = _with_epochs(load_config(), epochs=2)
    n, seq_len = 8, 4
    z_seq = torch.randn(n, seq_len, cfg.cvae.latent_dim)
    ca_days_target = torch.rand(n) * 7 + 280

    model = train_transformer(cfg, z_seq, ca_days_target)
    prediction = infer(model, z_seq)
    assert prediction.ca_days.shape == (n,)
    assert prediction.aggregated is False


def test_transformer_forward_with_padding_mask():
    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    z_seq = torch.randn(3, 10, cfg.cvae.latent_dim)
    pad_mask = torch.zeros(3, 10, dtype=torch.bool)
    pad_mask[:, 7:] = True  # last 3 positions are padding
    ca = model(z_seq, pad_mask)
    assert ca.shape == (3,)
    assert torch.isfinite(ca).all()


def test_transformer_rejects_sequence_longer_than_max_seq_len():
    cfg = load_config()
    model = CardiacAgeTransformer(cfg)
    z_seq = torch.randn(1, cfg.transformer.max_seq_len + 1, cfg.cvae.latent_dim)
    try:
        model(z_seq)
        assert False, "expected ValueError for sequence longer than max_seq_len"
    except ValueError:
        pass


def test_train_transformer_accepts_pad_mask():
    cfg = _with_epochs(load_config(), epochs=2)
    n, seq_len = 8, 4
    z_seq = torch.randn(n, seq_len, cfg.cvae.latent_dim)
    pad_mask = torch.zeros(n, seq_len, dtype=torch.bool)
    ca_days_target = torch.rand(n) * 7 + 280

    from fcac.transformer.train import train_transformer

    model = train_transformer(cfg, z_seq, ca_days_target, pad_mask)
    prediction = infer(model, z_seq)
    assert prediction.ca_days.shape == (n,)


def test_train_transformer_on_recordings_runs_end_to_end():
    from fcac.transformer.train import train_transformer_on_recordings

    cfg = _with_epochs(load_config(), epochs=2)
    z_by_recording = [torch.randn(6, cfg.cvae.latent_dim), torch.randn(3, cfg.cvae.latent_dim),
                       torch.randn(9, cfg.cvae.latent_dim)]
    ca_days_target = torch.tensor([280.0, 285.0, 279.0])

    model = train_transformer_on_recordings(cfg, z_by_recording, ca_days_target)
    from fcac.transformer.evaluate import predict_recordings

    predictions = predict_recordings(model, z_by_recording, cfg.transformer.crop_len)
    assert predictions.shape == (3,)
    assert torch.isfinite(predictions).all()
