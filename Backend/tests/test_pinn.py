"""tests/test_pinn.py

Tests for the PINN package: architecture wiring, the Laplace residual, the coordinate
scheme in pinn/coords.py, and an end-to-end train -> freeze -> extract pass on synthetic
windows.

These check that the machinery is correct and self-consistent. They deliberately do NOT
assert extraction quality: the coordinate scheme is an unconfirmed assumption (see the
header of src/fcac/pinn/coords.py), so a quality threshold here would encode a guess as
a requirement.

Implements: CLAUDE.md section 8 item 9.
"""

import numpy as np
import pytest
import torch

from fcac.config import Config, load_config
from fcac.pinn.coords import (
    electrode_positions,
    expected_input_dim,
    project_to_single_channel,
    readout,
    sample_collocation,
    window_to_coords,
)
from fcac.pinn.infer import extract_fecg, infer
from fcac.pinn.losses import hybrid_loss
from fcac.pinn.model import PINN
from fcac.pinn.physics import laplacian
from fcac.pinn.train import concat_fetal_band_windows, train_pinn, train_pinn_on_windows


def _with(cfg: Config, **pinn_overrides) -> Config:
    data = cfg.to_dict()
    data["pinn"].update(pinn_overrides)
    return Config(data)


def _synthetic_windows(n_windows=2, n_channels=4, n_samples=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 1000.0
    source = np.sin(2 * np.pi * 40 * t)
    gains = rng.uniform(0.5, 1.5, (n_windows, n_channels, 1))
    noise = 0.1 * rng.standard_normal((n_windows, n_channels, n_samples))
    return (gains * source + noise).astype(np.float32)


# --------------------------------------------------------------------------- model


def test_pinn_forward_shape():
    cfg = load_config()
    model = PINN(cfg)
    coords = torch.randn(16, cfg.pinn.input_dim)
    assert model(coords).shape == (16, cfg.pinn.output_dim)


def test_config_input_dim_matches_coordinate_scheme():
    """configs/default.yaml must agree with what pinn/coords.py actually builds."""
    cfg = load_config()
    assert cfg.pinn.input_dim == expected_input_dim(cfg)


def test_architecture_matches_locked_spec():
    """CLAUDE.md section 2: 5 layers x 512, Tanh 1-3, ReLU + Dropout(0.2) 4-5."""
    cfg = load_config()
    linears = [m for m in PINN(cfg).net if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [512, 512, 512, 512, 512, 1]

    activations = [m for m in PINN(cfg).net if isinstance(m, (torch.nn.Tanh, torch.nn.ReLU))]
    assert [type(m).__name__ for m in activations] == ["Tanh"] * 3 + ["ReLU"] * 2

    dropouts = [m for m in PINN(cfg).net if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 2 and all(d.p == 0.2 for d in dropouts)


# -------------------------------------------------------------------------- physics


def test_laplacian_of_linear_function_is_zero():
    coords = torch.randn(8, 3, requires_grad=True)
    v = coords.sum(dim=1, keepdim=True)  # linear -> Laplacian vanishes
    assert torch.allclose(laplacian(v, coords), torch.zeros(8, 1), atol=1e-5)


def test_laplacian_of_quadratic_is_correct():
    """For V = sum(x_i^2), the Laplacian is exactly 2*D."""
    coords = torch.randn(8, 3, requires_grad=True)
    v = (coords**2).sum(dim=1, keepdim=True)
    assert torch.allclose(laplacian(v, coords), torch.full((8, 1), 6.0), atol=1e-4)


def test_laplacian_dims_restricts_to_selected_columns():
    """Restricting to 2 of 3 columns must give 2*2=4, not 2*3=6 -- this is what keeps
    the time column out of the spatial Laplace residual."""
    coords = torch.randn(8, 3, requires_grad=True)
    v = (coords**2).sum(dim=1, keepdim=True)
    assert torch.allclose(laplacian(v, coords, dims=(0, 1)), torch.full((8, 1), 4.0), atol=1e-4)


def test_harmonic_function_has_zero_laplace_residual():
    """V = x^2 - y^2 is harmonic, so the physics loss should vanish on it."""
    from fcac.pinn.losses import physics_loss

    coords = torch.randn(16, 3, requires_grad=True)
    v = (coords[:, 0] ** 2 - coords[:, 1] ** 2).unsqueeze(1)
    assert physics_loss(v, coords, cfg=None).item() < 1e-8


# --------------------------------------------------------------------------- coords


def test_electrode_positions_shape_matches_channel_count():
    cfg = load_config()
    assert electrode_positions(cfg).shape == (cfg.preprocessing.n_channels, 3)


def test_window_to_coords_layout_is_channel_major():
    cfg = load_config()
    window = _synthetic_windows(n_windows=1, n_samples=32)[0]
    coords, target = window_to_coords(window, cfg)

    c, n = window.shape
    assert coords.shape == (c * n, cfg.pinn.input_dim)
    assert target.shape == (c * n, 1)
    # Channel-major: the first n rows must all sit at electrode 0.
    positions = electrode_positions(cfg)
    assert torch.allclose(coords[:n, :3], torch.from_numpy(positions[0]).expand(n, 3))
    # And the target must be that channel's samples, in order.
    assert torch.allclose(target[:n, 0], torch.from_numpy(window[0]))


def test_window_to_coords_carries_observation_context():
    """Columns 4: are the C-channel observation at each instant (see coords.py OPEN-Q 1)."""
    cfg = load_config()
    window = _synthetic_windows(n_windows=1, n_samples=32)[0]
    coords, _ = window_to_coords(window, cfg)
    n = window.shape[1]
    assert torch.allclose(coords[:n, 4:], torch.from_numpy(window.T))


def test_sample_collocation_stays_inside_the_inflated_box():
    cfg = load_config()
    window = _synthetic_windows(n_windows=1, n_samples=64)[0]
    points = sample_collocation(window, 128, cfg)

    assert points.shape == (128, cfg.pinn.input_dim)
    positions = electrode_positions(cfg)
    margin = cfg.pinn.collocation.box_margin
    spatial = points[:, :3].numpy()
    assert (spatial >= positions.min(axis=0) - margin - 1e-5).all()
    assert (spatial <= positions.max(axis=0) + margin + 1e-5).all()
    assert (points[:, 3] >= 0).all() and (points[:, 3] <= 1).all()


def test_project_to_single_channel_recovers_a_common_source():
    """4 channels carrying one scaled source must collapse back onto that source."""
    t = np.arange(500) / 1000.0
    source = np.sin(2 * np.pi * 30 * t)
    signal = np.outer(np.array([1.0, 0.6, -0.8, 0.4]), source).astype(np.float32)

    got = project_to_single_channel(signal)
    corr = np.corrcoef(got, source)[0, 1]
    assert abs(corr) > 0.99


def test_project_to_single_channel_sign_is_deterministic():
    """The largest deflection is forced positive, so flipping the input can't flip
    the output -- otherwise R peaks would invert at random between windows."""
    t = np.arange(500) / 1000.0
    source = np.sin(2 * np.pi * 30 * t)
    signal = np.outer(np.array([1.0, 0.6, -0.8, 0.4]), source).astype(np.float32)

    assert np.allclose(project_to_single_channel(signal), project_to_single_channel(-signal))


def test_readout_rejects_unknown_mode():
    cfg = _with(load_config(), readout="nonsense")
    window = _synthetic_windows(n_windows=1, n_samples=16)[0]
    with pytest.raises(ValueError, match="unknown pinn.readout"):
        readout(window, torch.zeros(window.size, 1), cfg)


def test_potential_readout_ignores_the_observation():
    cfg = _with(load_config(), readout="potential")
    window = _synthetic_windows(n_windows=1, n_samples=16)[0]
    v = torch.zeros(window.size, 1)
    # V is identically zero, so the potential readout is zero regardless of `window`.
    assert np.allclose(readout(window, v, cfg), 0.0)


# ---------------------------------------------------------------------------- loss


def test_hybrid_loss_is_finite():
    cfg = load_config()
    model = PINN(cfg)
    coords = torch.randn(10, cfg.pinn.input_dim, requires_grad=True)
    loss = hybrid_loss(model(coords), torch.randn(10, 1), coords, cfg)
    assert torch.isfinite(loss)


def test_hybrid_loss_accepts_separate_collocation_points():
    cfg = load_config()
    model = PINN(cfg)
    coords = torch.randn(10, cfg.pinn.input_dim, requires_grad=True)
    collocation = torch.randn(7, cfg.pinn.input_dim, requires_grad=True)
    loss = hybrid_loss(
        model(coords),
        torch.randn(10, 1),
        coords,
        cfg,
        v_collocation=model(collocation),
        collocation_coords=collocation,
    )
    assert torch.isfinite(loss)


def test_hybrid_loss_respects_locked_weights():
    """L = 0.7*L_data + 0.3*L_physics (CLAUDE.md section 2)."""
    from fcac.pinn.losses import data_loss, physics_loss

    cfg = load_config()
    model = PINN(cfg).eval()  # eval() so dropout doesn't randomise between calls
    coords = torch.randn(10, cfg.pinn.input_dim, requires_grad=True)
    target = torch.randn(10, 1)
    v = model(coords)

    expected = 0.7 * data_loss(v, target) + 0.3 * physics_loss(v, coords, cfg)
    assert torch.allclose(hybrid_loss(v, target, coords, cfg), expected, atol=1e-6)


# --------------------------------------------------------------------------- train


def test_resolve_device_auto_matches_cuda_availability():
    from fcac.pinn.train import _resolve_device

    resolved = _resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolved.type == expected


def test_resolve_device_explicit_cpu():
    from fcac.pinn.train import _resolve_device

    assert _resolve_device("cpu").type == "cpu"


def test_train_pinn_on_windows_respects_explicit_cpu_device():
    cfg = _with(load_config(), epochs=1, points_per_window=64, device="cpu")
    model = train_pinn_on_windows(cfg, _synthetic_windows(n_windows=2, n_samples=128))
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_train_pinn_on_windows_uses_cuda_when_available():
    """Real GPU training path -- skipped on machines without CUDA rather than faked,
    since a mocked device would not catch a real coords/collocation/model device
    mismatch (e.g. the pick-index CPU/CUDA mismatch this test caught during
    development)."""
    cfg = _with(load_config(), epochs=1, points_per_window=64, device="cuda")
    model = train_pinn_on_windows(cfg, _synthetic_windows(n_windows=2, n_samples=128))
    assert next(model.parameters()).device.type == "cuda"

    # infer()/extract_fecg() must also work against a CUDA-resident model.
    output = extract_fecg(model, _synthetic_windows(n_windows=2, n_samples=128), cfg)
    assert np.isfinite(output.fecg).all()


def test_train_pinn_runs_end_to_end_on_random_tensors():
    """cfg.pinn.device defaults to "auto", which resolves to CUDA on a machine with a
    GPU (this one does) -- so train_pinn's returned model may not live on CPU.
    Re-querying the model's actual device rather than assuming CPU is the same fix
    infer() needed in src/fcac/pinn/infer.py."""
    cfg = _with(load_config(), epochs=2)
    coords = torch.randn(32, cfg.pinn.input_dim)
    model = train_pinn(cfg, coords, torch.randn(32, 1))
    v = model(coords.to(next(model.parameters()).device))
    assert v.shape == (32, 1) and torch.isfinite(v).all()


def test_train_on_windows_then_extract_fecg():
    """The full MEMD-output -> PINN -> fECG contract (CLAUDE.md section 5)."""
    cfg = _with(load_config(), epochs=1, points_per_window=64)
    windows = _synthetic_windows(n_windows=2, n_samples=128)

    model = train_pinn_on_windows(cfg, windows)
    output = extract_fecg(model, windows, cfg)

    assert output.fecg.shape == (windows.shape[0], windows.shape[2])
    assert np.isfinite(output.fecg).all()


def test_train_on_windows_rejects_mismatched_input_dim():
    cfg = _with(load_config(), input_dim=4, epochs=1)
    with pytest.raises(ValueError, match="coordinate scheme"):
        train_pinn_on_windows(cfg, _synthetic_windows(n_samples=64))


def test_infer_does_not_build_a_graph():
    """Zero-shot inference must not leave anything for an optimizer to step on."""
    cfg = load_config()
    model = PINN(cfg)
    v = infer(model, torch.randn(8, cfg.pinn.input_dim))
    assert not v.requires_grad


def test_concat_fetal_band_windows():
    a, b = _synthetic_windows(n_windows=2), _synthetic_windows(n_windows=3, seed=1)
    assert concat_fetal_band_windows([a, b]).shape[0] == 5
