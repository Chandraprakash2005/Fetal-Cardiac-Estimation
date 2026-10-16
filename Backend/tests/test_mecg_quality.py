"""tests/test_mecg_quality.py

Tests for maternal ECG cancellation (src/fcac/mecg.py) and signal-quality gating
(src/fcac/quality.py) -- the two stages added to stop the extraction chain failing
silently.

Synthetic maternal+fetal mixtures only, no PhysioNet data (CLAUDE.md section 6).

Implements: CLAUDE.md section 8 item 9.
"""

import numpy as np
import pytest

from fcac.config import Config, load_config
from fcac.mecg import (
    adaptive_cancel,
    adaptive_cancel_windows,
    maternal_template_subtract,
)
from fcac.memd.diagnostics import pan_tompkins_bpm
from fcac.pinn.coords import project_to_single_channel
from fcac.quality import assess_record, assess_window, filter_acceptable

FS = 1000.0
N = 5000


def _beats(bpm, width, amplitude, phase=0.0, n=N, fs=FS):
    t = np.arange(n) / fs
    signal = np.zeros(n)
    for centre in np.arange(phase, n / fs, 60.0 / bpm):
        u = (t - centre) / width
        signal += amplitude * np.exp(-0.5 * u**2) * (1 - u**2)
    return signal


def _mixture(maternal_gain=1.0, fetal_gain=0.15, seed=0):
    """(4, N) abdominal window: maternal at 80 bpm, fetal at 140 bpm."""
    rng = np.random.default_rng(seed)
    maternal = _beats(80, 0.012, 1.0)
    fetal = _beats(140, 0.006, 1.0, phase=0.13)
    gm = maternal_gain * np.array([[1.0], [0.95], [1.05], [1.0]])
    gf = fetal_gain * np.array([[0.2], [1.0], [0.15], [0.5]])
    x = gm @ maternal[None, :] + gf @ fetal[None, :]
    return (x + 0.02 * rng.standard_normal((4, N))).astype(np.float32)


def _cfg(**quality_overrides) -> Config:
    cfg = load_config()
    if not quality_overrides:
        return cfg
    data = cfg.to_dict()
    data["quality"].update(quality_overrides)
    return Config(data)


# --------------------------------------------------------------------------- mecg


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_template_subtraction_shifts_the_balance_from_maternal_to_fetal():
    """The function's actual contract: suppress maternal, relatively favour fetal.

    Note it does NOT promise to flip the detected rate to fetal on a single window. A
    5 s window holds only ~7 maternal beats, so the median template is estimated from 7
    samples and cancellation is partial (measured here: maternal correlation 0.98 -> 0.45,
    fetal 0.12 -> 0.23). That is why `adaptive_cancel` is paired with the quality gate
    rather than trusted to rescue every window, and why a record-level template built
    from all ~400 beats would be the next improvement.
    """
    cfg = _cfg()
    maternal = _beats(80, 0.012, 1.0)
    fetal = _beats(140, 0.006, 1.0, phase=0.13)
    x = _mixture(maternal_gain=1.0, fetal_gain=0.15)
    y = maternal_template_subtract(x, FS, cfg)

    maternal_before, maternal_after = _corr(x[1], maternal), _corr(y[1], maternal)
    fetal_before, fetal_after = _corr(x[1], fetal), _corr(y[1], fetal)

    assert maternal_after < maternal_before - 0.2, "maternal not suppressed"
    assert fetal_after > fetal_before, "fetal not relatively enhanced"
    assert abs(fetal_after / maternal_after) > abs(fetal_before / maternal_before)


def test_template_subtraction_preserves_shape_and_finiteness():
    cfg = _cfg()
    x = _mixture()
    out = maternal_template_subtract(x, FS, cfg)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_template_subtraction_passes_through_when_too_few_beats():
    """A window with no detectable beats must come back untouched, not mangled."""
    cfg = _cfg()
    flat = np.zeros((4, N), dtype=np.float32)
    assert np.allclose(maternal_template_subtract(flat, FS, cfg), flat)


def test_adaptive_cancel_leaves_a_clean_fetal_window_alone():
    """The measured failure mode of always-on cancellation: it damages clean windows."""
    cfg = _cfg()
    fetal_dominant = np.stack([_beats(140, 0.006, 1.0, phase=0.13)] * 4).astype(np.float32)
    fetal_dominant += 0.02 * np.random.default_rng(0).standard_normal((4, N))

    out, applied = adaptive_cancel(fetal_dominant, FS, cfg)
    assert not applied, "cancellation fired on an already-fetal window"
    assert np.allclose(out, fetal_dominant)


def test_adaptive_cancel_fires_on_a_maternal_locked_window():
    cfg = _cfg()
    _, applied = adaptive_cancel(_mixture(maternal_gain=1.0, fetal_gain=0.10), FS, cfg)
    assert applied


def test_adaptive_cancel_modes():
    x = _mixture()
    base = load_config().to_dict()

    base["mecg_cancellation"]["mode"] = "never"
    out, applied = adaptive_cancel(x, FS, Config(base))
    assert not applied and np.allclose(out, x)

    base["mecg_cancellation"]["mode"] = "always"
    _, applied = adaptive_cancel(x, FS, Config(base))
    assert applied

    base["mecg_cancellation"]["mode"] = "nonsense"
    with pytest.raises(ValueError, match="unknown mecg_cancellation.mode"):
        adaptive_cancel(x, FS, Config(base))


def test_adaptive_cancel_windows_reports_which_fired():
    cfg = _cfg()
    windows = np.stack([_mixture(seed=i) for i in range(3)])
    out, applied = adaptive_cancel_windows(windows, FS, cfg)
    assert out.shape == windows.shape
    assert applied.shape == (3,) and applied.dtype == bool


# ------------------------------------------------------------------------ quality


def test_quality_rejects_a_maternal_locked_window():
    """The core gate: an 80 bpm output is the mother, not the fetus."""
    cfg = _cfg()
    report = assess_window(_beats(80, 0.012, 1.0), FS, cfg)
    assert not report.acceptable
    assert any("maternal" in r for r in report.reasons)


def test_quality_accepts_a_clean_fetal_window():
    cfg = _cfg()
    report = assess_window(_beats(140, 0.006, 1.0), FS, cfg)
    assert report.acceptable, report.reasons
    assert 110 <= report.heart_rate_bpm <= 160


def test_quality_rejects_a_silent_window():
    cfg = _cfg()
    report = assess_window(np.zeros(N, dtype=np.float32), FS, cfg)
    assert not report.acceptable


def test_assess_record_flags_an_unusable_record():
    cfg = _cfg()
    maternal = np.stack([_beats(80, 0.012, 1.0)] * 10).astype(np.float32)
    summary = assess_record(maternal, FS, cfg)
    assert not summary["usable"]
    assert summary["n_acceptable"] == 0


def test_assess_record_accepts_a_good_record():
    cfg = _cfg()
    fetal = np.stack([_beats(140, 0.006, 1.0, phase=0.01 * i) for i in range(10)]).astype(np.float32)
    summary = assess_record(fetal, FS, cfg)
    assert summary["usable"]
    assert summary["acceptable_fraction"] > 0.9


def test_filter_acceptable_drops_only_the_bad_windows():
    cfg = _cfg()
    good = [_beats(140, 0.006, 1.0, phase=0.01 * i) for i in range(6)]
    bad = [_beats(80, 0.012, 1.0) for _ in range(4)]
    windows = np.stack(good + bad).astype(np.float32)

    kept = filter_acceptable(windows, FS, cfg)
    assert kept.shape[0] == 6
    for w in range(kept.shape[0]):
        assert 110 <= pan_tompkins_bpm(kept[w], FS) <= 160


def test_quality_thresholds_come_from_config():
    """Widening the band must let a previously-rejected window through."""
    strict = assess_window(_beats(95, 0.010, 1.0), FS, _cfg())
    lenient = assess_window(_beats(95, 0.010, 1.0), FS, _cfg(fetal_hr_min_bpm=80.0))
    assert not strict.acceptable
    assert lenient.acceptable, lenient.reasons
