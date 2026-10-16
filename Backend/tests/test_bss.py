"""tests/test_bss.py -- synthetic checks for fcac.bss (summed pulse trains, mixed)."""

import numpy as np

from fcac.bss import extract_fecg_bss, select_fetal_component, unmix_record
from fcac.config import load_config
from fcac.memd.diagnostics import pan_tompkins_bpm

FS = 1000.0


def _pulse_train(rate_bpm, n, rng, jitter=0.01):
    x = np.zeros(n)
    t, period = 0.2, 60.0 / rate_bpm
    while t < n / FS - 0.1:
        i = int(t * FS)
        k = np.arange(-30, 31)
        x[i - 30:i + 31] += np.exp(-0.5 * (k / 6.0) ** 2) * (1 + 0.05 * rng.normal())
        t += period * (1 + jitter * rng.normal())
    return x


def _mixture(n_windows=4, seed=0):
    rng = np.random.default_rng(seed)
    n = 5000 * n_windows
    maternal = _pulse_train(72, n, rng) * 5.0   # strong, slow
    fetal = _pulse_train(140, n, rng)           # weak, fast
    noise = rng.normal(size=(4, n)) * 0.05
    mixing = np.array([[1.0, 0.6], [0.8, 1.0], [0.5, 0.9], [1.0, 0.3]])
    x = mixing @ np.stack([maternal, fetal]) + noise
    windows = x.reshape(4, n_windows, 5000).transpose(1, 0, 2).astype(np.float32)
    return windows, fetal.reshape(n_windows, 5000)


def test_unmix_record_preserves_shape():
    windows, _ = _mixture()
    assert unmix_record(windows, load_config()).shape == windows.shape


def test_bss_recovers_the_fetal_component_and_rate():
    cfg = load_config()
    windows, fetal = _mixture()
    fecg, chosen = extract_fecg_bss(windows, FS, cfg)
    for w in range(len(fecg)):
        assert abs(np.corrcoef(fecg[w], fetal[w])[0, 1]) > 0.8
        assert abs(pan_tompkins_bpm(fecg[w], FS) - 140) < 8


def test_select_fetal_component_ignores_maternal_rate_component():
    cfg = load_config()
    rng = np.random.default_rng(1)
    comps = np.stack([_pulse_train(70, 5000, rng), _pulse_train(138, 5000, rng), rng.normal(size=5000)])
    idx, hr = select_fetal_component(comps, FS, cfg)
    assert idx == 1 and 120 < hr < 150
