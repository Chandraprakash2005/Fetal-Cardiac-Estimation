"""tests/test_ninfea_doppler.py -- synthetic checks for fcac.io.ninfea_doppler."""

import numpy as np

from fcac.config import load_config
from fcac.io.ninfea_doppler import beat_times_from_envelope, doppler_envelope, median_otsu_threshold

FS = 284


def _image(duration_s=8.0, rate_bpm=135, height=400, axis=200, seed=0):
    """Black image, white zero-velocity line, periodic bright flow envelopes above the axis."""
    rng = np.random.default_rng(seed)
    width = int(duration_s * FS)
    img = np.zeros((height, width), dtype=np.uint8)
    period = 60.0 / rate_bpm
    centres = np.arange(0.3, duration_s - 0.2, period)
    for c in centres:
        col = int(c * FS)
        for dx in range(-12, 13):
            h = int(90 * np.exp(-0.5 * (dx / 6.0) ** 2)) + 5
            img[axis - h:axis - 2, col + dx] = 220
    img[axis, :] = 255
    img = np.clip(img + rng.integers(0, 6, img.shape), 0, 254).astype(np.uint8)
    img[axis, :] = 255
    return img, centres


def test_envelope_follows_the_lit_region_above_the_axis():
    img, centres = _image()
    x_up, x_down, axis = doppler_envelope(img)
    assert axis == 200 and x_up.shape == (img.shape[1],)
    peak_col = int(centres[2] * FS)
    assert 90 <= x_up[peak_col] <= 100
    assert x_up[:20].max() < 10          # quiet before the first beat
    assert np.all(x_down == 0)           # nothing below the axis


def test_threshold_is_a_valid_grey_level():
    img, _ = _image()
    t = median_otsu_threshold(img)
    assert 1 <= t <= 256


def test_beat_times_recover_rate():
    cfg = load_config(dataset="ninfea")
    img, centres = _image(duration_s=10.0, rate_bpm=135)
    x_up, x_down, _ = doppler_envelope(img)
    times, info = beat_times_from_envelope(x_up, x_down, cfg)
    assert abs(info["median_rate_bpm"] - 135) < 4
    assert len(times) == len(centres)
    assert np.abs(np.sort(times) - centres).max() < 0.05
