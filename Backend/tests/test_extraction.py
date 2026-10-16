"""tests/test_extraction.py -- synthetic checks for fcac.extraction (switchable extractor)."""

import numpy as np
import pytest
import torch

from fcac.bss import extract_fecg_bss
from fcac.config import load_config
from fcac.extraction import extract_with_method
from fcac.pinn.model import PINN

FS = 1000.0


def _windows(n_windows=2, seed=0):
    rng = np.random.default_rng(seed)
    n = 5000
    t = np.arange(n) / FS
    maternal = np.sin(2 * np.pi * 1.2 * t) ** 21
    fetal = np.sin(2 * np.pi * 2.3 * t) ** 21
    mix = np.array([[1.0, 0.5], [0.7, 1.0], [0.4, 0.8], [1.0, 0.2]])
    w = np.stack([(mix @ np.stack([maternal, 0.3 * fetal])) + rng.normal(0, 0.02, (4, n))
                  for _ in range(n_windows)]).astype(np.float32)
    return w


def _model(cfg):
    torch.manual_seed(0)
    return PINN(cfg).eval()


def test_pinn_method_marks_every_window_as_from_pinn():
    cfg = load_config(dataset="adfecgdb")
    w = _windows()
    fecg, from_pinn, cancelled = extract_with_method(w, w, FS, cfg, _model(cfg), method="pinn")
    assert fecg.shape == (2, 5000) and from_pinn.all() and cancelled.shape == (2,)


def test_hybrid_takes_pinn_windows_that_pass_the_gate_and_bss_for_the_rest():
    cfg = load_config(dataset="adfecgdb")
    w = _windows()
    model = _model(cfg)
    pinn_fecg, _, _ = extract_with_method(w, w, FS, cfg, model, method="pinn")
    fecg, from_pinn, _ = extract_with_method(w, w, FS, cfg, model, method="hybrid")
    bss_fecg, _ = extract_fecg_bss(w, FS, cfg)
    for i in range(len(fecg)):
        expected = pinn_fecg[i] if from_pinn[i] else bss_fecg[i]
        assert np.allclose(fecg[i], expected, atol=1e-5)


def test_unknown_method_and_mismatched_windows_are_rejected():
    cfg = load_config(dataset="adfecgdb")
    w = _windows()
    with pytest.raises(ValueError):
        extract_with_method(w, w, FS, cfg, _model(cfg), method="ica")
    with pytest.raises(ValueError):
        extract_with_method(w, w[:1], FS, cfg, _model(cfg), method="hybrid")
