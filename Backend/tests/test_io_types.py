"""tests/test_io_types.py

Implements: CLAUDE.md §8 item 9 -- io/types.py data contracts (CLAUDE.md §5).
"""

import numpy as np

from fcac.io.types import (
    CAPrediction,
    FeatureVector,
    Latent,
    PreprocessedRecording,
    Recording,
)


def test_recording_shapes():
    rec = Recording(
        signal=np.zeros((4, 1000), dtype=np.float32),
        fs_raw=1000.0,
        fetal_ref=np.zeros(1000, dtype=np.float32),
        ga_days=280.0,
        subject_id="r01",
    )
    assert rec.signal.shape == (4, 1000)


def test_preprocessed_recording_shapes():
    pre = PreprocessedRecording(
        windows=np.zeros((3, 4, 5000), dtype=np.float32),
        fs=1000.0,
        ref_windows=np.zeros((3, 5000), dtype=np.float32),
        ga_days=280.0,
        subject_id="r01",
    )
    assert pre.windows.shape == (3, 4, 5000)
    assert pre.ref_windows.shape == (3, 5000)


def test_feature_vector_and_latent_shapes():
    fv = FeatureVector(f=np.zeros((3, 12), dtype=np.float32))
    latent = Latent(z=np.zeros((3, 64), dtype=np.float32))
    assert fv.f.shape[1] == 12
    assert latent.z.shape[1] == 64


def test_ca_prediction_default_not_aggregated():
    pred = CAPrediction(ca_days=np.zeros(3, dtype=np.float32))
    assert pred.aggregated is False
