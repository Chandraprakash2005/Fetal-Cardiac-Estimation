"""tests/plot_ninfea_check.py

Visual QC: NInFEA before vs. after preprocessing (CLAUDE.md §6 "verify preprocessing
... visually"). Not a pytest test (no test_ prefix) -- run manually:
    python tests/plot_ninfea_check.py
"""
import matplotlib.pyplot as plt

from fcac.config import load_config
from fcac.io.ninfea import load_ninfea_record
from fcac.preprocessing.channels import kurtosis_select
from fcac.preprocessing.filters import butter_bandpass, notch_filter, resample_to, trim_edges

cfg = load_config(dataset="ninfea")
recording = load_ninfea_record("1", cfg)

band = cfg.preprocessing.bandpass_presets[cfg.preprocessing.bandpass_preset]
filtered = butter_bandpass(
    recording.signal, recording.fs_raw, band.low_hz, band.high_hz, cfg.preprocessing.bandpass_order
)
if cfg.preprocessing.notch_enabled:
    filtered = notch_filter(
        filtered,
        recording.fs_raw,
        cfg.preprocessing.notch.freq_hz,
        cfg.preprocessing.notch.quality_factor,
    )
filtered = trim_edges(filtered, recording.fs_raw, cfg.preprocessing.edge_trim_seconds)
resampled = resample_to(filtered, recording.fs_raw, cfg.preprocessing.target_fs_hz)
best4, indices, _kurtosis_scores = kurtosis_select(
    resampled, cfg.preprocessing.kurtosis_selection.n_select
)

seconds_to_show = 5
raw_n = int(seconds_to_show * recording.fs_raw)
raw_time = [i / recording.fs_raw for i in range(raw_n)]
proc_fs = cfg.preprocessing.target_fs_hz
proc_n = int(seconds_to_show * proc_fs)
proc_time = [i / proc_fs for i in range(proc_n)]

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(raw_time, recording.signal[indices[0], :raw_n], color="tab:red")
axes[0].set_title(f"Before filtering (channel index {indices[0]})")
axes[0].set_ylabel("Amplitude (mV)")
axes[0].set_xlabel("Time (seconds)")

axes[1].plot(proc_time, best4[0, :proc_n], color="tab:blue")
axes[1].set_title("After bandpass + notch filter, edge-trimmed, resampled, kurtosis-selected")
axes[1].set_ylabel("Amplitude (mV)")
axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig("results/figures/ninfea_before_after_v2.png", dpi=150)
print("Saved to results/figures/ninfea_before_after_v2.png")
