"""tests/plot_adfecgdb_check.py

Visual QC: ADFECGDB raw abdominal vs. scalp channel (CLAUDE.md §6 "verify
preprocessing ... visually"). Not a pytest test (no test_ prefix) -- run manually:
    python tests/plot_adfecgdb_check.py
"""
import matplotlib.pyplot as plt

from fcac.config import load_config
from fcac.io.adfecgdb import load_adfecgdb_record

cfg = load_config(dataset="adfecgdb")
recording = load_adfecgdb_record("r01", cfg)
abdominal = recording.signal  # (4, N)
scalp = recording.fetal_ref  # (N,)
fs = recording.fs_raw

seconds_to_show = 5
n_samples = int(seconds_to_show * fs)
time_axis = [i / fs for i in range(n_samples)]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(time_axis, abdominal[0, :n_samples], color="tab:red")
axes[0].set_title("Abdominal channel (model input, maternal-dominated)")
axes[0].set_ylabel("Amplitude (mV)")

axes[1].plot(time_axis, scalp[:n_samples], color="tab:green")
axes[1].set_title("Scalp channel (Direct_1, true fetal ECG, ground truth only)")
axes[1].set_ylabel("Amplitude (mV)")
axes[1].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig("results/figures/adfecgdb_abdominal_vs_scalp.png", dpi=150)
print("Saved to results/figures/adfecgdb_abdominal_vs_scalp.png")
