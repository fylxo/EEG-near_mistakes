import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
n_trials = 10
n_freqs = 15
n_times = 200
sfreq = 100  # Hz
times = np.linspace(-1, 1, n_times)

# --- Simulate power data: trials × freqs × times ---
np.random.seed(42)
power = np.random.normal(1, 0.1, (n_trials, n_freqs, n_times))

# Add a strong burst in some trials (only trial 5–9)
burst = np.exp(-((times - 0.1)**2) / (2 * 0.02**2))  # burst shape
burst_freq = 7

for trial in range(5, 10):  # only second half
    power[trial, burst_freq, :] += 3 * burst

# --- Global z-score: mean & std across entire time axis (per trial) ---
global_z = np.zeros_like(power)
for trial in range(n_trials):
    mu  = power[trial].mean(axis=1, keepdims=True)
    std = power[trial].std(axis=1, keepdims=True)
    global_z[trial] = (power[trial] - mu) / (std + 1e-12)

# --- Baseline z-score: mean & std from –0.6 to –0.2 s (per trial) ---
baseline_mask = (times >= -0.6) & (times <= -0.2)
baseline_z = np.zeros_like(power)
for trial in range(n_trials):
    base = power[trial][:, baseline_mask]  # Correctly mask the time axis
    mu  = base.mean(axis=1, keepdims=True)
    std = base.std(axis=1, keepdims=True)
    baseline_z[trial] = (power[trial] - mu) / (std + 1e-12)

# --- Average across trials ---
avg_raw = power.mean(axis=0)
avg_global_z = global_z.mean(axis=0)
avg_baseline_z = baseline_z.mean(axis=0)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# Raw average
im0 = axs[0].imshow(avg_raw, aspect='auto', extent=[times[0], times[-1], 0, n_freqs],
                    origin='lower', cmap='viridis')
axs[0].set_title("Avg Raw Power (some trials have burst)")
axs[0].axvline(0, color='w', linestyle='--')
fig.colorbar(im0, ax=axs[0])

# Global z-score average
im1 = axs[1].imshow(avg_global_z, aspect='auto', extent=[times[0], times[-1], 0, n_freqs],
                    origin='lower', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
axs[1].set_title("Avg Global Z-Score")
axs[1].axvline(0, color='k', linestyle='--')
fig.colorbar(im1, ax=axs[1])

# Baseline z-score average
im2 = axs[2].imshow(avg_baseline_z, aspect='auto', extent=[times[0], times[-1], 0, n_freqs],
                    origin='lower', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
axs[2].set_title("Avg Baseline Z-Score (–0.6 to –0.2s)")
axs[2].axvline(0, color='k', linestyle='--')
fig.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Index")

plt.tight_layout()
plt.show()
