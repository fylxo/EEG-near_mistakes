import numpy as np
import matplotlib.pyplot as plt
import mne

def morlet_spectrogram(eeg_ch: np.ndarray, sfreq: float = 200.0, freqs: np.ndarray = None, n_cycles: int = 7) -> tuple:
    """
    Compute Morlet time-frequency power. Returns freqs, power (freq x time).
    """
    if freqs is None:
        freqs = np.arange(1, 50, 1)
    data = eeg_ch[np.newaxis, np.newaxis, :]  # (n_epochs, n_channels, n_times)
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, output='power', zero_mean=True
    )
    return freqs, power[0, 0, :, :]  # (n_freqs, n_times)

def plot_morlet_spectrogram(times, freqs, power, log_power=True, title="Morlet Spectrogram", cmap='inferno'):
    """
    Plot Morlet time-frequency spectrogram.
    """
    plt.figure(figsize=(12, 5))
    if log_power:
        mesh = plt.pcolormesh(times, freqs, np.log10(power + 1e-12), shading='auto', cmap=cmap)
        plt.colorbar(mesh, label='Log Power (log₁₀ µV²)')
    else:
        mesh = plt.pcolormesh(times, freqs, power, shading='auto', cmap=cmap)
        plt.colorbar(mesh, label='Power (µV²)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def extract_morlet_band_power(power: np.ndarray, freqs: np.ndarray, band: tuple) -> np.ndarray:
    """
    Extract average Morlet power for a specific frequency band across time.
    """
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.mean(power[idx, :], axis=0)
    return band_power

def compute_all_channels_morlet(
    data,
    channels=None,
    sfreq=200,
    freqs=np.arange(4, 50, 1),
    n_cycles=7
):
    """
    Compute Morlet TFR for all (or subset) channels.
    Returns: freqs, tfr_array (channels, freqs, times), used_channels (indices)
    """
    
    if channels is None:
        channels = range(data['eeg'].shape[0])
    tfr_list = []
    for ch in channels:
        freqs_tfr, power = morlet_spectrogram(
            data['eeg'][ch, :], sfreq=sfreq, freqs=freqs, n_cycles=n_cycles
        )
        tfr_list.append(power)
    tfr_array = np.array(tfr_list)  # (n_channels, n_freqs, n_times)
    return freqs_tfr, tfr_array, list(channels)

def plot_mean_morlet(
    times, freqs, tfr_array, roi_indices=None, title="Mean Morlet Spectrogram", log_power=True
):
    """
    Plot mean Morlet spectrogram across channels (all or ROI).
    """
    if roi_indices is not None:
        tfr_array = tfr_array[roi_indices, :, :]
    mean_power = np.mean(tfr_array, axis=0)  # (n_freqs, n_times)
    plot_morlet_spectrogram(times, freqs, mean_power, log_power=log_power, title=title)

def extract_tfr_time_window(tfr_array, times, t_min, t_max):
    """
    Extract a window from the TFR.
    Returns: windowed_tfr (channels, freqs, window_times), window_times (array)
    """
    idx = np.logical_and(times >= t_min, times <= t_max)
    windowed_tfr = tfr_array[..., idx]
    window_times = times[idx]
    return windowed_tfr, window_times