import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

def plot_eeg(time, eeg_ch, title="EEG Trace", start=0, fs=200, duration=5):
    """
    Plot a segment of an EEG trace.
    """
    start_idx = int(start * fs)
    end_idx = int((start + duration) * fs)

    plt.figure(figsize=(12, 4))
    plt.plot(time[start_idx:end_idx], eeg_ch[start_idx:end_idx])
    plt.title(f"{title} (t = {start}-{start+duration}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def eeg_stats(eeg_ch, fs=200.0):
    """
    Compute basic statistics for an EEG channel.
    """
    return {
        'mean': float(np.mean(eeg_ch)),
        'std': float(np.std(eeg_ch)),
        'min': float(np.min(eeg_ch)),
        'max': float(np.max(eeg_ch)),
        'duration_s': len(eeg_ch) / fs
    }

def analyze_eeg_time_domain_channel(data, channel_index=0, plot=True, title_suffix=""):
    eeg = data['eeg']
    time = data['eeg_time'].flatten()

    eeg_ch = eeg[channel_index, :]

    stats = eeg_stats(eeg_ch)

    if plot:
        plot_eeg(time, eeg_ch, title=f"EEG Channel {channel_index} {title_suffix}")

    return stats

def analyze_all_channels_time_domain(data, channels=None, plot=False):
    """
    Run time-domain stats (mean, std, etc.) for all or a subset of channels.
    Returns a dict or DataFrame of results.
    """
    if channels is None:
        channels = range(data['eeg'].shape[0])
    results = []
    for ch in channels:
        stats = eeg_stats(data['eeg'][ch, :])
        stats['channel'] = ch
        results.append(stats)
        if plot:
            plot_eeg(data['eeg_time'].flatten(), data['eeg'][ch, :], title=f"Ch{ch}")
    return results  # Or convert to DataFrame if you prefer

def compute_psd(eeg_ch: np.ndarray, fs: float = 200.0, method: str = 'welch', nperseg: int = 1024, fmin: float = 1.0, fmax: float = 100.0) -> tuple:
    """
    Compute Power Spectral Density (PSD) using Welch or Multitaper method.
    """
    if method == 'welch':
        freqs, psd = welch(eeg_ch, fs=fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return freqs[idx], psd[idx]
    elif method == 'multitaper':
        psd, freqs = mne.time_frequency.psd_array_multitaper(eeg_ch, sfreq=fs, fmin=fmin, fmax=fmax, adaptive=True, normalization='full')
        return freqs, psd
    else:
        raise ValueError("Method must be 'welch' or 'multitaper'.")

def plot_psd(freqs: np.ndarray, psd: np.ndarray, title: str = "PSD", kind: str = "semilogy") -> None:
    """
    Plot the power spectral density.
    kind: 'semilogy', 'loglog', or 'flattened'
    """
    plt.figure(figsize=(10, 4))
    if kind == "semilogy":
        plt.semilogy(freqs, psd)
    elif kind == "loglog":
        plt.loglog(freqs, psd)
    elif kind == "flattened":
        plt.plot(freqs, psd * freqs)
        plt.ylabel("Flattened Power (µV²/Hz × Hz)")
    else:
        plt.plot(freqs, psd)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    if kind != "flattened":
        plt.ylabel("Power Spectral Density (µV²/Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_band_power(freqs: np.ndarray, psd: np.ndarray, bands: dict = None) -> dict:
    """
    Compute band power for specified frequency bands.
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = float(np.trapz(psd[idx], freqs[idx]))
    return band_powers

def find_peak_frequency(freqs: np.ndarray, psd: np.ndarray, fmin: float = 1, fmax: float = 45) -> tuple:
    """
    Find the peak frequency and its power in a given range.
    """
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if np.any(idx):
        peak_idx = np.argmax(psd[idx])
        return float(freqs[idx][peak_idx]), float(psd[idx][peak_idx])
    else:
        return None, None

def analyze_eeg_channel(
    data: dict,
    channel_index: int = 0,
    fs: float = 200.0,
    time_range: tuple = None,
    freq_method: str = 'welch',      # 'welch' or 'multitaper'
    tf_method: bool = False,         # Do Morlet yes/no
    morlet_freqs: np.ndarray = None,
    n_cycles: int = 7,
    plot_types: list = None,         # ['semilogy', 'loglog', 'flattened']
    bands: dict = None,
    extras: bool = False
) -> dict:
    """
    High-level analysis function for a single EEG channel.
    """
    eeg = data['eeg'][channel_index, :]
    time = data['eeg_time'].flatten()

    if time_range is not None:
        t0, t1 = time_range
        idx = np.logical_and(time >= t0, time <= t1)
        eeg = eeg[idx]
        time = time[idx]

    stats = eeg_stats(eeg)
    output = {'stats': stats}

    # Frequency analysis
    freqs, psd = compute_psd(eeg, fs=fs, method=freq_method)
    if plot_types is None:
        plot_types = ['semilogy']
    for kind in plot_types:
        plot_psd(freqs, psd, title=f"PSD - Channel {channel_index} ({kind})", kind=kind)

    output.update({'freqs': freqs, 'psd': psd})

    if bands is not None or extras:
        band_powers = compute_band_power(freqs, psd, bands)
        output['band_power'] = band_powers

    if extras:
        peak_freq, peak_power = find_peak_frequency(freqs, psd)
        output['peak_frequency'] = peak_freq
        output['peak_power'] = peak_power

    # Time-frequency analysis
    if tf_method:
        freqs_tf, power = morlet_spectrogram(eeg, sfreq=fs, freqs=morlet_freqs, n_cycles=n_cycles)
        plot_morlet_spectrogram(time, freqs_tf, power, title=f"Morlet Spectrogram - Channel {channel_index}")
        output['morlet_freqs'] = freqs_tf
        output['morlet_power'] = power

    return output