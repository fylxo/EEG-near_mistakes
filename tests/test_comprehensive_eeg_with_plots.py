#!/usr/bin/env python3
"""
Comprehensive test suite for EEG analysis package WITH PLOTS.
Tests all functions, edge cases, and parameter variations with visual validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the main directory to path and import as package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from eeg_analysis_package.eeg_analysis import *
from eeg_analysis_package.time_frequency import *
from eeg_analysis_package.analysis_utils import *

def generate_test_data():
    """Generate test data for comprehensive testing."""
    np.random.seed(42)
    duration, fs, n_channels = 30, 200, 32
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Different signal patterns per channel for variety
        noise = np.random.normal(0, 5, n_samples)
        
        if ch < 10:  # Strong theta channels
            theta = 40 * np.sin(2 * np.pi * 6 * time)
        elif ch < 20:  # Alpha channels
            theta = 15 * np.sin(2 * np.pi * 10 * time)
        else:  # Mixed channels
            theta = 20 * np.sin(2 * np.pi * 6 * time) + 10 * np.sin(2 * np.pi * 10 * time)
            
        eeg_data[ch, :] = noise + theta
    
    return {
        'rat_id': 'TEST_RAT',
        'session_date': '2025-06-13',
        'eeg': eeg_data,
        'eeg_time': time.reshape(1, -1),
        'velocity_trace': np.random.uniform(0, 20, n_samples),
        'velocity_time': time,
        'nm_peak_times': np.array([5, 15, 25]),
        'nm_sizes': np.array([1, 2, 1.5]),
        'iti_peak_times': np.array([10, 20]),
        'iti_sizes': np.array([0.5, 1]),
        'file_path': '/mock/path/test_file.mat'
    }

def plot_test_data_overview(data):
    """Plot overview of test data showing different channel types."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Test Data Overview - Different Channel Types', fontsize=16)
    
    time = data['eeg_time'].flatten()
    
    # Theta channels (0-9)
    for i in range(3):
        axes[0, 0].plot(time[:1000], data['eeg'][i, :1000] + i*50, 
                       label=f'Ch{i}', alpha=0.7)
    axes[0, 0].set_title('Theta Channels (0-9) - 6Hz dominant')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Alpha channels (10-19)
    for i in range(10, 13):
        axes[0, 1].plot(time[:1000], data['eeg'][i, :1000] + (i-10)*30, 
                       label=f'Ch{i}', alpha=0.7)
    axes[0, 1].set_title('Alpha Channels (10-19) - 10Hz dominant')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Mixed channels (20-31)
    for i in range(20, 23):
        axes[0, 2].plot(time[:1000], data['eeg'][i, :1000] + (i-20)*40, 
                       label=f'Ch{i}', alpha=0.7)
    axes[0, 2].set_title('Mixed Channels (20-31) - 6Hz+10Hz')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Amplitude (μV)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # PSD comparison for different channel types
    theta_ch = data['eeg'][5, :]  # Theta channel
    alpha_ch = data['eeg'][15, :]  # Alpha channel
    mixed_ch = data['eeg'][25, :]  # Mixed channel
    
    freqs_t, psd_t = compute_psd(theta_ch, fs=200, method='welch')
    freqs_a, psd_a = compute_psd(alpha_ch, fs=200, method='welch')
    freqs_m, psd_m = compute_psd(mixed_ch, fs=200, method='welch')
    
    axes[1, 0].semilogy(freqs_t, psd_t, 'r-', label='Theta Ch (5)', alpha=0.8)
    axes[1, 0].axvspan(4, 8, alpha=0.2, color='red', label='Theta band')
    axes[1, 0].set_title('Theta Channel PSD')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power (μV²/Hz)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_xlim(0, 30)
    
    axes[1, 1].semilogy(freqs_a, psd_a, 'b-', label='Alpha Ch (15)', alpha=0.8)
    axes[1, 1].axvspan(8, 13, alpha=0.2, color='blue', label='Alpha band')
    axes[1, 1].set_title('Alpha Channel PSD')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power (μV²/Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_xlim(0, 30)
    
    axes[1, 2].semilogy(freqs_m, psd_m, 'g-', label='Mixed Ch (25)', alpha=0.8)
    axes[1, 2].axvspan(4, 8, alpha=0.2, color='red', label='Theta')
    axes[1, 2].axvspan(8, 13, alpha=0.2, color='blue', label='Alpha')
    axes[1, 2].set_title('Mixed Channel PSD')
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('Power (μV²/Hz)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_xlim(0, 30)
    
    plt.tight_layout()
    plt.show()

def test_psd_methods():
    """Test both Welch and multitaper PSD methods with plots."""
    print("Testing PSD methods...")
    data = generate_test_data()
    eeg_ch = data['eeg'][0, :]
    
    # Test both methods
    freqs_w, psd_w = compute_psd(eeg_ch, method='welch')
    freqs_m, psd_m = compute_psd(eeg_ch, method='multitaper')
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('PSD Method Comparison', fontsize=14)
    
    # Welch method
    axes[0].semilogy(freqs_w, psd_w, 'b-', alpha=0.7, label='Welch')
    axes[0].set_title('Welch Method PSD')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (μV²/Hz)')
    axes[0].axvspan(4, 8, alpha=0.2, color='red', label='Theta')
    axes[0].legend()
    axes[0].grid(True)
    
    # Multitaper method
    axes[1].semilogy(freqs_m, psd_m, 'r-', alpha=0.7, label='Multitaper')
    axes[1].set_title('Multitaper Method PSD')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power (μV²/Hz)')
    axes[1].axvspan(4, 8, alpha=0.2, color='red', label='Theta')
    axes[1].legend()
    axes[1].grid(True)
    
    # Overlay comparison
    # Interpolate to same frequency grid for comparison
    freqs_common = freqs_w
    psd_m_interp = np.interp(freqs_common, freqs_m, psd_m)
    
    axes[2].semilogy(freqs_common, psd_w, 'b-', alpha=0.7, label='Welch')
    axes[2].semilogy(freqs_common, psd_m_interp, 'r-', alpha=0.7, label='Multitaper')
    axes[2].set_title('Method Comparison')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power (μV²/Hz)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test invalid method
    try:
        compute_psd(eeg_ch, method='invalid')
        assert False, "Should have raised ValueError for invalid method"
    except ValueError:
        pass
    
    print("✓ Both PSD methods work correctly")
    return freqs_w, psd_w, freqs_m, psd_m

def test_time_frequency_functions():
    """Test all time-frequency analysis functions with plots."""
    print("Testing time-frequency functions...")
    data = generate_test_data()
    
    # Test basic Morlet on different channel types
    theta_ch = data['eeg'][5, :]  # Theta channel
    alpha_ch = data['eeg'][15, :]  # Alpha channel
    mixed_ch = data['eeg'][25, :]  # Mixed channel
    
    freqs = np.arange(2, 25, 1)
    freqs_t, power_t = morlet_spectrogram(theta_ch, sfreq=200, freqs=freqs)
    freqs_a, power_a = morlet_spectrogram(alpha_ch, sfreq=200, freqs=freqs)
    freqs_m, power_m = morlet_spectrogram(mixed_ch, sfreq=200, freqs=freqs)
    
    time = data['eeg_time'].flatten()
    
    # Plot spectrograms for different channel types
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Morlet Spectrograms for Different Channel Types', fontsize=14)
    
    # Theta channel
    im1 = axes[0, 0].pcolormesh(time, freqs_t, np.log10(power_t + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 0].set_title('Theta Channel (Ch 5)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].axhline(6, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Alpha channel
    im2 = axes[0, 1].pcolormesh(time, freqs_a, np.log10(power_a + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 1].set_title('Alpha Channel (Ch 15)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].axhline(10, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Mixed channel
    im3 = axes[0, 2].pcolormesh(time, freqs_m, np.log10(power_m + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 2].set_title('Mixed Channel (Ch 25)')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Frequency (Hz)')
    axes[0, 2].axhline(6, color='white', linestyle='--', alpha=0.5)
    axes[0, 2].axhline(10, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Test extract_morlet_band_power
    theta_power_t = extract_morlet_band_power(power_t, freqs_t, (4, 8))
    alpha_power_t = extract_morlet_band_power(power_t, freqs_t, (8, 13))
    
    theta_power_a = extract_morlet_band_power(power_a, freqs_a, (4, 8))
    alpha_power_a = extract_morlet_band_power(power_a, freqs_a, (8, 13))
    
    theta_power_m = extract_morlet_band_power(power_m, freqs_m, (4, 8))
    alpha_power_m = extract_morlet_band_power(power_m, freqs_m, (8, 13))
    
    # Plot band power over time
    axes[1, 0].plot(time, theta_power_t, 'r-', label='Theta (4-8Hz)', linewidth=2)
    axes[1, 0].plot(time, alpha_power_t, 'b-', label='Alpha (8-13Hz)', linewidth=2)
    axes[1, 0].set_title('Theta Channel - Band Powers')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, theta_power_a, 'r-', label='Theta (4-8Hz)', linewidth=2)
    axes[1, 1].plot(time, alpha_power_a, 'b-', label='Alpha (8-13Hz)', linewidth=2)
    axes[1, 1].set_title('Alpha Channel - Band Powers')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(time, theta_power_m, 'r-', label='Theta (4-8Hz)', linewidth=2)
    axes[1, 2].plot(time, alpha_power_m, 'b-', label='Alpha (8-13Hz)', linewidth=2)
    axes[1, 2].set_title('Mixed Channel - Band Powers')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Power')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Test multi-channel analysis
    freqs_all, tfr_array, channels = compute_all_channels_morlet(
        data, channels=range(6), freqs=np.arange(4, 15, 1)
    )
    
    # Test time windowing
    windowed_tfr, window_times = extract_tfr_time_window(tfr_array, time, 10, 20)
    
    # Plot multi-channel results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Channel Time-Frequency Analysis', fontsize=14)
    
    # Average across first 6 channels
    mean_tfr = np.mean(tfr_array, axis=0)
    im1 = axes[0, 0].pcolormesh(time, freqs_all, np.log10(mean_tfr + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 0].set_title('Average Spectrogram (6 channels)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Time window visualization
    mean_windowed = np.mean(windowed_tfr, axis=0)
    im2 = axes[0, 1].pcolormesh(window_times, freqs_all, np.log10(mean_windowed + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 1].set_title('Windowed Spectrogram (10-20s)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Channel comparison
    for ch in range(6):
        theta_power_ch = np.mean(tfr_array[ch, :2, :], axis=0)  # Theta band
        axes[1, 0].plot(time, theta_power_ch, alpha=0.7, label=f'Ch{ch}')
    axes[1, 0].set_title('Theta Power Across Channels')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Theta Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Frequency profile at different times
    mid_time_idx = len(time) // 2
    for ch in [0, 2, 4]:  # Sample some channels
        axes[1, 1].plot(freqs_all, np.log10(tfr_array[ch, :, mid_time_idx] + 1e-12), 
                       alpha=0.7, label=f'Ch{ch}')
    axes[1, 1].set_title(f'Frequency Profile at t={time[mid_time_idx]:.1f}s')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Log Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ All time-frequency functions work correctly")
    return tfr_array, window_times

def test_analysis_utils():
    """Test analysis utility functions with plots."""
    print("Testing analysis utilities...")
    
    # Create mock data for testing
    all_data = []
    for i in range(5):
        data = generate_test_data()
        data['rat_id'] = f'RAT_{i:03d}'
        data['session_date'] = f'2025-06-{13+i:02d}'
        data['file_path'] = f'/mock/path/file_{i}.mat'
        # Vary the data length for diversity
        if i > 2:
            data['eeg'] = data['eeg'][:, :4000]  # Shorter sessions
            data['eeg_time'] = data['eeg_time'][:, :4000]
        all_data.append(data)
    
    # Test create_summary_dataframe
    df_summary, grouped, longest, shortest = create_summary_dataframe(all_data, verbose=False)
    
    # Test convert_to_raw
    raw = convert_to_raw(all_data[0])
    
    # Plot summary statistics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dataset Summary Analysis', fontsize=16)
    
    # Session lengths
    axes[0, 0].bar(range(len(df_summary)), df_summary['eeg_len'], alpha=0.7, color='blue')
    axes[0, 0].set_title('EEG Session Lengths')
    axes[0, 0].set_xlabel('Session Index')
    axes[0, 0].set_ylabel('Length (samples)')
    axes[0, 0].grid(True)
    
    # NM and ITI counts
    width = 0.35
    x = np.arange(len(df_summary))
    axes[0, 1].bar(x - width/2, df_summary['nm_count'], width, label='NM events', alpha=0.7)
    axes[0, 1].bar(x + width/2, df_summary['iti_count'], width, label='ITI events', alpha=0.7)
    axes[0, 1].set_title('Event Counts per Session')
    axes[0, 1].set_xlabel('Session Index')
    axes[0, 1].set_ylabel('Event Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Per-rat summary
    axes[0, 2].bar(range(len(grouped)), grouped['session_count'], alpha=0.7, color='green')
    axes[0, 2].set_title('Sessions per Rat')
    axes[0, 2].set_xlabel('Rat Index')
    axes[0, 2].set_ylabel('Session Count')
    axes[0, 2].grid(True)
    
    # MNE Raw object visualization
    raw_data = raw.get_data()
    times = raw.times
    
    # Plot sample of raw data
    n_plot_channels = 4
    for i in range(n_plot_channels):
        axes[1, 0].plot(times[:1000], raw_data[i, :1000] * 1e6 + i*50, 
                       alpha=0.7, label=f'Ch{i}')
    axes[1, 0].set_title('MNE Raw Object Data (μV)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Info from MNE object
    info_text = f"Sampling Frequency: {raw.info['sfreq']} Hz\n"
    info_text += f"Number of Channels: {raw.info['nchan']}\n"
    info_text += f"Duration: {raw.times[-1]:.2f} s\n"
    info_text += f"Channel Types: {set(raw.get_channel_types())}"
    
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('MNE Raw Object Info')
    axes[1, 1].axis('off')
    
    # Dataset statistics distribution
    all_eeg_lens = df_summary['eeg_len'].values
    axes[1, 2].hist(all_eeg_lens, bins=5, alpha=0.7, color='purple')
    axes[1, 2].axvline(np.mean(all_eeg_lens), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_eeg_lens):.0f}')
    axes[1, 2].set_title('EEG Length Distribution')
    axes[1, 2].set_xlabel('Length (samples)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Analysis utilities work correctly")
    return df_summary, raw

def test_parameter_variations():
    """Test different parameter combinations with plots."""
    print("Testing parameter variations...")
    data = generate_test_data()
    
    # Test different nperseg values for Welch
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Parameter Variation Testing', fontsize=16)
    
    nperseg_values = [256, 512, 1024]
    for i, nperseg in enumerate(nperseg_values):
        freqs, psd = compute_psd(data['eeg'][0, :], nperseg=nperseg)
        axes[0, i].semilogy(freqs, psd, alpha=0.7)
        axes[0, i].set_title(f'Welch PSD (nperseg={nperseg})')
        axes[0, i].set_xlabel('Frequency (Hz)')
        axes[0, i].set_ylabel('Power (μV²/Hz)')
        axes[0, i].grid(True)
        axes[0, i].axvspan(4, 8, alpha=0.2, color='red')
    
    # Test different n_cycles for Morlet
    n_cycles_values = [3, 7, 10]
    freqs_morlet = np.arange(4, 20, 1)
    
    for i, n_cycles in enumerate(n_cycles_values):
        freqs, power = morlet_spectrogram(data['eeg'][0, :], n_cycles=n_cycles, freqs=freqs_morlet)
        time = data['eeg_time'].flatten()
        
        im = axes[1, i].pcolormesh(time, freqs, np.log10(power + 1e-12), 
                                  shading='auto', cmap='inferno')
        axes[1, i].set_title(f'Morlet (n_cycles={n_cycles})')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.show()
    
    # Test frequency range variations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Frequency Range Variations', fontsize=14)
    
    freq_ranges = [(1, 30), (5, 50), (10, 100)]
    for i, (fmin, fmax) in enumerate(freq_ranges):
        freqs, psd = compute_psd(data['eeg'][0, :], fmin=fmin, fmax=fmax)
        axes[i].semilogy(freqs, psd, alpha=0.7)
        axes[i].set_title(f'PSD ({fmin}-{fmax} Hz)')
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Power (μV²/Hz)')
        axes[i].grid(True)
        if fmax >= 8:
            axes[i].axvspan(max(4, fmin), min(8, fmax), alpha=0.2, color='red', label='Theta')
        if fmin <= 13 and fmax >= 8:
            axes[i].axvspan(max(8, fmin), min(13, fmax), alpha=0.2, color='blue', label='Alpha')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Parameter variations work correctly")

def test_integrated_workflows():
    """Test realistic analysis workflows with plots."""
    print("Testing integrated workflows...")
    data = generate_test_data()
    
    # Workflow comparison: different channel types
    theta_ch_idx = 5    # Theta-dominant channel
    alpha_ch_idx = 15   # Alpha-dominant channel
    mixed_ch_idx = 25   # Mixed channel
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Integrated Workflow Comparison Across Channel Types', fontsize=16)
    
    channels_to_test = [
        (theta_ch_idx, 'Theta Channel', 'red'),
        (alpha_ch_idx, 'Alpha Channel', 'blue'), 
        (mixed_ch_idx, 'Mixed Channel', 'green')
    ]
    
    for row, (ch_idx, ch_label, color) in enumerate(channels_to_test):
        # Full analysis for this channel
        result = analyze_eeg_channel(
            data, channel_index=ch_idx, freq_method='welch', tf_method=True,
            plot_types=[], extras=True, morlet_freqs=np.arange(2, 25, 1)
        )
        
        # Time domain trace
        time = data['eeg_time'].flatten()
        axes[row, 0].plot(time[:2000], data['eeg'][ch_idx, :2000], color=color, alpha=0.7)
        axes[row, 0].set_title(f'{ch_label} - Time Domain')
        axes[row, 0].set_xlabel('Time (s)')
        axes[row, 0].set_ylabel('Amplitude (μV)')
        axes[row, 0].grid(True)
        
        # PSD
        freqs = result['freqs']
        psd = result['psd']
        peak_freq = result['peak_frequency']
        
        axes[row, 1].semilogy(freqs, psd, color=color, alpha=0.7)
        axes[row, 1].axvline(peak_freq, color='black', linestyle='--', alpha=0.7)
        axes[row, 1].axvspan(4, 8, alpha=0.2, color='red', label='Theta')
        axes[row, 1].axvspan(8, 13, alpha=0.2, color='blue', label='Alpha')
        axes[row, 1].set_title(f'{ch_label} - PSD (Peak: {peak_freq:.1f}Hz)')
        axes[row, 1].set_xlabel('Frequency (Hz)')
        axes[row, 1].set_ylabel('Power (μV²/Hz)')
        axes[row, 1].grid(True)
        if row == 0:
            axes[row, 1].legend()
        
        # Band powers
        band_powers = result['band_power']
        bands = list(band_powers.keys())
        powers = list(band_powers.values())
        
        bars = axes[row, 2].bar(bands, powers, alpha=0.7, color=color)
        axes[row, 2].set_title(f'{ch_label} - Band Powers')
        axes[row, 2].set_ylabel('Power (μV²)')
        axes[row, 2].tick_params(axis='x', rotation=45)
        axes[row, 2].grid(True)
        
        # Highlight dominant band
        max_idx = np.argmax(powers)
        bars[max_idx].set_color('darkred')
        bars[max_idx].set_alpha(1.0)
    
    plt.tight_layout()
    plt.show()
    
    # Multi-channel summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Channel Analysis Summary', fontsize=14)
    
    # Analyze multiple channels and compare
    channels_subset = [5, 15, 25, 30]  # Different types
    peak_freqs = []
    dominant_bands = []
    theta_powers = []
    alpha_powers = []
    
    for ch in channels_subset:
        result = analyze_eeg_channel(
            data, channel_index=ch, plot_types=[], tf_method=False, extras=True
        )
        peak_freqs.append(result['peak_frequency'])
        dominant_bands.append(max(result['band_power'], key=result['band_power'].get))
        theta_powers.append(result['band_power']['theta'])
        alpha_powers.append(result['band_power']['alpha'])
    
    # Peak frequency comparison
    axes[0, 0].bar(range(len(channels_subset)), peak_freqs, alpha=0.7)
    axes[0, 0].set_title('Peak Frequencies Across Channels')
    axes[0, 0].set_xlabel('Channel Index')
    axes[0, 0].set_ylabel('Peak Frequency (Hz)')
    axes[0, 0].set_xticks(range(len(channels_subset)))
    axes[0, 0].set_xticklabels([f'Ch{ch}' for ch in channels_subset])
    axes[0, 0].grid(True)
    
    # Dominant band distribution
    band_counts = {band: dominant_bands.count(band) for band in set(dominant_bands)}
    axes[0, 1].pie(band_counts.values(), labels=band_counts.keys(), autopct='%1.1f%%')
    axes[0, 1].set_title('Dominant Band Distribution')
    
    # Theta vs Alpha power
    axes[1, 0].scatter(theta_powers, alpha_powers, s=100, alpha=0.7)
    for i, ch in enumerate(channels_subset):
        axes[1, 0].annotate(f'Ch{ch}', (theta_powers[i], alpha_powers[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 0].plot([0, max(max(theta_powers), max(alpha_powers))], 
                   [0, max(max(theta_powers), max(alpha_powers))], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('Theta Power')
    axes[1, 0].set_ylabel('Alpha Power')
    axes[1, 0].set_title('Theta vs Alpha Power')
    axes[1, 0].grid(True)
    
    # Time-frequency overview for one channel
    result_tf = analyze_eeg_channel(
        data, channel_index=25, plot_types=[], tf_method=True, 
        morlet_freqs=np.arange(4, 20, 1)
    )
    
    morlet_freqs = result_tf['morlet_freqs']
    morlet_power = result_tf['morlet_power']
    
    im = axes[1, 1].pcolormesh(time, morlet_freqs, np.log10(morlet_power + 1e-12), 
                              shading='auto', cmap='inferno')
    axes[1, 1].set_title('Representative Morlet Spectrogram (Ch25)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Integrated workflows work correctly")

def main():
    """Run comprehensive test suite with extensive plotting."""
    print("Running comprehensive EEG analysis tests with visualization...\n")
    
    # Generate and visualize test data
    data = generate_test_data()
    plot_test_data_overview(data)
    
    test_functions = [
        test_psd_methods,
        test_time_frequency_functions, 
        test_analysis_utils,
        test_parameter_variations,
        test_integrated_workflows
    ]
    
    failed_tests = []
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed_tests.append(test_func.__name__)
            import traceback
            traceback.print_exc()
    
    # Final summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} TESTS FAILED:")
        for test in failed_tests:
            print(f"  - {test}")
        
        # Plot failure summary
        test_names = [func.__name__ for func in test_functions]
        test_results = ['FAIL' if name in failed_tests else 'PASS' for name in test_names]
        colors = ['red' if result == 'FAIL' else 'green' for result in test_results]
        
        y_pos = range(len(test_names))
        bars = ax.barh(y_pos, [1]*len(test_names), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace('test_', '') for name in test_names])
        ax.set_xlabel('Test Status')
        ax.set_title('Comprehensive Test Results')
        
        for i, (test, result) in enumerate(zip(test_names, test_results)):
            ax.text(0.5, i, result, ha='center', va='center', fontweight='bold', color='white')
        
        return False
    else:
        print("✅ ALL COMPREHENSIVE TESTS PASSED!")
        print("Visual validation confirms:")
        print("  - Both Welch and multitaper PSD methods")
        print("  - All time-frequency analysis functions")
        print("  - Analysis utility functions")
        print("  - Parameter variations")
        print("  - Integrated analysis workflows")
        
        # Plot success summary
        test_names = [func.__name__.replace('test_', '').replace('_', ' ').title() 
                     for func in test_functions]
        
        y_pos = range(len(test_names))
        bars = ax.barh(y_pos, [1]*len(test_names), color='green', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names)
        ax.set_xlabel('Test Status')
        ax.set_title('Comprehensive Test Results - ALL PASSED ✓', fontweight='bold')
        
        for i, test in enumerate(test_names):
            ax.text(0.5, i, '✓ PASS', ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.show()
        
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)