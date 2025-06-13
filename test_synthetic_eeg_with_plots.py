#!/usr/bin/env python3
"""
Test script for EEG analysis package using synthetic data WITH PLOTS.
Generates synthetic EEG data with known oscillatory patterns and validates
that our analysis functions produce expected results with visual verification.
"""

import numpy as np
import matplotlib.pyplot as plt
from eeg_analysis_package import *

def generate_synthetic_eeg(duration=60, fs=200, n_channels=32, seed=42):
    """
    Generate synthetic EEG data with known oscillatory patterns.
    
    Parameters:
    - duration: length in seconds
    - fs: sampling frequency 
    - n_channels: number of channels
    - seed: random seed for reproducibility
    
    Returns dict matching the expected data structure.
    """
    np.random.seed(seed)
    
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    # Create synthetic EEG with specific patterns
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Base noise
        noise = np.random.normal(0, 5, n_samples)
        
        # Strong theta bursts (6 Hz) - our main signal to detect
        theta_freq = 6.0
        theta_amplitude = 50  # Increased amplitude to ensure dominance
        
        # Create theta bursts at specific intervals
        burst_times = np.arange(5, duration-5, 10)  # Every 10 seconds
        theta_signal = np.zeros(n_samples)
        
        for burst_start in burst_times:
            burst_start_idx = int(burst_start * fs)
            burst_end_idx = int((burst_start + 2) * fs)  # 2-second bursts
            
            # Gaussian envelope for burst
            burst_len = burst_end_idx - burst_start_idx
            envelope = np.exp(-0.5 * ((np.arange(burst_len) - burst_len/2) / (burst_len/6))**2)
            
            # Theta oscillation with envelope
            t_burst = np.arange(burst_len) / fs
            theta_burst = theta_amplitude * envelope * np.sin(2 * np.pi * theta_freq * t_burst)
            theta_signal[burst_start_idx:burst_end_idx] += theta_burst
        
        # Add some alpha activity (10 Hz) - weaker
        alpha_signal = 4 * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
        
        # Add some beta activity (20 Hz) - even weaker
        beta_signal = 2 * np.sin(2 * np.pi * 20 * time + np.random.uniform(0, 2*np.pi))
        
        # Combine all components
        eeg_data[ch, :] = noise + theta_signal + alpha_signal + beta_signal
    
    # Create data structure matching the expected format
    synthetic_data = {
        'rat_id': 'SYNTHETIC_001',
        'session_date': '2025-06-13',
        'eeg': eeg_data,
        'eeg_time': time.reshape(1, -1),
        'velocity_trace': np.random.uniform(0, 20, n_samples),
        'velocity_time': time,
        'nm_peak_times': np.array([10, 20, 30, 40, 50]),  # Mock data
        'nm_sizes': np.array([1, 2, 1.5, 3, 2]),
        'iti_peak_times': np.array([15, 25, 35, 45]),
        'iti_sizes': np.array([0.5, 1, 0.8, 1.2])
    }
    
    return synthetic_data

def plot_data_overview(data):
    """Plot overview of synthetic data structure."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Synthetic EEG Data Overview', fontsize=16)
    
    time = data['eeg_time'].flatten()
    
    # Raw EEG traces - first 4 channels
    for i in range(4):
        axes[0, 0].plot(time[:2000], data['eeg'][i, :2000] + i*100, 
                       label=f'Ch{i}', alpha=0.7)
    axes[0, 0].set_title('Raw EEG Traces (first 10s)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Theta burst visualization
    axes[0, 1].plot(time[:4000], data['eeg'][0, :4000])
    axes[0, 1].axvspan(5, 7, alpha=0.3, color='red', label='Expected theta burst')
    axes[0, 1].axvspan(15, 17, alpha=0.3, color='red')
    axes[0, 1].set_title('Theta Bursts (Channel 0, first 20s)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Channel statistics
    means = [np.mean(data['eeg'][ch, :]) for ch in range(32)]
    stds = [np.std(data['eeg'][ch, :]) for ch in range(32)]
    
    axes[1, 0].bar(range(32), means, alpha=0.7)
    axes[1, 0].set_title('Mean Amplitude per Channel')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Mean (μV)')
    axes[1, 0].grid(True)
    
    axes[1, 1].bar(range(32), stds, alpha=0.7, color='orange')
    axes[1, 1].set_title('Standard Deviation per Channel')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Std (μV)')
    axes[1, 1].grid(True)
    
    # Velocity and event data
    axes[2, 0].plot(data['velocity_time'][:2000], data['velocity_trace'][:2000])
    axes[2, 0].set_title('Velocity Trace (first 10s)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Velocity')
    axes[2, 0].grid(True)
    
    # Event timing
    axes[2, 1].scatter(data['nm_peak_times'], data['nm_sizes'], 
                      color='red', s=100, alpha=0.7, label='NM events')
    axes[2, 1].scatter(data['iti_peak_times'], data['iti_sizes'], 
                      color='blue', s=100, alpha=0.7, label='ITI events')
    axes[2, 1].set_title('Event Timing')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Event Size')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def test_time_domain_functions(data):
    """Test time-domain analysis functions with plots."""
    print("=" * 50)
    print("TESTING TIME-DOMAIN FUNCTIONS")
    print("=" * 50)
    
    # Test single channel stats
    stats = analyze_eeg_time_domain_channel(data, channel_index=0, plot=True)
    print(f"Channel 0 stats: {stats}")
    
    # Expected: mean ~0, std should be reasonable (around 10-20 due to our signal composition)
    assert abs(stats['mean']) < 5, f"Mean too high: {stats['mean']}"
    assert 8 < stats['std'] < 25, f"Std outside expected range: {stats['std']}"
    print("✓ Single channel time-domain stats look reasonable")
    
    # Test all channels and plot distribution
    all_stats = analyze_all_channels_time_domain(data, plot=False)
    assert len(all_stats) == 32, f"Expected 32 channels, got {len(all_stats)}"
    
    # Plot statistics distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Time-Domain Statistics Across All Channels', fontsize=14)
    
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]
    mins = [s['min'] for s in all_stats]
    maxs = [s['max'] for s in all_stats]
    
    axes[0, 0].hist(means, bins=10, alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution of Mean Values')
    axes[0, 0].set_xlabel('Mean (μV)')
    axes[0, 0].axvline(0, color='red', linestyle='--', label='Expected ~0')
    axes[0, 0].legend()
    
    axes[0, 1].hist(stds, bins=10, alpha=0.7, color='orange')
    axes[0, 1].set_title('Distribution of Standard Deviations')
    axes[0, 1].set_xlabel('Std (μV)')
    
    axes[1, 0].hist(mins, bins=10, alpha=0.7, color='green')
    axes[1, 0].set_title('Distribution of Minimum Values')
    axes[1, 0].set_xlabel('Min (μV)')
    
    axes[1, 1].hist(maxs, bins=10, alpha=0.7, color='red')
    axes[1, 1].set_title('Distribution of Maximum Values')
    axes[1, 1].set_xlabel('Max (μV)')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ All channels analyzed successfully")
    return all_stats

def test_psd_functions(data):
    """Test PSD analysis functions with plots."""
    print("=" * 50)
    print("TESTING PSD FUNCTIONS")
    print("=" * 50)
    
    # Test PSD computation and plotting
    eeg_ch = data['eeg'][0, :]
    freqs, psd = compute_psd(eeg_ch, fs=200, method='welch')
    
    # Create comprehensive PSD plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Power Spectral Density Analysis', fontsize=16)
    
    # Plot PSD in different scales
    axes[0, 0].semilogy(freqs, psd)
    axes[0, 0].set_title('PSD - Semilog Scale')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Power (μV²/Hz)')
    axes[0, 0].axvspan(4, 8, alpha=0.3, color='red', label='Theta (4-8Hz)')
    axes[0, 0].axvspan(8, 13, alpha=0.3, color='blue', label='Alpha (8-13Hz)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(freqs, psd)
    axes[0, 1].set_title('PSD - Linear Scale')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power (μV²/Hz)')
    axes[0, 1].axvspan(4, 8, alpha=0.3, color='red', label='Theta')
    axes[0, 1].axvspan(8, 13, alpha=0.3, color='blue', label='Alpha')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Find and highlight peaks
    theta_idx = np.logical_and(freqs >= 5, freqs <= 7)
    theta_power = np.mean(psd[theta_idx])
    
    alpha_idx = np.logical_and(freqs >= 9, freqs <= 11)
    alpha_power = np.mean(psd[alpha_idx])
    
    print(f"Theta power (5-7 Hz): {theta_power:.2e}")
    print(f"Alpha power (9-11 Hz): {alpha_power:.2e}")
    
    # Theta should be stronger than alpha in our synthetic data
    assert theta_power > alpha_power, f"Theta power {theta_power:.2e} should be > alpha power {alpha_power:.2e}"
    print("✓ Theta band shows higher power than alpha as expected")
    
    # Test band power calculation and visualization
    band_powers = compute_band_power(freqs, psd)
    print(f"Band powers: {band_powers}")
    
    # Plot band powers
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    
    axes[1, 0].bar(bands, powers, alpha=0.7, color=['purple', 'red', 'blue', 'green', 'orange'])
    axes[1, 0].set_title('Band Power Comparison')
    axes[1, 0].set_ylabel('Power (μV²)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True)
    
    # Theta should be the dominant band
    max_band = max(band_powers, key=band_powers.get)
    assert max_band == 'theta', f"Expected theta to be dominant band, got {max_band}"
    print("✓ Theta band is correctly identified as dominant")
    
    # Test peak frequency finding and plot
    peak_freq, peak_power = find_peak_frequency(freqs, psd, fmin=4, fmax=8)
    print(f"Peak frequency in theta range: {peak_freq:.2f} Hz")
    assert 5.5 <= peak_freq <= 6.5, f"Peak frequency {peak_freq} not near expected 6 Hz"
    
    axes[1, 1].semilogy(freqs, psd)
    axes[1, 1].scatter([peak_freq], [peak_power], color='red', s=100, zorder=5)
    axes[1, 1].axvline(peak_freq, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'Peak Detection (Peak at {peak_freq:.2f} Hz)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power (μV²/Hz)')
    axes[1, 1].set_xlim(4, 8)
    axes[1, 1].grid(True)
    
    print("✓ Peak frequency correctly identified in theta range")
    
    plt.tight_layout()
    plt.show()
    
    return freqs, psd, band_powers

def test_morlet_functions(data):
    """Test Morlet spectrogram functions with plots."""
    print("=" * 50)
    print("TESTING MORLET SPECTROGRAM FUNCTIONS")
    print("=" * 50)
    
    # Test single channel Morlet
    eeg_ch = data['eeg'][0, :]
    freqs, power = morlet_spectrogram(eeg_ch, sfreq=200, freqs=np.arange(2, 30, 1))
    time = data['eeg_time'].flatten()
    
    print(f"Morlet output shape: {power.shape}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    
    # Create comprehensive Morlet visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Morlet Time-Frequency Analysis', fontsize=16)
    
    # Full spectrogram
    im1 = axes[0, 0].pcolormesh(time, freqs, np.log10(power + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 0].set_title('Full Morlet Spectrogram (Log Power)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    # Mark expected burst times
    for burst_time in [5, 15, 25, 35, 45, 55]:
        if burst_time < time[-1]:
            axes[0, 0].axvline(burst_time, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0, 0], label='Log Power')
    
    # Zoomed theta range
    theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
    im2 = axes[0, 1].pcolormesh(time, freqs[theta_idx], 
                               np.log10(power[theta_idx, :] + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[0, 1].set_title('Theta Band (4-8 Hz) Spectrogram')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    for burst_time in [5, 15, 25, 35, 45, 55]:
        if burst_time < time[-1]:
            axes[0, 1].axvline(burst_time, color='white', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1], label='Log Power')
    
    # Check for theta burst detection
    theta_power_time = np.mean(power[theta_idx, :], axis=0)
    
    # Find burst times
    burst_peaks = []
    for i in range(1, len(theta_power_time)-1):
        if (theta_power_time[i] > theta_power_time[i-1] and 
            theta_power_time[i] > theta_power_time[i+1] and
            theta_power_time[i] > np.mean(theta_power_time) + 2*np.std(theta_power_time)):
            burst_peaks.append(time[i])
    
    # Plot theta power over time
    axes[1, 0].plot(time, theta_power_time, 'b-', alpha=0.7)
    axes[1, 0].axhline(np.mean(theta_power_time) + 2*np.std(theta_power_time), 
                      color='red', linestyle='--', label='Detection threshold')
    for peak in burst_peaks[:5]:
        axes[1, 0].axvline(peak, color='green', alpha=0.7)
    # Mark expected burst times
    for burst_time in [5, 15, 25, 35, 45, 55]:
        if burst_time < time[-1]:
            axes[1, 0].axvspan(burst_time, burst_time+2, alpha=0.2, color='red')
    axes[1, 0].set_title('Theta Band Power Over Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Theta Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    print(f"Detected theta burst peaks at times: {burst_peaks[:5]}")
    print("✓ Morlet spectrogram computed successfully")
    
    # Test multi-channel Morlet
    freqs_all, tfr_array, channels = compute_all_channels_morlet(
        data, channels=range(4), freqs=np.arange(4, 15, 1)  # Test subset
    )
    
    print(f"Multi-channel TFR shape: {tfr_array.shape}")
    assert tfr_array.shape[0] == 4, f"Expected 4 channels, got {tfr_array.shape[0]}"
    
    # Plot multi-channel comparison
    for ch in range(4):
        theta_power_ch = np.mean(tfr_array[ch, :3, :], axis=0)  # First 3 freqs are theta
        axes[1, 1].plot(time, theta_power_ch, alpha=0.7, label=f'Ch{ch}')
    
    axes[1, 1].set_title('Theta Power Comparison Across Channels')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Theta Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Average spectrogram across channels
    mean_tfr = np.mean(tfr_array, axis=0)
    im3 = axes[2, 0].pcolormesh(time, freqs_all, np.log10(mean_tfr + 1e-12), 
                               shading='auto', cmap='inferno')
    axes[2, 0].set_title('Average Spectrogram (4 channels)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im3, ax=axes[2, 0], label='Log Power')
    
    # Frequency-averaged power over time
    freq_avg_power = np.mean(mean_tfr, axis=0)
    axes[2, 1].plot(time, freq_avg_power)
    axes[2, 1].set_title('Frequency-Averaged Power Over Time')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Average Power')
    axes[2, 1].grid(True)
    
    print("✓ Multi-channel Morlet analysis successful")
    
    plt.tight_layout()
    plt.show()
    
    return freqs, power, tfr_array

def test_integrated_analysis(data):
    """Test the high-level integrated analysis function with plots."""
    print("=" * 50)
    print("TESTING INTEGRATED ANALYSIS")
    print("=" * 50)
    
    # Test comprehensive channel analysis
    result = analyze_eeg_channel(
        data, 
        channel_index=0,
        fs=200,
        freq_method='welch',
        tf_method=True,
        morlet_freqs=np.arange(2, 25, 1),
        plot_types=['semilogy'],  # Enable one plot type
        extras=True
    )
    
    # Check all expected outputs are present
    expected_keys = ['stats', 'freqs', 'psd', 'band_power', 'peak_frequency', 'peak_power', 'morlet_freqs', 'morlet_power']
    for key in expected_keys:
        assert key in result, f"Missing key in analysis result: {key}"
    
    print("✓ Integrated analysis returns all expected outputs")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Integrated Analysis Summary', fontsize=16)
    
    # Time-domain statistics visualization
    stats_data = result['stats']
    stats_names = ['Mean', 'Std', 'Min', 'Max']
    stats_values = [stats_data['mean'], stats_data['std'], stats_data['min'], stats_data['max']]
    
    axes[0, 0].bar(stats_names, stats_values, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
    axes[0, 0].set_title('Time-Domain Statistics')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].grid(True)
    
    # Band power comparison
    band_powers = result['band_power']
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    
    bars = axes[0, 1].bar(bands, powers, alpha=0.7, color=['purple', 'red', 'blue', 'green', 'orange'])
    axes[0, 1].set_title('Band Power Analysis')
    axes[0, 1].set_ylabel('Power (μV²)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True)
    
    # Highlight dominant band
    max_idx = np.argmax(powers)
    bars[max_idx].set_color('darkred')
    bars[max_idx].set_alpha(1.0)
    
    # PSD with peak marked
    freqs = result['freqs']
    psd = result['psd']
    peak_freq = result['peak_frequency']
    peak_power = result['peak_power']
    
    axes[1, 0].semilogy(freqs, psd, 'b-', alpha=0.7)
    axes[1, 0].scatter([peak_freq], [peak_power], color='red', s=100, zorder=5)
    axes[1, 0].axvline(peak_freq, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title(f'PSD with Peak at {peak_freq:.2f} Hz')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power (μV²/Hz)')
    axes[1, 0].grid(True)
    
    # Morlet spectrogram summary
    morlet_freqs = result['morlet_freqs']
    morlet_power = result['morlet_power']
    time = data['eeg_time'].flatten()
    
    im = axes[1, 1].pcolormesh(time, morlet_freqs, np.log10(morlet_power + 1e-12), 
                              shading='auto', cmap='inferno')
    axes[1, 1].set_title('Morlet Spectrogram')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[1, 1], label='Log Power')
    
    plt.tight_layout()
    plt.show()
    
    # Validate theta dominance
    assert result['band_power']['theta'] > result['band_power']['alpha'], "Theta should dominate alpha"
    assert result['band_power']['theta'] > result['band_power']['beta'], "Theta should dominate beta"
    
    print(f"Peak frequency: {result['peak_frequency']:.2f} Hz")
    print(f"Band power ratios - Theta: {result['band_power']['theta']:.2e}, Alpha: {result['band_power']['alpha']:.2e}")
    print("✓ All validation checks passed")
    
    return result

def main():
    """Main test runner with comprehensive visualization."""
    print("Generating synthetic EEG data...")
    synthetic_data = generate_synthetic_eeg(duration=60, fs=200, n_channels=32)
    
    print(f"Generated data shape: {synthetic_data['eeg'].shape}")
    print(f"Time range: {synthetic_data['eeg_time'].min():.2f} - {synthetic_data['eeg_time'].max():.2f} s")
    
    # Plot data overview first
    plot_data_overview(synthetic_data)
    
    # Run all tests with plots
    try:
        time_stats = test_time_domain_functions(synthetic_data)
        freqs, psd, band_powers = test_psd_functions(synthetic_data)
        morlet_freqs, morlet_power, tfr_array = test_morlet_functions(synthetic_data)
        integrated_result = test_integrated_analysis(synthetic_data)
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("Summary:")
        print(f"- Data shape: {synthetic_data['eeg'].shape}")
        print(f"- Dominant frequency band: theta ({band_powers['theta']:.2e})")
        print(f"- Peak frequency: {integrated_result['peak_frequency']:.2f} Hz")
        print(f"- Morlet TFR shape: {tfr_array.shape}")
        
        # Final summary plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create validation summary
        validation_results = {
            'Time-domain stats': '✓ PASS',
            'PSD analysis': '✓ PASS', 
            'Band power calc': '✓ PASS',
            'Morlet analysis': '✓ PASS',
            'Integrated workflow': '✓ PASS'
        }
        
        y_pos = range(len(validation_results))
        ax.barh(y_pos, [1]*len(validation_results), color='green', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(validation_results.keys()))
        ax.set_xlabel('Test Status')
        ax.set_title('EEG Analysis Package Validation Summary')
        
        for i, (test, result) in enumerate(validation_results.items()):
            ax.text(0.5, i, result, ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)