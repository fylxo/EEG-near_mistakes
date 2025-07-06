#!/usr/bin/env python3
"""
Comprehensive test suite for EEG analysis package.
Tests all functions, edge cases, and parameter variations.
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

# Disable plots for automated testing
plt.ioff()

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
        'iti_sizes': np.array([0.5, 1])
    }

def test_psd_methods():
    """Test both Welch and multitaper PSD methods."""
    print("Testing PSD methods...")
    data = generate_test_data()
    eeg_ch = data['eeg'][0, :]
    
    # Test Welch method
    freqs_w, psd_w = compute_psd(eeg_ch, method='welch')
    assert len(freqs_w) > 0, "Welch PSD returned empty frequencies"
    assert len(psd_w) > 0, "Welch PSD returned empty power"
    
    # Test multitaper method
    freqs_m, psd_m = compute_psd(eeg_ch, method='multitaper')
    assert len(freqs_m) > 0, "Multitaper PSD returned empty frequencies"
    assert len(psd_m) > 0, "Multitaper PSD returned empty power"
    
    # Test invalid method
    try:
        compute_psd(eeg_ch, method='invalid')
        assert False, "Should have raised ValueError for invalid method"
    except ValueError:
        pass
    
    print("✓ Both PSD methods work correctly")
    return freqs_w, psd_w, freqs_m, psd_m

def test_time_frequency_functions():
    """Test all time-frequency analysis functions."""
    print("Testing time-frequency functions...")
    data = generate_test_data()
    
    # Test basic Morlet
    freqs, power = morlet_spectrogram(data['eeg'][0, :], sfreq=200)
    assert power.shape[0] == len(freqs), "Frequency dimension mismatch"
    
    # Test extract_morlet_band_power
    theta_power = extract_morlet_band_power(power, freqs, (4, 8))
    assert len(theta_power) == power.shape[1], "Time dimension mismatch"
    
    # Test compute_all_channels_morlet
    freqs_all, tfr_array, channels = compute_all_channels_morlet(
        data, channels=range(5), freqs=np.arange(4, 15, 1)
    )
    assert tfr_array.shape[0] == 5, "Wrong number of channels"
    assert tfr_array.shape[1] == len(freqs_all), "Frequency dimension mismatch"
    
    # Test extract_tfr_time_window
    times = data['eeg_time'].flatten()
    windowed_tfr, window_times = extract_tfr_time_window(tfr_array, times, 5, 15)
    assert windowed_tfr.shape[2] == len(window_times), "Window time dimension mismatch"
    assert np.all(window_times >= 5) and np.all(window_times <= 15), "Time window incorrect"
    
    print("✓ All time-frequency functions work correctly")
    return tfr_array, window_times

def test_analysis_utils():
    """Test analysis utility functions."""
    print("Testing analysis utilities...")
    
    # Create mock data for testing
    all_data = []
    for i in range(3):
        data = generate_test_data()
        data['rat_id'] = f'RAT_{i:03d}'
        data['session_date'] = f'2025-06-{13+i:02d}'
        data['file_path'] = f'/mock/path/file_{i}.mat'  # Add required field
        all_data.append(data)
    
    # Test create_summary_dataframe
    df_summary, grouped, longest, shortest = create_summary_dataframe(all_data, verbose=False)
    assert len(df_summary) == 3, "Wrong number of sessions in summary"
    assert 'rat_id' in df_summary.columns, "Missing rat_id column"
    assert 'eeg_len' in df_summary.columns, "Missing eeg_len column"
    
    # Test convert_to_raw
    raw = convert_to_raw(all_data[0])
    assert hasattr(raw, 'info'), "Raw object missing info attribute"
    assert raw.info['sfreq'] == 200, "Wrong sampling frequency"
    
    print("✓ Analysis utilities work correctly")
    return df_summary, raw

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    # Test empty data
    try:
        eeg_stats(np.array([]))
        assert False, "Should fail with empty array"
    except:
        pass
    
    # Test single sample
    stats = eeg_stats(np.array([1.0]))
    assert stats['mean'] == 1.0, "Single sample mean incorrect"
    assert stats['std'] == 0.0, "Single sample std should be 0"
    
    # Test custom frequency bands
    freqs = np.arange(1, 50, 0.5)
    psd = np.random.random(len(freqs))
    custom_bands = {'low': (1, 10), 'high': (10, 30)}
    band_powers = compute_band_power(freqs, psd, custom_bands)
    assert 'low' in band_powers, "Missing custom band"
    assert 'high' in band_powers, "Missing custom band"
    
    # Test find_peak_frequency with no peak in range
    peak_freq, peak_power = find_peak_frequency(freqs, psd, fmin=100, fmax=200)
    assert peak_freq is None, "Should return None for out-of-range"
    assert peak_power is None, "Should return None for out-of-range"
    
    print("✓ Edge cases handled correctly")

def test_parameter_variations():
    """Test different parameter combinations."""
    print("Testing parameter variations...")
    data = generate_test_data()
    
    # Test different nperseg values for Welch
    for nperseg in [256, 512, 1024]:
        freqs, psd = compute_psd(data['eeg'][0, :], nperseg=nperseg)
        assert len(freqs) > 0, f"Failed with nperseg={nperseg}"
    
    # Test different n_cycles for Morlet
    for n_cycles in [3, 7, 10]:
        freqs, power = morlet_spectrogram(data['eeg'][0, :], n_cycles=n_cycles)
        assert power.shape[0] == len(freqs), f"Failed with n_cycles={n_cycles}"
    
    # Test time range slicing in analyze_eeg_channel
    result = analyze_eeg_channel(
        data, channel_index=0, time_range=(5, 15), 
        plot_types=[], tf_method=False
    )
    assert 'stats' in result, "Missing stats in time-sliced analysis"
    
    # Test different frequency ranges
    for fmin, fmax in [(1, 30), (5, 50), (10, 100)]:
        freqs, psd = compute_psd(data['eeg'][0, :], fmin=fmin, fmax=fmax)
        assert freqs[0] >= fmin, f"Frequency range wrong for {fmin}-{fmax}Hz"
        assert freqs[-1] <= fmax, f"Frequency range wrong for {fmin}-{fmax}Hz"
    
    print("✓ Parameter variations work correctly")

def test_integrated_workflows():
    """Test realistic analysis workflows."""
    print("Testing integrated workflows...")
    data = generate_test_data()
    
    # Workflow 1: Full single-channel analysis
    result = analyze_eeg_channel(
        data, channel_index=0, freq_method='welch', tf_method=True,
        plot_types=[], extras=True
    )
    expected_keys = ['stats', 'freqs', 'psd', 'band_power', 'peak_frequency', 
                     'peak_power', 'morlet_freqs', 'morlet_power']
    for key in expected_keys:
        assert key in result, f"Missing {key} in integrated analysis"
    
    # Workflow 2: Multi-channel time-domain analysis
    all_stats = analyze_all_channels_time_domain(data, channels=range(10))
    assert len(all_stats) == 10, "Wrong number of channel stats"
    
    # Workflow 3: Multi-channel time-frequency analysis
    freqs, tfr_array, channels = compute_all_channels_morlet(data, channels=range(5))
    theta_power_all = extract_morlet_band_power(tfr_array[0], freqs, (4, 8))
    assert len(theta_power_all) > 0, "Theta power extraction failed"
    
    print("✓ Integrated workflows work correctly")

def main():
    """Run comprehensive test suite."""
    print("Running comprehensive EEG analysis tests...\n")
    
    test_functions = [
        test_psd_methods,
        test_time_frequency_functions, 
        test_analysis_utils,
        test_edge_cases,
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
    
    print("\n" + "="*60)
    if failed_tests:
        print(f"❌ {len(failed_tests)} TESTS FAILED:")
        for test in failed_tests:
            print(f"  - {test}")
        return False
    else:
        print("✅ ALL COMPREHENSIVE TESTS PASSED!")
        print("Coverage includes:")
        print("  - Both Welch and multitaper PSD methods")
        print("  - All time-frequency analysis functions")
        print("  - Analysis utility functions")
        print("  - Edge cases and error conditions")
        print("  - Parameter variations")
        print("  - Integrated analysis workflows")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)