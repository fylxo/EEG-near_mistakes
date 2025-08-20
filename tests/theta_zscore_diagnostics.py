#!/usr/bin/env python3
"""
Theta Z-Score Diagnostics Script

This script provides comprehensive diagnostics and visualization for theta power analysis
around Near Mistake (NM) events. It shows:
1. Raw (unnormalized) theta power spectrograms by NM size
2. Session-wide mean and std used for z-scoring
3. Single-event raw and z-scored windows with event alignment
4. Detailed numerical analysis of power values, means, and z-scores

Diagnostic tools for theta z-score analysis in EEG near-mistake research
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import defaultdict

# Add src/core to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

from nm_theta_single_basic import load_session_data, compute_roi_theta_spectrogram
from electrode_utils import get_channels, load_electrode_mappings

def extract_event_windows(power_matrix: np.ndarray, 
                         times: np.ndarray, 
                         event_times: np.ndarray,
                         window_duration: float = 1.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Extract event windows from power matrix.
    
    Parameters:
    -----------
    power_matrix : np.ndarray
        Power matrix (n_freqs, n_times)
    times : np.ndarray
        Time vector
    event_times : np.ndarray
        Event time points
    window_duration : float
        Total window duration (±half around event)
    
    Returns:
    --------
    windows : List[np.ndarray]
        List of event windows (n_freqs, window_samples)
    window_times : np.ndarray
        Time vector for windows relative to event (e.g., -0.5 to +0.5)
    """
    half_window = window_duration / 2
    
    # Calculate actual sampling frequency for information
    time_diffs = np.diff(times)
    actual_sfreq = 1 / np.median(time_diffs)
    
    # ALWAYS use exactly 200 Hz for consistency across all rats and sessions
    # This ensures cross-rat averaging will work properly
    sfreq = 200.0
    window_samples = int(window_duration * sfreq)
    
    # Warn if there's a significant difference from actual frequency
    if abs(actual_sfreq - sfreq) > 1.0:
        print(f"Warning: Actual sampling frequency {actual_sfreq:.2f} Hz differs from standard {sfreq} Hz")
        print(f"Using standard {sfreq} Hz for cross-rat consistency")
    
    # Create relative time vector for windows
    window_times = np.linspace(-half_window, half_window, window_samples)
    
    windows = []
    for event_time in event_times:
        # Find closest time index
        event_idx = np.argmin(np.abs(times - event_time))
        
        # Extract window
        start_idx = max(0, event_idx - window_samples // 2)
        end_idx = min(len(times), start_idx + window_samples)
        
        if end_idx - start_idx == window_samples:
            window = power_matrix[:, start_idx:end_idx]
            windows.append(window)
    
    return windows, window_times

def compute_zscore_normalization(power_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute z-score normalization parameters and normalized matrix.
    
    Parameters:
    -----------
    power_matrix : np.ndarray
        Raw power matrix (n_freqs, n_times)
    
    Returns:
    --------
    normalized_power : np.ndarray
        Z-score normalized power matrix
    session_mean : np.ndarray
        Mean per frequency (n_freqs,)
    session_std : np.ndarray
        Standard deviation per frequency (n_freqs,)
    """
    session_mean = np.mean(power_matrix, axis=1)  # Mean per frequency
    session_std = np.std(power_matrix, axis=1)    # Std per frequency
    session_std = np.maximum(session_std, 1e-12)  # Avoid division by zero
    
    # Z-score normalize
    normalized_power = (power_matrix - session_mean[:, np.newaxis]) / session_std[:, np.newaxis]
    
    return normalized_power, session_mean, session_std

def plot_raw_power_by_nm_size(windows_by_size: Dict, 
                             window_times: np.ndarray, 
                             freqs: np.ndarray,
                             roi_name: str):
    """Plot raw theta power spectrograms by NM size."""
    nm_sizes = sorted(windows_by_size.keys())
    n_sizes = len(nm_sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(4 * n_sizes, 6))
    if n_sizes == 1:
        axes = [axes]
    
    for i, nm_size in enumerate(nm_sizes):
        windows = windows_by_size[nm_size]
        if len(windows) > 0:
            # Average across events
            avg_power = np.mean(windows, axis=0)
            
            im = axes[i].pcolormesh(window_times, freqs, avg_power, 
                                   shading='auto', cmap='viridis')
            axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2, label='NM Event')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_title(f'Raw Power - NM Size {nm_size}\n({len(windows)} events)')
            axes[i].legend()
            plt.colorbar(im, ax=axes[i], label='Power (µV²)')
    
    plt.suptitle(f'Raw Theta Power by NM Size - {roi_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_session_normalization_stats(session_mean: np.ndarray, 
                                    session_std: np.ndarray, 
                                    freqs: np.ndarray,
                                    roi_name: str):
    """Plot session-wide mean and std used for z-scoring."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mean
    ax1.plot(freqs, session_mean, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Mean Power (µV²)')
    ax1.set_title('Session-wide Mean Power per Frequency')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot std
    ax2.plot(freqs, session_std, 'r-', linewidth=2, marker='s')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Std Power (µV²)')
    ax2.set_title('Session-wide Std Power per Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(f'Z-Score Normalization Parameters - {roi_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_single_event_comparison(raw_window: np.ndarray,
                                zscore_window: np.ndarray,
                                window_times: np.ndarray,
                                freqs: np.ndarray,
                                event_idx: int,
                                nm_size: int,
                                roi_name: str):
    """Plot raw vs z-scored single event window."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Raw power
    im1 = ax1.pcolormesh(window_times, freqs, raw_window, 
                        shading='auto', cmap='viridis')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='NM Event')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Raw Power - Event {event_idx}\n(NM Size {nm_size})')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Power (µV²)')
    
    # Z-scored power
    im2 = ax2.pcolormesh(window_times, freqs, zscore_window, 
                        shading='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='NM Event')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Z-Scored Power - Event {event_idx}\n(NM Size {nm_size})')
    ax2.legend()
    plt.colorbar(im2, ax=ax2, label='Z-Score')
    
    plt.suptitle(f'Single Event Comparison - {roi_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_diagnostic_statistics(raw_windows: List[np.ndarray],
                               zscore_windows: List[np.ndarray],
                               session_mean: np.ndarray,
                               session_std: np.ndarray,
                               freqs: np.ndarray,
                               nm_sizes: List[int]):
    """Print comprehensive diagnostic statistics."""
    print("\n" + "="*80)
    print("THETA POWER DIAGNOSTIC STATISTICS")
    print("="*80)
    
    # Session-wide statistics
    print(f"\n📊 SESSION-WIDE NORMALIZATION PARAMETERS:")
    print(f"   Frequency range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz ({len(freqs)} frequencies)")
    print(f"   Mean power range: {session_mean.min():.2e} - {session_mean.max():.2e} µV²")
    print(f"   Std power range: {session_std.min():.2e} - {session_std.max():.2e} µV²")
    print(f"   Mean/Std ratio range: {(session_mean/session_std).min():.2f} - {(session_mean/session_std).max():.2f}")
    
    # Raw power statistics by NM size
    print(f"\n📈 RAW POWER STATISTICS BY NM SIZE:")
    for nm_size in sorted(set(nm_sizes)):
        size_windows = [w for w, s in zip(raw_windows, nm_sizes) if s == nm_size]
        if size_windows:
            all_power = np.concatenate([w.flatten() for w in size_windows])
            print(f"   NM Size {nm_size} ({len(size_windows)} events):")
            print(f"     Power range: {all_power.min():.2e} - {all_power.max():.2e} µV²")
            print(f"     Power mean±std: {all_power.mean():.2e} ± {all_power.std():.2e} µV²")
            print(f"     Power percentiles (25th, 50th, 75th): {np.percentile(all_power, [25, 50, 75])}")
    
    # Z-score statistics by NM size
    print(f"\n📉 Z-SCORE STATISTICS BY NM SIZE:")
    for nm_size in sorted(set(nm_sizes)):
        size_windows = [w for w, s in zip(zscore_windows, nm_sizes) if s == nm_size]
        if size_windows:
            all_zscores = np.concatenate([w.flatten() for w in size_windows])
            print(f"   NM Size {nm_size} ({len(size_windows)} events):")
            print(f"     Z-score range: {all_zscores.min():.2f} - {all_zscores.max():.2f}")
            print(f"     Z-score mean±std: {all_zscores.mean():.2f} ± {all_zscores.std():.2f}")
            print(f"     Z-score percentiles (25th, 50th, 75th): {np.percentile(all_zscores, [25, 50, 75])}")
    
    # Frequency-specific analysis
    print(f"\n🎯 FREQUENCY-SPECIFIC ANALYSIS:")
    for i, freq in enumerate(freqs[::5]):  # Sample every 5th frequency
        idx = i * 5
        if idx < len(freqs):
            freq_raw = [w[idx, :].flatten() for w in raw_windows]
            freq_zscore = [w[idx, :].flatten() for w in zscore_windows]
            
            all_freq_raw = np.concatenate(freq_raw)
            all_freq_zscore = np.concatenate(freq_zscore)
            
            print(f"   {freq:.2f} Hz:")
            print(f"     Session mean: {session_mean[idx]:.2e} µV²")
            print(f"     Session std: {session_std[idx]:.2e} µV²")
            print(f"     Event raw power: {all_freq_raw.mean():.2e} ± {all_freq_raw.std():.2e} µV²")
            print(f"     Event z-scores: {all_freq_zscore.mean():.2f} ± {all_freq_zscore.std():.2f}")

def main():
    """Main diagnostics function."""
    print("="*80)
    print("THETA Z-SCORE DIAGNOSTICS")
    print("="*80)
    
    try:
        # Load session data
        print("Loading EEG session data...")
        session_data = load_session_data('data/processed/all_eeg_data.pkl', session_index=0)
        
        # Analysis parameters
        roi_specification = 'frontal'  # or 'hippocampus', or [1, 2, 3] for custom channels
        freq_range = (2, 10)
        n_freqs = 25
        window_duration = 1.0
        n_cycles_factor = 3.0
        
        print(f"Analysis parameters:")
        print(f"  ROI: {roi_specification}")
        print(f"  Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"  Number of frequencies: {n_freqs}")
        print(f"  Window duration: ±{window_duration/2:.1f}s")
        
        # Get ROI channels
        rat_id = session_data.get('rat_id', 'unknown')
        mapping_df = load_electrode_mappings()
        roi_channels = get_channels(rat_id, roi_specification, mapping_df)
        roi_name = f"{roi_specification}" if isinstance(roi_specification, str) else f"Custom_{len(roi_channels)}ch"
        
        print(f"  ROI channels: {roi_channels}")
        print(f"  Rat ID: {rat_id}")
        
        # Compute ROI theta spectrogram (this gives us the raw average ROI power)
        print("\nComputing ROI theta spectrogram...")
        freqs, roi_power, channel_powers = compute_roi_theta_spectrogram(
            session_data['eeg'],
            roi_channels,
            sfreq=200.0,
            freq_range=freq_range,
            n_freqs=n_freqs,
            n_cycles_factor=n_cycles_factor
        )
        
        # Note: roi_power is already z-score normalized per channel, then averaged
        # We need to get the raw power for diagnostics
        print("\nRecomputing individual channel powers for raw analysis...")
        
        # Recompute without normalization for raw power analysis
        times = session_data['eeg_time'].flatten()
        nm_times = session_data['nm_peak_times']
        nm_sizes = session_data['nm_sizes']
        
        # Get raw ROI power (average of raw channel powers)
        raw_roi_power = np.mean(channel_powers, axis=0)  # Average across channels
        
        # For true raw power, we need to recompute without any normalization
        # Let's extract one channel's raw power for detailed analysis
        test_channel = roi_channels[0]
        eeg_channel = session_data['eeg'][test_channel, :]
        
        # Compute raw spectrogram for single channel
        from nm_theta_single_basic import compute_high_res_theta_spectrogram
        _, single_channel_raw_power = compute_high_res_theta_spectrogram(
            eeg_channel, 
            sfreq=200.0, 
            freq_range=freq_range, 
            n_freqs=n_freqs, 
            n_cycles_factor=n_cycles_factor
        )
        
        print(f"Using channel {test_channel} for detailed raw power analysis")
        
        # Compute normalization parameters
        normalized_power, session_mean, session_std = compute_zscore_normalization(single_channel_raw_power)
        
        # Extract event windows
        print(f"\nExtracting event windows for {len(nm_times)} NM events...")
        raw_windows, window_times = extract_event_windows(
            single_channel_raw_power, times, nm_times, window_duration
        )
        zscore_windows, _ = extract_event_windows(
            normalized_power, times, nm_times, window_duration
        )
        
        # Group windows by NM size
        windows_by_size = defaultdict(list)
        valid_nm_sizes = []
        valid_raw_windows = []
        valid_zscore_windows = []
        
        for i, (raw_win, zscore_win) in enumerate(zip(raw_windows, zscore_windows)):
            if i < len(nm_sizes):
                nm_size = nm_sizes[i]
                windows_by_size[nm_size].append(raw_win)
                valid_nm_sizes.append(nm_size)
                valid_raw_windows.append(raw_win)
                valid_zscore_windows.append(zscore_win)
        
        print(f"Valid event windows: {len(valid_raw_windows)}")
        print(f"NM sizes present: {sorted(set(valid_nm_sizes))}")
        
        # 1. Plot raw power by NM size
        print("\n1. Plotting raw theta power by NM size...")
        plot_raw_power_by_nm_size(windows_by_size, window_times, freqs, roi_name)
        
        # 2. Plot session normalization parameters
        print("2. Plotting session normalization parameters...")
        plot_session_normalization_stats(session_mean, session_std, freqs, roi_name)
        
        # 3. Plot single event comparisons (1-2 random events)
        print("3. Plotting single event comparisons...")
        n_events_to_show = min(2, len(valid_raw_windows))
        random_indices = np.random.choice(len(valid_raw_windows), n_events_to_show, replace=False)
        
        for i, event_idx in enumerate(random_indices):
            nm_size = valid_nm_sizes[event_idx]
            plot_single_event_comparison(
                valid_raw_windows[event_idx],
                valid_zscore_windows[event_idx],
                window_times,
                freqs,
                event_idx,
                nm_size,
                roi_name
            )
        
        # 4. Print comprehensive statistics
        print("4. Computing comprehensive diagnostic statistics...")
        print_diagnostic_statistics(
            valid_raw_windows,
            valid_zscore_windows,
            session_mean,
            session_std,
            freqs,
            valid_nm_sizes
        )
        
        print(f"\n✅ Diagnostics completed successfully!")
        print(f"   - Analyzed {len(valid_raw_windows)} events")
        print(f"   - {len(set(valid_nm_sizes))} different NM sizes")
        print(f"   - {len(freqs)} frequencies from {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")
        
    except Exception as e:
        print(f"❌ Error in diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()