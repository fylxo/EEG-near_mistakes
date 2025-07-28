#!/usr/bin/env python3
"""
Near Mistake Theta Oscillation Analysis

This script analyzes theta oscillations (3-8 Hz) around Near Mistake (NM) events.
It computes spectrograms with high frequency resolution, extracts event windows,
performs z-score normalization, and generates visualization plots.

Usage:
    python nm_theta_analysis.py

Author: Generated for EEG near-mistake analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from scipy import signal
import mne

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing package
from eeg_analysis_package.time_frequency import morlet_spectrogram
from electrode_utils import get_channels, load_electrode_mappings, ROI_MAP


def get_electrode_numbers_from_channels(rat_id: Union[str, int], 
                                       channel_indices: List[int], 
                                       mapping_df: pd.DataFrame) -> List[int]:
    """
    Convert channel indices back to electrode numbers for display purposes.
    
    Parameters:
    -----------
    rat_id : Union[str, int]
        Rat identifier
    channel_indices : List[int]
        0-based channel indices
    mapping_df : pd.DataFrame
        Electrode mapping dataframe
    
    Returns:
    --------
    electrode_numbers : List[int]
        Electrode numbers (1-32) corresponding to the channel indices
    """
    # Handle rat_id type conversion (same logic as in electrode_utils.py)
    str_id = str(rat_id)
    int_id = int(rat_id) if isinstance(rat_id, (str, int)) and str(rat_id).isdigit() else None

    if rat_id in mapping_df.index:
        row = mapping_df.loc[rat_id].values
    elif str_id in mapping_df.index:
        row = mapping_df.loc[str_id].values
    elif int_id is not None and int_id in mapping_df.index:
        row = mapping_df.loc[int_id].values
    else:
        raise ValueError(f"Rat ID {rat_id} not found in mapping DataFrame")

    # Remove NaN values and convert to integers, then to list for consistent display
    row = row[~pd.isna(row)].astype(int).tolist()
    
    # Get electrode numbers for the given channel indices
    electrode_numbers = []
    for ch_idx in channel_indices:
        if ch_idx < len(row):
            electrode_numbers.append(row[ch_idx])
        else:
            electrode_numbers.append(f"Ch{ch_idx}")  # Fallback for out-of-range channels
    
    return electrode_numbers


def load_session_data(pkl_path: str, session_index: int = 0) -> Dict:
    """
    Load a specific session from all_eeg_data.pkl.
    
    Parameters:
    -----------
    pkl_path : str
        Path to all_eeg_data.pkl
    session_index : int
        Index of session to load (0-based)
    
    Returns:
    --------
    session_data : Dict
        Single session data dictionary
    """
    print(f"Loading session {session_index} from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        try:
            all_data = pickle.load(f)
            if session_index >= len(all_data):
                raise IndexError(f"Session index {session_index} out of range. Total sessions: {len(all_data)}")
            
            session_data = all_data[session_index]
            print(f"Loaded session: {session_data.get('rat_id', 'unknown')} - {session_data.get('session_date', 'unknown')}")
            print(f"EEG shape: {session_data['eeg'].shape}")
            print(f"NM events: {len(session_data['nm_peak_times'])}")
            print(f"NM sizes: {np.unique(session_data['nm_sizes'])}")
            
            return session_data
        except Exception as e:
            print(f"Error loading session data: {e}")
            raise


def compute_roi_theta_spectrogram(eeg_data: np.ndarray,
                                 roi_channels: List[int],
                                 sfreq: float = 200.0,
                                 freq_range: Tuple[float, float] = (3, 8),
                                 n_freqs: int = 20,
                                 n_cycles_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute high-resolution spectrogram for ROI channels with per-channel normalization.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data (n_channels, n_samples)
    roi_channels : List[int]
        List of channel indices to include in ROI
    sfreq : float
        Sampling frequency in Hz
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    n_freqs : int
        Number of logarithmically spaced frequencies (default: 20)
    n_cycles_factor : float
        Factor for determining number of cycles per frequency
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    roi_power : np.ndarray
        Average ROI power matrix (n_freqs, n_times)
    channel_powers : np.ndarray
        Individual channel powers (n_channels, n_freqs, n_times)
    """
    print(f"Computing ROI theta spectrogram for {len(roi_channels)} channels...")
    print(f"ROI channels: {roi_channels}")
    
    # Additional verification for the channels being processed
    print(f"📊 PROCESSING VERIFICATION:")
    print(f"   Using {len(roi_channels)} channels: {sorted(roi_channels)}")
    print(f"   EEG data shape: {eeg_data.shape}")
    print(f"   Each channel will be z-score normalized individually, then averaged")
    
    # Create logarithmically spaced frequency vector
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    #n_cycles = np.maximum(3, freqs * n_cycles_factor)
    #n_cycles = np.linspace(3, 10, len(freqs))
    n_cycles = np.full(len(freqs), 5)


    # Print exact frequencies being used
    print(f"📊 Using {n_freqs} logarithmically spaced frequencies:")
    print(f"   Range: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    print(f"   Frequencies: {[f'{f:.2f}' for f in freqs]}")
    print(f"   N-cycles: {[f'{nc:.1f}' for nc in n_cycles]}")
    
    # Storage for individual channel spectrograms
    channel_powers = []
    channel_means = []
    channel_stds = []
    
    # Process each channel individually
    for i, ch_idx in enumerate(roi_channels):
        print(f"Processing channel {ch_idx} ({i+1}/{len(roi_channels)})")
        
        # Extract channel data
        eeg_channel = eeg_data[ch_idx, :]
        
        # OPTIMIZED: Process all frequencies at once with adaptive n_cycles
        data = eeg_channel[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
        power = mne.time_frequency.tfr_array_morlet(
            data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, 
            output='power', zero_mean=True
        )
        channel_power = power[0, 0, :, :]  # (n_freqs, n_times)
        
        # Compute per-channel statistics for normalization
        ch_mean = np.mean(channel_power, axis=1)  # Mean per frequency
        ch_std = np.std(channel_power, axis=1)    # Std per frequency
        ch_std = np.maximum(ch_std, 1e-12)        # Avoid division by zero
        
        # Z-score normalize this channel
        normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
        
        channel_powers.append(normalized_power)
        channel_means.append(ch_mean)
        channel_stds.append(ch_std)
        
        print(f"  Channel {ch_idx} power range: {channel_power.min():.2e} - {channel_power.max():.2e}")
        print(f"  Channel {ch_idx} z-score range: {normalized_power.min():.2f} - {normalized_power.max():.2f}")
    
    # Convert to arrays
    channel_powers = np.array(channel_powers)  # (n_channels, n_freqs, n_times)
    channel_means = np.array(channel_means)    # (n_channels, n_freqs)
    channel_stds = np.array(channel_stds)      # (n_channels, n_freqs)
    
    # Average across channels after normalization
    roi_power = np.mean(channel_powers, axis=0)  # (n_freqs, n_times)
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    
    return freqs, roi_power, channel_powers


def compute_high_res_theta_spectrogram(eeg_data: np.ndarray, 
                                     sfreq: float = 200.0,
                                     freq_range: Tuple[float, float] = (3, 8),
                                     n_freqs: int = 20,
                                     n_cycles_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute high-resolution spectrogram focused on theta range (3-8 Hz).
    [DEPRECATED: Use compute_roi_theta_spectrogram for ROI analysis]
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data for single channel (n_samples,)
    sfreq : float
        Sampling frequency in Hz
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    n_freqs : int
        Number of logarithmically spaced frequencies (default: 20)
    n_cycles_factor : float
        Factor for determining number of cycles per frequency (higher = better freq resolution)
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    power : np.ndarray
        Time-frequency power matrix (n_freqs, n_times)
    """
    # Create logarithmically spaced frequency vector
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    
    # For low frequencies, use more cycles for better frequency resolution
    # Use adaptive n_cycles: more cycles for lower frequencies
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    print(f"Computing high-resolution theta spectrogram...")
    print(f"📊 Using {n_freqs} logarithmically spaced frequencies:")
    print(f"   Range: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    print(f"   Frequencies: {[f'{f:.2f}' for f in freqs]}")
    print(f"   N-cycles: {[f'{nc:.1f}' for nc in n_cycles]}")
    
    # Compute Morlet spectrogram with adaptive cycles
    try:
        # OPTIMIZED: Process all frequencies at once with adaptive n_cycles
        data = eeg_data[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
        power = mne.time_frequency.tfr_array_morlet(
            data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, 
            output='power', zero_mean=True
        )
        power = power[0, 0, :, :]  # (n_freqs, n_times)
        
        # Validate time axis alignment
        if len(times) != power.shape[1]:
            print(f"Warning: Time axis length mismatch for rat {rat_id}")
            print(f"  Original times length: {len(times)}")
            print(f"  Spectrogram time axis: {power.shape[1]}")
            print(f"  Difference: {len(times) - power.shape[1]} samples")
        
    except Exception as e:
        print(f"Error in spectrogram computation: {e}")
        # Fallback to fixed n_cycles
        print("Falling back to fixed n_cycles approach...")
        _, power = morlet_spectrogram(
            eeg_data, 
            sfreq=sfreq, 
            freqs=freqs, 
            n_cycles=5
        )
    
    print(f"Spectrogram computed. Shape: {power.shape}")
    return freqs, power


def compute_global_statistics(power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and standard deviation for each frequency bin.
    
    Parameters:
    -----------
    power : np.ndarray
        Time-frequency power matrix (n_freqs, n_times)
    
    Returns:
    --------
    global_mean : np.ndarray
        Mean power per frequency (n_freqs,)
    global_std : np.ndarray
        Standard deviation per frequency (n_freqs,)
    """
    print("Computing global statistics for normalization...")
    
    # Compute statistics across time for each frequency
    global_mean = np.mean(power, axis=1)
    global_std = np.std(power, axis=1)
    
    # Ensure no zero standard deviations (would cause division by zero)
    global_std = np.maximum(global_std, 1e-12)
    
    print(f"Global mean range: {global_mean.min():.2e} - {global_mean.max():.2e}")
    print(f"Global std range: {global_std.min():.2e} - {global_std.max():.2e}")
    
    return global_mean, global_std


def extract_nm_event_windows(power: np.ndarray,
                            times: np.ndarray,
                            nm_peak_times: np.ndarray,
                            nm_sizes: np.ndarray,
                            window_duration: float = 1.0) -> Dict:
    """
    Extract time-frequency windows around NM events for each NM size.
    
    Parameters:
    -----------
    power : np.ndarray
        Time-frequency power matrix (n_freqs, n_times)
    times : np.ndarray
        Time vector for the spectrogram
    nm_peak_times : np.ndarray
        NM event times in seconds
    nm_sizes : np.ndarray
        NM event sizes
    window_duration : float
        Total window duration in seconds (centered on event)
    
    Returns:
    --------
    nm_windows : Dict
        Dictionary with windows organized by NM size
    """
    print(f"Extracting NM event windows (±{window_duration/2:.1f}s around events)...")
    
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
    
    print(f"Actual sampling frequency: {actual_sfreq:.2f} Hz → Using {sfreq} Hz (window size: {window_samples} samples)")
    
    # Initialize storage for each NM size
    nm_windows = defaultdict(list)
    nm_window_times = np.linspace(-half_window, half_window, window_samples)
    valid_events = defaultdict(list)
    
    unique_sizes = np.unique(nm_sizes)
    print(f"Found NM sizes: {unique_sizes}")
    
    for i, (event_time, event_size) in enumerate(zip(nm_peak_times, nm_sizes)):
        # Check if window is within recording bounds
        start_time = event_time - half_window
        end_time = event_time + half_window
        
        if start_time >= times[0] and end_time <= times[-1]:
            # Find time indices
            start_idx = np.searchsorted(times, start_time)
            end_idx = np.searchsorted(times, end_time)
            
            if end_idx <= len(times) and start_idx < end_idx:
                # Extract window
                window_power = power[:, start_idx:end_idx]
                actual_samples = end_idx - start_idx
                expected_samples = int((end_time - start_time) * 200)  # Based on 200Hz assumption
                
                # Allow sample count variations due to timing precision (up to 5% tolerance)
                max_tolerance = max(10, int(expected_samples * 0.05))  # At least 10 samples or 5%
                if abs(actual_samples - expected_samples) <= max_tolerance:
                    nm_windows[event_size].append(window_power)
                    valid_events[event_size].append(i)
                else:
                    difference = abs(actual_samples - expected_samples)
                    print(f"Warning: NM event {i} at {event_time:.2f}s has significant timing mismatch")
                    print(f"  Expected samples: {expected_samples}, actual: {actual_samples} (diff: {difference})")
                    print(f"  Time window: {start_time:.3f} to {end_time:.3f}s")
                    print(f"  Tolerance exceeded: {difference} > {max_tolerance} samples")
            else:
                print(f"Warning: NM event {i} at {event_time:.2f}s too close to recording end")
        else:
            print(f"Warning: NM event {i} at {event_time:.2f}s outside recording bounds")
    
    # Convert lists to arrays and report
    results = {}
    for size in unique_sizes:
        if nm_windows[size]:
            results[size] = {
                'windows': np.array(nm_windows[size]),  # (n_events, n_freqs, n_times)
                'window_times': nm_window_times,
                'valid_events': np.array(valid_events[size]),
                'n_events': len(nm_windows[size])
            }
            print(f"NM size {size}: {len(nm_windows[size])} valid events")
        else:
            print(f"NM size {size}: No valid events found")
            
    return results


def normalize_windows(nm_windows: Dict, 
                     global_mean: np.ndarray, 
                     global_std: np.ndarray) -> Dict:
    """
    Normalize (z-score) the extracted windows using global statistics.
    
    Parameters:
    -----------
    nm_windows : Dict
        Raw NM windows by size
    global_mean : np.ndarray
        Global mean per frequency (n_freqs,)
    global_std : np.ndarray
        Global standard deviation per frequency (n_freqs,)
    
    Returns:
    --------
    normalized_windows : Dict
        Z-scored windows by size
    """
    print("Normalizing windows using global statistics...")
    
    normalized_windows = {}
    
    for size, data in nm_windows.items():
        windows = data['windows']  # (n_events, n_freqs, n_times)
        
        # Z-score normalization: (x - mean) / std for each frequency
        # Reshape global stats to broadcast correctly with windows
        mean_expanded = global_mean[np.newaxis, :, np.newaxis]  # (1, n_freqs, 1)
        std_expanded = global_std[np.newaxis, :, np.newaxis]    # (1, n_freqs, 1)
        
        normalized = (windows - mean_expanded) / std_expanded
        
        normalized_windows[size] = {
            'windows': normalized,
            'window_times': data['window_times'],
            'valid_events': data['valid_events'],
            'n_events': data['n_events']
        }
        
        print(f"NM size {size}: normalized {data['n_events']} windows")
        print(f"  Z-score range: {normalized.min():.2f} to {normalized.max():.2f}")
    
    return normalized_windows


def save_results(session_data: Dict,
                normalized_windows: Dict,
                freqs: np.ndarray,
                global_mean: np.ndarray,
                global_std: np.ndarray,
                channel_idx: int,
                save_path: str):
    """
    Save the analysis results with metadata.
    
    Parameters:
    -----------
    session_data : Dict
        Original session data
    normalized_windows : Dict
        Normalized NM windows
    freqs : np.ndarray
        Frequency vector
    global_mean : np.ndarray
        Global mean per frequency
    global_std : np.ndarray
        Global standard deviation per frequency
    channel_idx : int
        Index of analyzed channel
    save_path : str
        Directory to save results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'session_metadata': {
            'rat_id': session_data.get('rat_id', 'unknown'),
            'session_date': session_data.get('session_date', 'unknown'),
            'eeg_shape': session_data['eeg'].shape,
            'channel_analyzed': channel_idx,
            'total_nm_events': len(session_data['nm_peak_times']),
            'nm_sizes_available': np.unique(session_data['nm_sizes']).tolist()
        },
        'analysis_parameters': {
            'frequency_range': (freqs[0], freqs[-1]),
            'frequency_step': freqs[1] - freqs[0] if len(freqs) > 1 else 1.0,
            'n_frequencies': len(freqs),
            'window_duration': (normalized_windows[list(normalized_windows.keys())[0]]['window_times'][-1] - 
                              normalized_windows[list(normalized_windows.keys())[0]]['window_times'][0]) if normalized_windows else 1.0,
            'normalization': 'z-score using global mean/std per frequency'
        },
        'frequencies': freqs,
        'global_statistics': {
            'mean': global_mean,
            'std': global_std
        },
        'nm_windows': normalized_windows
    }
    
    # Save main results
    results_file = os.path.join(save_path, 'nm_theta_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary for quick inspection
    summary = {
        'session': f"{results['session_metadata']['rat_id']}_{results['session_metadata']['session_date']}",
        'channel': channel_idx,
        'frequency_range': f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz",
        'nm_sizes_analyzed': list(normalized_windows.keys()),
        'events_per_size': {size: data['n_events'] for size, data in normalized_windows.items()},
        'total_events_analyzed': sum(data['n_events'] for data in normalized_windows.values())
    }
    
    summary_file = os.path.join(save_path, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("NM Theta Analysis Summary\n")
        f.write("=" * 30 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def plot_nm_theta_results(normalized_windows: Dict,
                         freqs: np.ndarray,
                         save_path: str = None):
    """
    Plot average z-scored spectrograms for each NM size.
    
    Parameters:
    -----------
    normalized_windows : Dict
        Normalized NM windows by size
    freqs : np.ndarray
        Frequency vector
    save_path : str, optional
        Path to save plots
    """
    print("Generating plots...")
    
    nm_sizes = sorted(normalized_windows.keys())
    n_sizes = len(nm_sizes)
    
    if n_sizes == 0:
        print("No NM windows to plot!")
        return
    
    # Create figure with subplots for each NM size
    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 6))
    if n_sizes == 1:
        axes = [axes]
    
    fig.suptitle('Average Z-scored Theta Oscillations Around NM Events', 
                 fontsize=16, fontweight='bold')
    
    # Color map and normalization for consistent scaling, centered at 0
    all_spectrograms = [np.mean(data['windows'], axis=0) for data in normalized_windows.values()]
    vmin = min(spec.min() for spec in all_spectrograms)
    vmax = max(spec.max() for spec in all_spectrograms)
    # Center colormap at 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    for i, size in enumerate(nm_sizes):
        data = normalized_windows[size]
        
        # Compute average across events
        avg_spectrogram = np.mean(data['windows'], axis=0)  # (n_freqs, n_times)
        window_times = data['window_times']
        n_events = data['n_events']
        
        # Plot heatmap with diverging colormap centered at 0
        im = axes[i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        
        # Add event line
        axes[i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        # Set y-axis ticks to show actual frequency values
        # Show every 5th frequency or max 10 ticks to avoid overcrowding
        freq_step = max(1, len(freqs) // 10)
        freq_ticks = freqs[::freq_step]
        axes[i].set_yticks(freq_ticks)
        axes[i].set_yticklabels([f'{f:.1f}' for f in freq_ticks])
        
        # Formatting
        axes[i].set_title(f'NM Size {size}\n(n={n_events} events)', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Frequency (Hz)', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('Z-score', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plot_file = os.path.join(save_path, 'nm_theta_spectrograms.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
    
    plt.show()
    
    # Additional plot: Frequency profiles at event time
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Find time index closest to event (t=0)
    for size in nm_sizes:
        data = normalized_windows[size]
        window_times = data['window_times']
        event_idx = np.argmin(np.abs(window_times))
        
        avg_spectrogram = np.mean(data['windows'], axis=0)
        freq_profile = avg_spectrogram[:, event_idx]
        
        ax.plot(freqs, freq_profile, 'o-', linewidth=2, markersize=6,
                label=f'NM Size {size} (n={data["n_events"]})', alpha=0.8)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Z-score at Event Time', fontsize=12)
    ax.set_title('Frequency Profiles at NM Event Time (t=0)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        profile_file = os.path.join(save_path, 'nm_frequency_profiles.png')
        plt.savefig(profile_file, dpi=300, bbox_inches='tight')
        print(f"Frequency profile plot saved: {profile_file}")
    
    plt.show()


def analyze_session_nm_theta_roi(session_data: Dict,
                               roi_or_channels: Union[str, List[int]],
                               freq_range: Tuple[float, float] = (3, 8),
                               n_freqs: int = 20,
                               window_duration: float = 1.0,
                               n_cycles_factor: float = 3.0,
                               save_path: str = 'nm_theta_results',
                               mapping_df: Optional[pd.DataFrame] = None,
                               show_plots: bool = True,
                               show_frequency_profiles: bool = False) -> Dict:
    """
    Complete NM theta analysis for a ROI in a single session.
    
    Parameters:
    -----------
    session_data : Dict
        Session data dictionary
    roi_or_channels : Union[str, List[int]]
        Either ROI name (e.g., 'frontal', 'hippocampus') or list of channel numbers (1-32, as in electrode mappings)
    freq_range : Tuple[float, float]
        Frequency range for analysis (default: 3-8 Hz)
    n_freqs : int
        Number of logarithmically spaced frequencies (default: 20)
    window_duration : float
        Event window duration (default: 1.0s = ±0.5s)
    n_cycles_factor : float
        Factor for adaptive n_cycles (default: 3.0)
    save_path : str
        Directory to save results
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe. If None, loads from default CSV
    show_plots : bool
        Whether to display plots (default: True)
    show_frequency_profiles : bool
        Whether to display ROI frequency profiles plot (default: False)
    
    Returns:
    --------
    results : Dict
        Complete analysis results
    """
    print("=" * 60)
    print("NM THETA ROI ANALYSIS")
    print("=" * 60)
    
    if len(session_data['nm_peak_times']) == 0:
        raise ValueError("No NM events found in session")
    
    # Step 1: Get ROI channel indices
    print(f"Step 1: Determining ROI channels")
    rat_id = session_data.get('rat_id', 'unknown')
    
    # Always use get_channels for proper 1-32 to 0-31 conversion
    if mapping_df is None:
        mapping_df = load_electrode_mappings()
    
    roi_channels = get_channels(rat_id, roi_or_channels, mapping_df)
    
    # MAPPING VERIFICATION OUTPUTS
    print("=" * 60)
    print("🔍 ELECTRODE MAPPING VERIFICATION")
    print("=" * 60)
    
    if isinstance(roi_or_channels, str):
        # Show the complete mapping chain for ROI
        electrode_numbers = ROI_MAP.get(roi_or_channels, [])
        print(f"✓ ROI specification: '{roi_or_channels}'")
        print(f"✓ ROI_MAP['{roi_or_channels}'] = {electrode_numbers}")
        print(f"✓ Rat {rat_id} electrode mapping:")
        
        # Show which electrodes map to which channels
        # Handle rat_id type conversion (same logic as in electrode_utils.py)
        lookup_rat_id = rat_id
        if rat_id not in mapping_df.index:
            # Try converting to int if it's a string
            if isinstance(rat_id, str) and rat_id.isdigit():
                lookup_rat_id = int(rat_id)
            # Try converting to string if it's an int  
            elif isinstance(rat_id, int):
                rat_id_str = str(rat_id)
                if rat_id_str in mapping_df.index:
                    lookup_rat_id = rat_id_str
        
        rat_mapping = mapping_df.loc[lookup_rat_id].values
        for electrode in electrode_numbers:
            channel_idx = np.where(rat_mapping == electrode)[0]
            if len(channel_idx) > 0:
                print(f"   Electrode {electrode:2d} -> Channel index {channel_idx[0]:2d}")
        
        print(f"✓ Final channel indices: {sorted(roi_channels)}")
        print(f"✓ EEG data access: eeg_data[{sorted(roi_channels)}, :]")
    else:
        print(f"✓ Custom channel specification: {roi_or_channels}")
        print(f"✓ Resulting channel indices: {sorted(roi_channels)}")
    
    print("=" * 60)
    
    # Validate channel indices
    max_channel = session_data['eeg'].shape[0] - 1
    invalid_channels = [ch for ch in roi_channels if ch > max_channel or ch < 0]
    if invalid_channels:
        raise ValueError(f"Invalid channel indices {invalid_channels}. Valid range: 0-{max_channel}")
    
    if not roi_channels:
        raise ValueError("No valid channels found for ROI")
    
    times = session_data['eeg_time'].flatten()
    
    # Step 2: Compute ROI theta spectrogram
    print(f"Step 2: Computing ROI theta spectrogram ({freq_range[0]}-{freq_range[1]} Hz)")
    freqs, roi_power, channel_powers = compute_roi_theta_spectrogram(
        session_data['eeg'],
        roi_channels,
        sfreq=200.0,
        freq_range=freq_range,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor
    )
    
    # Step 3: ROI power is already normalized per channel, no global stats needed
    print("Step 3: Using per-channel normalized ROI power")
    # For compatibility with existing functions, create dummy global stats
    global_mean = np.zeros(len(freqs))
    global_std = np.ones(len(freqs))
    
    # Step 4: Extract NM event windows from ROI power
    print("Step 4: Extracting NM event windows")
    nm_windows = extract_nm_event_windows(
        roi_power, times, 
        session_data['nm_peak_times'],
        session_data['nm_sizes'],
        window_duration
    )
    
    if not nm_windows:
        raise ValueError("No valid NM event windows extracted")
    
    # Step 5: Since ROI power is already normalized, we skip additional normalization
    print("Step 5: ROI windows already normalized per channel")
    # For compatibility, use the windows as "normalized"
    normalized_windows = nm_windows
    
    # Step 6: Save results
    print("Step 6: Saving results")
    save_roi_results(
        session_data, normalized_windows, freqs, 
        roi_channels, roi_or_channels, channel_powers, save_path
    )
    
    # Step 7: Generate plots
    if show_plots:
        print("Step 7: Generating plots")
        plot_roi_theta_results(
            normalized_windows, 
            freqs, 
            roi_channels, 
            save_path, 
            rat_id=rat_id, 
            mapping_df=mapping_df,
            show_frequency_profiles=show_frequency_profiles
        )
    else:
        print("Step 7: Skipping plots (show_plots=False)")
    
    print("=" * 60)
    print("ROI ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'freqs': freqs,
        'roi_power': roi_power,
        'channel_powers': channel_powers,
        'roi_channels': roi_channels,
        'normalized_windows': normalized_windows,
        'roi_specification': roi_or_channels
    }


def save_roi_results(session_data: Dict,
                   normalized_windows: Dict,
                   freqs: np.ndarray,
                   roi_channels: List[int],
                   roi_specification: Union[str, List[int]],
                   channel_powers: np.ndarray,
                   save_path: str):
    """
    Save the ROI analysis results with metadata.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'session_metadata': {
            'rat_id': session_data.get('rat_id', 'unknown'),
            'session_date': session_data.get('session_date', 'unknown'),
            'eeg_shape': session_data['eeg'].shape,
            'roi_specification': roi_specification,
            'roi_channels': roi_channels,
            'n_roi_channels': len(roi_channels),
            'total_nm_events': len(session_data['nm_peak_times']),
            'nm_sizes_available': np.unique(session_data['nm_sizes']).tolist()
        },
        'analysis_parameters': {
            'frequency_range': (freqs[0], freqs[-1]),
            'frequency_step': freqs[1] - freqs[0] if len(freqs) > 1 else 1.0,
            'n_frequencies': len(freqs),
            'window_duration': (normalized_windows[list(normalized_windows.keys())[0]]['window_times'][-1] - 
                              normalized_windows[list(normalized_windows.keys())[0]]['window_times'][0]) if normalized_windows else 1.0,
            'normalization': 'per-channel z-score, then averaged across ROI channels'
        },
        'frequencies': freqs,
        'roi_data': {
            'channels': roi_channels,
            'specification': roi_specification,
            'channel_powers': channel_powers  # Individual channel powers for SEM computation
        },
        'nm_windows': normalized_windows
    }
    
    # Save main results
    results_file = os.path.join(save_path, 'nm_roi_theta_analysis_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = {
        'session': f"{results['session_metadata']['rat_id']}_{results['session_metadata']['session_date']}",
        'roi_specification': roi_specification,
        'roi_channels': roi_channels,
        'n_channels': len(roi_channels),
        'frequency_range': f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz",
        'nm_sizes_analyzed': list(normalized_windows.keys()),
        'events_per_size': {size: data['n_events'] for size, data in normalized_windows.items()},
        'total_events_analyzed': sum(data['n_events'] for data in normalized_windows.values())
    }
    
    summary_file = os.path.join(save_path, 'roi_analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("NM ROI Theta Analysis Summary\n")
        f.write("=" * 35 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"ROI results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def plot_roi_theta_results(normalized_windows: Dict,
                         freqs: np.ndarray,
                         roi_channels: List[int],
                         save_path: str = None,
                         rat_id: Union[str, int] = None,
                         mapping_df: pd.DataFrame = None,
                         show_frequency_profiles: bool = False):
    """
    Plot average z-scored spectrograms for each NM size (ROI version).
    """
    print("Generating ROI plots...")
    
    nm_sizes = sorted(normalized_windows.keys())
    n_sizes = len(nm_sizes)
    
    if n_sizes == 0:
        print("No NM windows to plot!")
        return
    
    # Create figure with subplots for each NM size
    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 6))
    if n_sizes == 1:
        axes = [axes]
    
    # Get electrode numbers for display
    electrode_display = roi_channels  # Default to channel indices
    if rat_id is not None and mapping_df is not None:
        try:
            electrode_numbers = get_electrode_numbers_from_channels(rat_id, roi_channels, mapping_df)
            electrode_display = electrode_numbers
            electrode_label = "electrodes"
        except Exception as e:
            print(f"Warning: Could not convert channels to electrodes: {e}")
            electrode_label = "channels"
    else:
        electrode_label = "channels"
    
    fig.suptitle(f'Average Z-scored Theta Oscillations Around NM Events\nROI: {len(roi_channels)} {electrode_label} {electrode_display}', 
                 fontsize=16, fontweight='bold')
    
    # Color map and normalization for consistent scaling, centered at 0
    all_spectrograms = [np.mean(data['windows'], axis=0) for data in normalized_windows.values()]
    vmin = min(spec.min() for spec in all_spectrograms)
    vmax = max(spec.max() for spec in all_spectrograms)
    # Center colormap at 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    # Helper: choose indices to label (first, last, and a few intermediates)
    def get_label_indices(n, n_labels=8):
        if n <= n_labels:
            return list(range(n))
        idxs = [0]
        step = (n - 1) / (n_labels - 1)
        for i in range(1, n_labels-1):
            idxs.append(int(round(i * step)))
        idxs.append(n-1)
        return sorted(set(idxs))
    
    label_indices = get_label_indices(len(freqs), n_labels=5)
    
    for i, size in enumerate(nm_sizes):
        data = normalized_windows[size]
        
        # Compute average across events
        avg_spectrogram = np.mean(data['windows'], axis=0)  # (n_freqs, n_times)
        window_times = data['window_times']
        n_events = data['n_events']
        
        # Plot heatmap with diverging colormap centered at 0
        im = axes[i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        
        # Add event line
        axes[i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        # Set y-axis ticks to all frequency positions, label only selected
        freq_ticks = freqs
        freq_labels = [''] * len(freq_ticks)
        for idx in label_indices:
            freq_labels[idx] = f'{freq_ticks[idx]:.1f}'
        axes[i].set_yticks(freq_ticks)
        axes[i].set_yticklabels(freq_labels)
        
        # Formatting
        axes[i].set_title(f'NM Size {size}\n(n={n_events} events)', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time (s)', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('Frequency (Hz)', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('Z-score', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plot_file = os.path.join(save_path, 'nm_roi_theta_spectrograms.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"ROI plot saved: {plot_file}")
    
    plt.show()
    
    # Additional plot: Frequency profiles at event time (optional)
    if show_frequency_profiles:
        print("Generating frequency profiles plot...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Find time index closest to event (t=0)
        for size in nm_sizes:
            data = normalized_windows[size]
            window_times = data['window_times']
            event_idx = np.argmin(np.abs(window_times))
            
            avg_spectrogram = np.mean(data['windows'], axis=0)
            freq_profile = avg_spectrogram[:, event_idx]
            
            ax.plot(freqs, freq_profile, 'o-', linewidth=2, markersize=6,
                    label=f'NM Size {size} (n={data["n_events"]})', alpha=0.8)
        
        # Set x-axis ticks to all frequency positions, label only selected
        freq_ticks = freqs
        freq_labels = [''] * len(freq_ticks)
        for idx in label_indices:
            freq_labels[idx] = f'{freq_ticks[idx]:.1f}'
        ax.set_xticks(freq_ticks)
        ax.set_xticklabels(freq_labels)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Z-score at Event Time', fontsize=12)
        ax.set_title(f'ROI Frequency Profiles at NM Event Time (t=0)\n{electrode_label.capitalize()}: {electrode_display}', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            profile_file = os.path.join(save_path, 'nm_roi_frequency_profiles.png')
            plt.savefig(profile_file, dpi=300, bbox_inches='tight')
            print(f"ROI frequency profile plot saved: {profile_file}")
        
        plt.show()
    else:
        print("Skipping frequency profiles plot (show_frequency_profiles=False)")


# Keep the original single-channel function for backward compatibility
def analyze_session_nm_theta(session_data: Dict,
                           channel_of_interest: int,
                           freq_range: Tuple[float, float] = (3, 8),
                           freq_step: float = 1.0,
                           window_duration: float = 1.0,
                           n_cycles_factor: float = 3.0,
                           save_path: str = 'nm_theta_results',
                           show_plots: bool = True) -> Dict:
    """
    Complete NM theta analysis for a single session (single channel - DEPRECATED).
    Use analyze_session_nm_theta_roi for ROI-based analysis.
    """
    print("WARNING: Using deprecated single-channel analysis. Consider using analyze_session_nm_theta_roi instead.")
    
    # Convert single channel to channel list and use ROI function
    return analyze_session_nm_theta_roi(
        session_data=session_data,
        roi_or_channels=[channel_of_interest],
        freq_range=freq_range,
        freq_step=freq_step,
        window_duration=window_duration,
        n_cycles_factor=n_cycles_factor,
        save_path=save_path,
        show_plots=show_plots
    )


def main():
    """
    Main function to run NM ROI theta analysis on a sample session.
    """
    try:
        # Load session data
        print("Loading EEG session data...")
        session_data = load_session_data('data/processed/all_eeg_data.pkl', session_index=0)
        
        # Analysis parameters
        roi_specification = [2]
        freq_range = (1, 45)       # Extended theta range
        n_freqs = 40               # 25 log-spaced frequencies across 2-10 Hz
        window_duration = 1      # ±0.5s around events
        n_cycles_factor = 3.0      # For good frequency resolution
        
        # Run ROI analysis
        results = analyze_session_nm_theta_roi(
            session_data=session_data,
            roi_or_channels=roi_specification, #or just a list of channel numbers like [10, 11, 12]
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path='nm_roi_theta_results',
            show_frequency_profiles=True  # Set to True to show frequency profiles plot
        )
        
        # Print summary
        print("\nROI Analysis Summary:")
        print(f"- ROI specification: {roi_specification}")
        print(f"- ROI channels: {results['roi_channels']}")
        print(f"- Number of channels: {len(results['roi_channels'])}")
        print(f"- Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"- NM sizes found: {[float(key) for key in results['normalized_windows'].keys()]}")
        print(f"- Total events analyzed: {sum(data['n_events'] for data in results['normalized_windows'].values())}")
        
        return True
        
    except Exception as e:
        print(f"Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)