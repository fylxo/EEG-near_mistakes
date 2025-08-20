#!/usr/bin/env python3
"""
Baseline Normalization for NM Theta Analysis

This module implements baseline normalization for NM theta analysis, where instead of 
using global statistics (mean/std across entire recording), we use baseline statistics 
from -1.0 to -0.5 seconds before each event.

The key functions are:
- compute_baseline_statistics(): Extract baseline period and compute statistics
- normalize_windows_baseline(): Apply baseline z-score normalization  
- extract_nm_event_windows_with_baseline(): Extract windows with baseline period

This provides event-specific normalization that is more sensitive to changes
relative to the immediate pre-event baseline.

Baseline normalization implementation for EEG near-mistake analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_baseline_statistics(nm_windows: Dict, 
                               window_times: np.ndarray,
                               baseline_start: float = -1.0, 
                               baseline_end: float = -0.5) -> Dict:
    """
    Compute baseline statistics from a specific time period for each event.
    
    This function extracts the baseline period (-1.0 to -0.5 seconds by default)
    from each event window and computes mean and standard deviation for each
    frequency band. These statistics are then used for event-specific normalization.
    
    Parameters:
    -----------
    nm_windows : Dict
        Raw NM windows by size, each containing:
        - 'windows': np.ndarray (n_events, n_freqs, n_times)
        - 'window_times': np.ndarray (n_times,)
        - Other metadata
    window_times : np.ndarray
        Time vector for the windows (e.g., -1.0 to +1.0 seconds)
    baseline_start : float
        Start time of baseline period (default: -1.0s)
    baseline_end : float
        End time of baseline period (default: -0.5s)
    
    Returns:
    --------
    baseline_stats : Dict
        Dictionary containing baseline statistics for each NM size:
        - 'baseline_mean': np.ndarray (n_events, n_freqs)
        - 'baseline_std': np.ndarray (n_events, n_freqs)
        - 'baseline_time_indices': np.ndarray (n_baseline_samples,)
        - 'n_baseline_samples': int
    """
    print(f"Computing baseline statistics for period [{baseline_start:.1f}, {baseline_end:.1f}]s...")
    
    # Find baseline time indices
    baseline_mask = (window_times >= baseline_start) & (window_times <= baseline_end)
    baseline_indices = np.where(baseline_mask)[0]
    
    if len(baseline_indices) == 0:
        raise ValueError(f"No time points found in baseline period [{baseline_start}, {baseline_end}]s")
    
    n_baseline_samples = len(baseline_indices)
    print(f"Found {n_baseline_samples} time points in baseline period")
    print(f"Baseline time range: {window_times[baseline_indices[0]]:.3f} to {window_times[baseline_indices[-1]]:.3f}s")
    
    baseline_stats = {}
    
    for size, data in nm_windows.items():
        windows = data['windows']  # (n_events, n_freqs, n_times)
        n_events, n_freqs, n_times = windows.shape
        
        print(f"NM size {size}: {n_events} events, {n_freqs} frequencies")
        
        # Extract baseline period for all events
        baseline_windows = windows[:, :, baseline_indices]  # (n_events, n_freqs, n_baseline_samples)
        
        # PROPER Z-SCORE METHOD: Average each baseline first, then compute stats across events
        # Step 1: Average each event's baseline period (removes temporal oscillations)
        baseline_averages = np.mean(baseline_windows, axis=2)  # (n_events, n_freqs) - single value per event/freq
        
        # Step 2: Compute statistics across events (proper z-score denominator)
        baseline_mean = np.mean(baseline_averages, axis=0)  # (n_freqs,) - mean across events  
        baseline_std = np.std(baseline_averages, axis=0)    # (n_freqs,) - std across events
        
        # Ensure no zero standard deviations (would cause division by zero)
        baseline_std = np.maximum(baseline_std, 1e-12)
        
        
        # Store statistics
        baseline_stats[size] = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_time_indices': baseline_indices,
            'n_baseline_samples': n_baseline_samples,
            'baseline_period': [baseline_start, baseline_end],
            'n_events': n_events,
            'n_freqs': n_freqs
        }
        
        print(f"  Baseline mean range: {baseline_mean.min():.2e} - {baseline_mean.max():.2e}")
        print(f"  Baseline std range: {baseline_std.min():.2e} - {baseline_std.max():.2e}")
        
        # ADD: Validate minimum events for stable statistics
        MIN_EVENTS = 3
        if n_events < MIN_EVENTS:
            warning_msg = f"NM size {size} has only {n_events} events (< {MIN_EVENTS}). Statistics may be unstable."
            print(f"⚠️  WARNING: {warning_msg}")
            print(f"   Consider combining with other NM sizes or excluding from analysis.")
            
            # Add warning to the stats dictionary
            baseline_stats[size]['warnings'] = baseline_stats[size].get('warnings', [])
            baseline_stats[size]['warnings'].append(warning_msg)
        
        # ADD: Additional validation for very small standard deviations
        if baseline_std.min() < 1e-6:
            warning_msg = f"NM size {size} has very small baseline variability (min std: {baseline_std.min():.2e}). This may indicate insufficient data variation."
            print(f"⚠️  WARNING: {warning_msg}")
            
            baseline_stats[size]['warnings'] = baseline_stats[size].get('warnings', [])
            baseline_stats[size]['warnings'].append(warning_msg)
    
    return baseline_stats


def normalize_windows_baseline(nm_windows: Dict, 
                              baseline_stats: Dict) -> Dict:
    """
    Normalize event windows using baseline statistics (event-specific z-score).
    
    This function applies z-score normalization to each event window using the
    baseline statistics computed for that specific event. This provides more
    sensitive normalization compared to global statistics.
    
    Parameters:
    -----------
    nm_windows : Dict
        Raw NM windows by size
    baseline_stats : Dict
        Baseline statistics from compute_baseline_statistics()
    
    Returns:
    --------
    normalized_windows : Dict
        Baseline-normalized windows by size
    """
    print("Normalizing windows using baseline statistics...")
    
    normalized_windows = {}
    
    for size, data in nm_windows.items():
        windows = data['windows']  # (n_events, n_freqs, n_times)
        
        if size not in baseline_stats:
            raise ValueError(f"No baseline statistics found for NM size {size}")
        
        baseline_mean = baseline_stats[size]['baseline_mean']  # (n_freqs,) - cross-event mean
        baseline_std = baseline_stats[size]['baseline_std']    # (n_freqs,) - cross-event std
        
        # Expand dimensions to broadcast with windows (n_events, n_freqs, n_times)
        mean_expanded = baseline_mean[np.newaxis, :, np.newaxis]  # (1, n_freqs, 1)
        std_expanded = baseline_std[np.newaxis, :, np.newaxis]    # (1, n_freqs, 1)
        
        # Apply proper z-score normalization: (x - cross_event_baseline_mean) / cross_event_baseline_std
        normalized = (windows - mean_expanded) / std_expanded
        
        normalized_windows[size] = {
            'windows': normalized,
            'window_times': data['window_times'],
            'valid_events': data['valid_events'],
            'n_events': data['n_events'],
            'baseline_stats': baseline_stats[size]  # Include baseline stats for reference
        }
        
        print(f"NM size {size}: normalized {data['n_events']} windows using baseline statistics")
        print(f"  Z-score range: {normalized.min():.2f} to {normalized.max():.2f}")
    
    return normalized_windows


def extract_nm_event_windows_with_baseline(power: np.ndarray,
                                          times: np.ndarray,
                                          nm_peak_times: np.ndarray,
                                          nm_sizes: np.ndarray,
                                          window_duration: float = 2.0,
                                          baseline_start: float = -1.0,
                                          baseline_end: float = -0.5) -> Dict:
    """
    Extract time-frequency windows around NM events with baseline normalization.
    
    This function extracts event windows (default: -1.0 to +1.0 seconds) and
    applies baseline normalization using statistics from the baseline period
    (default: -1.0 to -0.5 seconds).
    
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
    baseline_start : float
        Start time of baseline period (default: -1.0s)
    baseline_end : float
        End time of baseline period (default: -0.5s)
    
    Returns:
    --------
    results : Dict
        Dictionary containing:
        - 'normalized_windows': Dict with baseline-normalized windows by NM size
        - 'baseline_stats': Dict with baseline statistics
        - 'raw_windows': Dict with raw (unnormalized) windows
    """
    print(f"Extracting NM event windows with baseline normalization...")
    print(f"Window duration: ±{window_duration/2:.1f}s around events")
    print(f"Baseline period: [{baseline_start:.1f}, {baseline_end:.1f}]s")
    
    # Validate baseline period is within window
    half_window = window_duration / 2
    if baseline_start < -half_window or baseline_end > half_window:
        raise ValueError(f"Baseline period [{baseline_start}, {baseline_end}] must be within window [±{half_window}]")
    
    # Extract raw windows first (using existing logic)
    raw_windows = _extract_raw_windows(power, times, nm_peak_times, nm_sizes, window_duration)
    
    if not raw_windows:
        print("No valid windows extracted")
        return {'normalized_windows': {}, 'baseline_stats': {}, 'raw_windows': {}}
    
    # Get window times from first window (all should be the same)
    window_times = list(raw_windows.values())[0]['window_times']
    
    # Compute baseline statistics
    baseline_stats = compute_baseline_statistics(raw_windows, window_times, baseline_start, baseline_end)
    
    # Apply baseline normalization
    normalized_windows = normalize_windows_baseline(raw_windows, baseline_stats)
    
    return {
        'normalized_windows': normalized_windows,
        'baseline_stats': baseline_stats,
        'raw_windows': raw_windows
    }


def _extract_raw_windows(power: np.ndarray,
                        times: np.ndarray,
                        nm_peak_times: np.ndarray,
                        nm_sizes: np.ndarray,
                        window_duration: float = 2.0) -> Dict:
    """
    Extract raw (unnormalized) time-frequency windows around NM events.
    
    This is a helper function that implements the core window extraction logic
    similar to the existing extract_nm_event_windows() function.
    
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
        Dictionary with raw windows organized by NM size
    """
    half_window = window_duration / 2
    
    # Calculate actual sampling frequency
    time_diffs = np.diff(times)
    actual_sfreq = 1 / np.median(time_diffs)
    
    # Use exactly 200 Hz for consistency (matching existing code)
    sfreq = 200.0
    window_samples = int(window_duration * sfreq)
    
    # Warn if there's a significant difference from actual frequency
    if abs(actual_sfreq - sfreq) > 1.0:
        print(f"Warning: Actual sampling frequency {actual_sfreq:.2f} Hz differs from standard {sfreq} Hz")
        print(f"Using standard {sfreq} Hz for cross-rat consistency")
    
    print(f"Actual sampling frequency: {actual_sfreq:.2f} Hz → Using {sfreq} Hz (window size: {window_samples} samples)")
    
    # Initialize storage
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
                    print(f"  Tolerance exceeded: {difference} > {max_tolerance} samples")
            else:
                print(f"Warning: NM event {i} at {event_time:.2f}s too close to recording end")
        else:
            print(f"Warning: NM event {i} at {event_time:.2f}s outside recording bounds")
    
    # Convert to final format
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


def compare_normalization_methods(nm_windows: Dict,
                                 window_times: np.ndarray,
                                 global_mean: np.ndarray,
                                 global_std: np.ndarray,
                                 baseline_start: float = -1.0,
                                 baseline_end: float = -0.5) -> Dict:
    """
    Compare global vs baseline normalization methods.
    
    This function applies both global and baseline normalization to the same
    windows and returns both results for comparison.
    
    Parameters:
    -----------
    nm_windows : Dict
        Raw NM windows by size
    window_times : np.ndarray
        Time vector for the windows
    global_mean : np.ndarray
        Global mean per frequency (n_freqs,)
    global_std : np.ndarray
        Global standard deviation per frequency (n_freqs,)
    baseline_start : float
        Start time of baseline period (default: -1.0s)
    baseline_end : float
        End time of baseline period (default: -0.5s)
    
    Returns:
    --------
    comparison : Dict
        Dictionary containing:
        - 'global_normalized': Dict with globally normalized windows
        - 'baseline_normalized': Dict with baseline normalized windows
        - 'baseline_stats': Dict with baseline statistics
        - 'global_stats': Dict with global statistics
    """
    print("Comparing global vs baseline normalization methods...")
    
    # Apply global normalization (existing method)
    global_normalized = {}
    for size, data in nm_windows.items():
        windows = data['windows']
        mean_expanded = global_mean[np.newaxis, :, np.newaxis]
        std_expanded = global_std[np.newaxis, :, np.newaxis]
        normalized = (windows - mean_expanded) / std_expanded
        
        global_normalized[size] = {
            'windows': normalized,
            'window_times': data['window_times'],
            'valid_events': data['valid_events'],
            'n_events': data['n_events']
        }
    
    # Apply baseline normalization
    baseline_stats = compute_baseline_statistics(nm_windows, window_times, baseline_start, baseline_end)
    baseline_normalized = normalize_windows_baseline(nm_windows, baseline_stats)
    
    # Compute comparison statistics
    comparison_stats = {}
    for size in nm_windows.keys():
        if size in global_normalized and size in baseline_normalized:
            global_values = global_normalized[size]['windows']
            baseline_values = baseline_normalized[size]['windows']
            
            comparison_stats[size] = {
                'global_range': [float(global_values.min()), float(global_values.max())],
                'baseline_range': [float(baseline_values.min()), float(baseline_values.max())],
                'global_mean': float(global_values.mean()),
                'baseline_mean': float(baseline_values.mean()),
                'global_std': float(global_values.std()),
                'baseline_std': float(baseline_values.std()),
                'correlation': float(np.corrcoef(global_values.flatten(), baseline_values.flatten())[0, 1])
            }
    
    return {
        'global_normalized': global_normalized,
        'baseline_normalized': baseline_normalized,
        'baseline_stats': baseline_stats,
        'global_stats': {'global_mean': global_mean, 'global_std': global_std},
        'comparison_stats': comparison_stats
    }


def save_baseline_results(results: Dict, 
                         freqs: np.ndarray,
                         session_data: Dict,
                         roi_channels: List[int],
                         save_path: str,
                         analysis_params: Dict = None):
    """
    Save baseline normalization results with comprehensive metadata.
    
    Parameters:
    -----------
    results : Dict
        Results from extract_nm_event_windows_with_baseline()
    freqs : np.ndarray
        Frequency vector
    session_data : Dict
        Original session data
    roi_channels : List[int]
        Channel indices that were analyzed
    save_path : str
        Directory to save results
    analysis_params : Dict, optional
        Additional analysis parameters
    """
    import pickle
    import json
    from datetime import datetime
    
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare comprehensive results
    save_data = {
        'analysis_type': 'baseline_normalized_nm_theta',
        'timestamp': datetime.now().isoformat(),
        'session_metadata': {
            'rat_id': session_data.get('rat_id', 'unknown'),
            'session_date': session_data.get('session_date', 'unknown'),
            'eeg_shape': session_data['eeg'].shape,
            'roi_channels': roi_channels,
            'total_nm_events': len(session_data['nm_peak_times']),
            'nm_sizes_available': np.unique(session_data['nm_sizes']).tolist()
        },
        'frequency_info': {
            'frequencies': freqs.tolist(),
            'n_freqs': len(freqs),
            'freq_range': [float(freqs.min()), float(freqs.max())]
        },
        'normalized_windows': results['normalized_windows'],
        'baseline_stats': results['baseline_stats'],
        'raw_windows': results['raw_windows'],
        'analysis_parameters': analysis_params or {}
    }
    
    # Save main results
    results_file = os.path.join(save_path, 'baseline_normalized_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Save summary JSON
    summary_data = {
        'analysis_type': 'baseline_normalized_nm_theta',
        'rat_id': session_data.get('rat_id', 'unknown'),
        'session_date': session_data.get('session_date', 'unknown'),
        'roi_channels': roi_channels,
        'n_frequencies': len(freqs),
        'frequency_range': [float(freqs.min()), float(freqs.max())],
        'nm_sizes_analyzed': list(results['normalized_windows'].keys()),
        'total_events_by_size': {str(size): data['n_events'] 
                                for size, data in results['normalized_windows'].items()},
        'baseline_period': results['baseline_stats'][list(results['baseline_stats'].keys())[0]]['baseline_period']
        if results['baseline_stats'] else None,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(save_path, 'baseline_normalized_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✓ Baseline normalization results saved to: {results_file}")
    print(f"✓ Summary saved to: {summary_file}")


# Example usage and integration functions
def analyze_session_with_baseline_normalization(session_data: Dict,
                                               roi_or_channels: Union[str, List[int]],
                                               freq_range: Tuple[float, float] = (3, 8),
                                               n_freqs: int = 20,
                                               window_duration: float = 2.0,
                                               baseline_start: float = -1.0,
                                               baseline_end: float = -0.5,
                                               n_cycles_factor: float = 3.0,
                                               save_path: str = None,
                                               compare_with_global: bool = True) -> Dict:
    """
    Complete analysis pipeline with baseline normalization.
    
    This function provides a complete analysis pipeline that can be used as a
    drop-in replacement for existing analysis functions, but with baseline
    normalization instead of global normalization.
    
    Parameters:
    -----------
    session_data : Dict
        EEG session data
    roi_or_channels : Union[str, List[int]]
        ROI specification or channel list
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    baseline_start : float
        Baseline period start
    baseline_end : float
        Baseline period end
    n_cycles_factor : float
        Morlet wavelet cycles factor
    save_path : str, optional
        Path to save results
    compare_with_global : bool
        Whether to also compute global normalization for comparison
    
    Returns:
    --------
    results : Dict
        Complete analysis results with baseline normalization
    """
    # Import required functions locally to avoid circular imports
    from utils.electrode_utils import get_channels
    from implementations.nm_theta_single_basic import compute_roi_theta_spectrogram, compute_global_statistics
    
    print("🧠 Starting baseline-normalized NM theta analysis...")
    
    # Get ROI channels
    roi_channels = get_channels(session_data['rat_id'], roi_or_channels)
    print(f"Using ROI channels: {roi_channels}")
    
    # Compute spectrogram
    freqs, roi_power, _ = compute_roi_theta_spectrogram(
        session_data['eeg'],
        roi_channels,
        sfreq=200.0,
        freq_range=freq_range,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor
    )
    
    # Create time vector
    n_samples = session_data['eeg'].shape[1]
    times = np.arange(n_samples) / 200.0
    
    # Extract windows with baseline normalization
    results = extract_nm_event_windows_with_baseline(
        roi_power,
        times,
        session_data['nm_peak_times'],
        session_data['nm_sizes'],
        window_duration=window_duration,
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )
    
    # Add comparison with global normalization if requested
    if compare_with_global:
        print("Computing global normalization for comparison...")
        global_mean, global_std = compute_global_statistics(roi_power)
        
        window_times = list(results['raw_windows'].values())[0]['window_times']
        comparison = compare_normalization_methods(
            results['raw_windows'],
            window_times,
            global_mean,
            global_std,
            baseline_start,
            baseline_end
        )
        results['comparison'] = comparison
    
    # Add metadata
    results['analysis_metadata'] = {
        'roi_channels': roi_channels,
        'frequencies': freqs,
        'freq_range': freq_range,
        'n_freqs': n_freqs,
        'window_duration': window_duration,
        'baseline_period': [baseline_start, baseline_end],
        'n_cycles_factor': n_cycles_factor,
        'session_info': {
            'rat_id': session_data.get('rat_id', 'unknown'),
            'session_date': session_data.get('session_date', 'unknown'),
            'eeg_shape': session_data['eeg'].shape
        }
    }
    
    # Save results if path provided
    if save_path:
        save_baseline_results(
            results,
            freqs,
            session_data,
            roi_channels,
            save_path,
            results['analysis_metadata']
        )
    
    print("✅ Baseline-normalized analysis completed!")
    return results