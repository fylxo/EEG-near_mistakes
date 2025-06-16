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
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from scipy import signal
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing package
from eeg_analysis_package.time_frequency import morlet_spectrogram


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


def compute_high_res_theta_spectrogram(eeg_data: np.ndarray, 
                                     sfreq: float = 200.0,
                                     freq_range: Tuple[float, float] = (3, 8),
                                     freq_step: float = 1.0,
                                     n_cycles_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute high-resolution spectrogram focused on theta range (3-8 Hz).
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data for single channel (n_samples,)
    sfreq : float
        Sampling frequency in Hz
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    freq_step : float
        Frequency step in Hz
    n_cycles_factor : float
        Factor for determining number of cycles per frequency (higher = better freq resolution)
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    power : np.ndarray
        Time-frequency power matrix (n_freqs, n_times)
    """
    # Create frequency vector with specified resolution
    freqs = np.arange(freq_range[0], freq_range[1] + freq_step, freq_step)
    
    # For low frequencies, use more cycles for better frequency resolution
    # Use adaptive n_cycles: more cycles for lower frequencies
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    print(f"Computing high-resolution theta spectrogram...")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz, step: {freq_step} Hz")
    print(f"Number of frequencies: {len(freqs)}")
    print(f"Cycles per frequency: {n_cycles}")
    
    # Compute Morlet spectrogram with adaptive cycles
    try:
        # Use the existing morlet_spectrogram function with custom n_cycles
        # Note: We'll compute each frequency separately for adaptive n_cycles
        power_list = []
        
        for i, (freq, cycles) in enumerate(zip(freqs, n_cycles)):
            print(f"Processing frequency {freq:.1f} Hz ({i+1}/{len(freqs)}) with {cycles:.1f} cycles")
            
            # Compute single frequency
            single_freq = np.array([freq])
            _, single_power = morlet_spectrogram(
                eeg_data, 
                sfreq=sfreq, 
                freqs=single_freq, 
                n_cycles=int(cycles)
            )
            power_list.append(single_power[0, :])  # Extract single frequency
        
        power = np.array(power_list)
        
    except Exception as e:
        print(f"Error in spectrogram computation: {e}")
        # Fallback to fixed n_cycles
        print("Falling back to fixed n_cycles approach...")
        _, power = morlet_spectrogram(
            eeg_data, 
            sfreq=sfreq, 
            freqs=freqs, 
            n_cycles=7
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
    sfreq = 1 / np.median(np.diff(times))
    window_samples = int(window_duration * sfreq)
    
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
            end_idx = start_idx + window_samples
            
            if end_idx <= len(times):
                # Extract window
                window_power = power[:, start_idx:end_idx]
                
                if window_power.shape[1] == window_samples:
                    nm_windows[event_size].append(window_power)
                    valid_events[event_size].append(i)
                else:
                    print(f"Warning: NM event {i} at {event_time:.2f}s has insufficient data")
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
    
    # Color map and normalization for consistent scaling (moved after empty check)
    vmin = min(np.mean(data['windows'], axis=0).min() for data in normalized_windows.values())
    vmax = max(np.mean(data['windows'], axis=0).max() for data in normalized_windows.values())
    
    for i, size in enumerate(nm_sizes):
        data = normalized_windows[size]
        
        # Compute average across events
        avg_spectrogram = np.mean(data['windows'], axis=0)  # (n_freqs, n_times)
        window_times = data['window_times']
        n_events = data['n_events']
        
        # Plot heatmap
        im = axes[i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        
        # Add event line
        axes[i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
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


def analyze_session_nm_theta(session_data: Dict,
                           channel_of_interest: int,
                           freq_range: Tuple[float, float] = (3, 8),
                           freq_step: float = 1.0,
                           window_duration: float = 1.0,
                           n_cycles_factor: float = 3.0,
                           save_path: str = 'nm_theta_results') -> Dict:
    """
    Complete NM theta analysis for a single session.
    
    Parameters:
    -----------
    session_data : Dict
        Session data dictionary
    channel_of_interest : int
        Channel index to analyze
    freq_range : Tuple[float, float]
        Frequency range for analysis (default: 3-8 Hz)
    freq_step : float
        Frequency resolution (default: 1 Hz)
    window_duration : float
        Event window duration (default: 1.0s = ±0.5s)
    n_cycles_factor : float
        Factor for adaptive n_cycles (default: 3.0)
    save_path : str
        Directory to save results
    
    Returns:
    --------
    results : Dict
        Complete analysis results
    """
    print("=" * 60)
    print("NM THETA OSCILLATION ANALYSIS")
    print("=" * 60)
    
    # Validate inputs
    if channel_of_interest >= session_data['eeg'].shape[0]:
        raise ValueError(f"Channel {channel_of_interest} not available. Max channel: {session_data['eeg'].shape[0]-1}")
    
    if len(session_data['nm_peak_times']) == 0:
        raise ValueError("No NM events found in session")
    
    # Step 1: Extract channel data
    print(f"Step 1: Analyzing channel {channel_of_interest}")
    eeg_channel = session_data['eeg'][channel_of_interest, :]
    times = session_data['eeg_time'].flatten()
    
    # Step 2: Compute high-resolution theta spectrogram
    print(f"Step 2: Computing theta spectrogram ({freq_range[0]}-{freq_range[1]} Hz)")
    freqs, power = compute_high_res_theta_spectrogram(
        eeg_channel, 
        sfreq=200.0,
        freq_range=freq_range,
        freq_step=freq_step,
        n_cycles_factor=n_cycles_factor
    )
    
    # Step 3: Compute global statistics
    print("Step 3: Computing global normalization statistics")
    global_mean, global_std = compute_global_statistics(power)
    
    # Step 4: Extract NM event windows
    print("Step 4: Extracting NM event windows")
    nm_windows = extract_nm_event_windows(
        power, times, 
        session_data['nm_peak_times'],
        session_data['nm_sizes'],
        window_duration
    )
    
    if not nm_windows:
        raise ValueError("No valid NM event windows extracted")
    
    # Step 5: Normalize windows
    print("Step 5: Z-score normalizing windows")
    normalized_windows = normalize_windows(nm_windows, global_mean, global_std)
    
    # Step 6: Save results
    print("Step 6: Saving results")
    save_results(
        session_data, normalized_windows, freqs, 
        global_mean, global_std, channel_of_interest, save_path
    )
    
    # Step 7: Generate plots
    print("Step 7: Generating plots")
    plot_nm_theta_results(normalized_windows, freqs, save_path)
    
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'freqs': freqs,
        'power': power,
        'global_mean': global_mean,
        'global_std': global_std,
        'normalized_windows': normalized_windows,
        'channel_analyzed': channel_of_interest
    }


def main():
    """
    Main function to run NM theta analysis on a sample session.
    """
    try:
        # Load session data
        print("Loading EEG session data...")
        session_data = load_session_data('all_eeg_data.pkl', session_index=0)
        
        # Analysis parameters
        channel_of_interest = 1  # Analyze first channel
        freq_range = (2, 10)      # Theta range
        freq_step = 0.125         # 1 Hz resolution
        window_duration = 1.0    # ±0.5s around events
        n_cycles_factor = 3.0    # For good frequency resolution
        
        # Run analysis
        results = analyze_session_nm_theta(
            session_data=session_data,
            channel_of_interest=channel_of_interest,
            freq_range=freq_range,
            freq_step=freq_step,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path='nm_theta_results'
        )
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"- Channel analyzed: {channel_of_interest}")
        print(f"- Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"- NM sizes found: {list(results['normalized_windows'].keys())}")
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