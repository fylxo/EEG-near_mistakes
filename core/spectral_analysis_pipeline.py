"""
Spectral Analysis Pipeline for EEG Near-Mistake Events

This module provides functions to:
1. Compute Morlet spectrograms for entire EEG sessions
2. Extract time-frequency windows around events (NM and ITI)
3. Organize and average windows by event type and size
4. Generate plots and save results

Author: Generated for EEG near-mistake analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

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
    
    # Load only the specific session to avoid memory issues
    with open(pkl_path, 'rb') as f:
        try:
            all_data = pickle.load(f)
            if session_index >= len(all_data):
                raise IndexError(f"Session index {session_index} out of range. Total sessions: {len(all_data)}")
            
            session_data = all_data[session_index]
            print(f"Loaded session: {session_data.get('rat_id', 'unknown')} - {session_data.get('session_date', 'unknown')}")
            print(f"EEG shape: {session_data['eeg'].shape}")
            print(f"NM events: {len(session_data['nm_peak_times'])}")
            print(f"ITI events: {len(session_data['iti_peak_times'])}")
            
            return session_data
        except MemoryError:
            print("Memory error loading full dataset. Consider implementing session-by-session loading.")
            raise


def compute_full_session_morlet(session_data: Dict, 
                              sfreq: float = 200.0, 
                              freqs: np.ndarray = None,
                              n_cycles: int = 7,
                              channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Compute Morlet spectrogram for entire EEG session across all or selected channels.
    
    Parameters:
    -----------
    session_data : Dict
        Dictionary containing 'eeg' (32, n_timepoints), 'eeg_time' (1, n_timepoints)
    sfreq : float
        Sampling frequency in Hz
    freqs : np.ndarray
        Frequency vector for analysis (default: 4-50 Hz)
    n_cycles : int
        Number of cycles for Morlet wavelets
    channels : List[int]
        Channel indices to analyze (default: all 32 channels)
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector used
    tfr_full : np.ndarray
        Time-frequency representation (n_channels, n_freqs, n_times)
    times : np.ndarray
        Time vector for the session
    used_channels : List[int]
        Indices of channels that were processed
    """
    if freqs is None:
        freqs = np.arange(4, 50, 1)
    
    if channels is None:
        channels = list(range(session_data['eeg'].shape[0]))
    
    # Validate channel indices
    num_channels_available = session_data['eeg'].shape[0]
    for ch_idx in channels:
        if not (0 <= ch_idx < num_channels_available):
            raise ValueError(f"Invalid channel index: {ch_idx}. Must be between 0 and {num_channels_available - 1}")
    
    times = session_data['eeg_time'].flatten()
    n_channels = len(channels)
    n_freqs = len(freqs)
    n_times = session_data['eeg'].shape[1]
    
    tfr_full = np.zeros((n_channels, n_freqs, n_times))
    
    print(f"Computing Morlet spectrogram for {n_channels} channels...")
    for i, ch_idx in enumerate(channels):
        print(f"Processing channel {ch_idx} ({i+1}/{n_channels})")
        _, power = morlet_spectrogram(
            session_data['eeg'][ch_idx, :], 
            sfreq=sfreq, 
            freqs=freqs, 
            n_cycles=n_cycles
        )
        tfr_full[i, :, :] = power
    
    return freqs, tfr_full, times, channels


def extract_event_windows(tfr_full: np.ndarray,
                         times: np.ndarray,
                         event_times: np.ndarray,
                         event_sizes: np.ndarray,
                         window_duration: float = 4.0,
                         event_type: str = 'NM') -> Dict:
    """
    Extract time-frequency windows around events from full-session spectrogram.
    
    Parameters:
    -----------
    tfr_full : np.ndarray
        Full session TFR (n_channels, n_freqs, n_times)
    times : np.ndarray
        Time vector for full session
    event_times : np.ndarray
        Event onset times in seconds
    event_sizes : np.ndarray
        Event sizes (1, 2, or 3)
    window_duration : float
        Total window duration in seconds (centered on event)
    event_type : str
        Event type label ('NM' or 'ITI')
    
    Returns:
    --------
    event_windows : Dict
        Dictionary with keys 'windows', 'window_times', 'sizes', 'valid_events', 'event_type'
    """
    half_window = window_duration / 2
    time_diffs = np.diff(times)
    sfreq = 1 / np.median(time_diffs)
    
    # Check for variable sampling rate
    if np.std(time_diffs) > 0.001:  # Threshold for sampling rate variability
        print(f"Warning: Variable sampling rate detected (std={np.std(time_diffs):.4f}s). Results may be inaccurate.")
    
    window_samples = int(window_duration * sfreq)
    
    valid_windows = []
    valid_sizes = []
    valid_event_indices = []
    
    print(f"Extracting {event_type} event windows...")
    for i, event_time in enumerate(event_times):
        start_time = event_time - half_window
        end_time = event_time + half_window
        
        # Check if window is within recording bounds
        if start_time >= times[0] and end_time <= times[-1]:
            # Find time indices
            start_idx = np.searchsorted(times, start_time)
            end_idx = np.searchsorted(times, end_time)
            
            # Ensure exact window size
            if end_idx - start_idx >= window_samples:
                original_end_idx = end_idx
                end_idx = start_idx + window_samples
                window_tfr = tfr_full[:, :, start_idx:end_idx]
                valid_windows.append(window_tfr)
                valid_sizes.append(event_sizes[i])
                valid_event_indices.append(i)
                
                # Warn if significant truncation occurred
                if original_end_idx - end_idx > 5:
                    print(f"Warning: {event_type} event {i} at {event_time:.2f}s: Truncated window by {original_end_idx - end_idx} samples")
            else:
                print(f"Warning: {event_type} event {i} at {event_time:.2f}s too close to recording end")
        else:
            print(f"Warning: {event_type} event {i} at {event_time:.2f}s outside recording bounds")
    
    if valid_windows:
        windows_array = np.stack(valid_windows, axis=0)  # (n_events, n_channels, n_freqs, n_times)
        window_times = np.linspace(-half_window, half_window, window_samples)
        print(f"Extracted {len(valid_windows)} valid {event_type} windows")
    else:
        # Ensure consistent shape for empty arrays
        windows_array = np.empty((0, tfr_full.shape[0], tfr_full.shape[1], window_samples))
        window_times = np.linspace(-half_window, half_window, window_samples)
        print(f"No valid {event_type} windows extracted")
    
    return {
        'windows': windows_array,
        'window_times': window_times,
        'sizes': np.array(valid_sizes),
        'valid_events': np.array(valid_event_indices),
        'event_type': event_type
    }


def organize_windows_by_groups(nm_windows: Dict, iti_windows: Dict) -> Dict:
    """
    Organize extracted windows by event type and size groups.
    
    Parameters:
    -----------
    nm_windows : Dict
        NM event windows from extract_event_windows
    iti_windows : Dict
        ITI event windows from extract_event_windows
    
    Returns:
    --------
    organized_windows : Dict
        Nested dictionary with structure: [event_type][size][channel_group]
    """
    organized = defaultdict(lambda: defaultdict(dict))
    
    # Process NM windows
    if nm_windows['windows'].size > 0:
        for size in [1, 2, 3]:
            size_mask = nm_windows['sizes'] == size
            if np.any(size_mask):
                organized['NM'][size]['windows'] = nm_windows['windows'][size_mask]
                organized['NM'][size]['window_times'] = nm_windows['window_times']
                organized['NM'][size]['n_events'] = np.sum(size_mask)
                print(f"NM size {size}: {np.sum(size_mask)} events")
    
    # Process ITI windows
    if iti_windows['windows'].size > 0:
        for size in [1, 2, 3]:
            size_mask = iti_windows['sizes'] == size
            if np.any(size_mask):
                organized['ITI'][size]['windows'] = iti_windows['windows'][size_mask]
                organized['ITI'][size]['window_times'] = iti_windows['window_times']
                organized['ITI'][size]['n_events'] = np.sum(size_mask)
                print(f"ITI size {size}: {np.sum(size_mask)} events")
    
    return dict(organized)


def compute_group_averages(organized_windows: Dict) -> Dict:
    """
    Compute average spectrograms within each group (event type x size).
    
    Parameters:
    -----------
    organized_windows : Dict
        Output from organize_windows_by_groups
    
    Returns:
    --------
    group_averages : Dict
        Dictionary with average spectrograms for each group
    """
    averages = defaultdict(lambda: defaultdict(dict))
    
    print("Computing group averages...")
    for event_type in organized_windows:
        for size in organized_windows[event_type]:
            windows = organized_windows[event_type][size]['windows']
            if windows.size > 0:
                # Average across events: (n_events, n_channels, n_freqs, n_times) -> (n_channels, n_freqs, n_times)
                avg_spectrogram = np.mean(windows, axis=0)
                averages[event_type][size]['avg_spectrogram'] = avg_spectrogram
                averages[event_type][size]['window_times'] = organized_windows[event_type][size]['window_times']
                averages[event_type][size]['n_events'] = organized_windows[event_type][size]['n_events']
                print(f"Averaged {event_type} size {size}: {windows.shape[0]} events")
    
    return dict(averages)


def extract_theta_power_timecourse(group_averages: Dict, 
                                 freqs: np.ndarray,
                                 theta_band: Tuple[float, float] = (4, 8)) -> Dict:
    """
    Extract theta power time-courses from average spectrograms.
    
    Parameters:
    -----------
    group_averages : Dict
        Output from compute_group_averages
    freqs : np.ndarray
        Frequency vector
    theta_band : Tuple[float, float]
        Theta frequency band (low, high) in Hz
    
    Returns:
    --------
    theta_timecourses : Dict
        Dictionary with theta power time-courses for each group and channel
    """
    theta_data = defaultdict(lambda: defaultdict(dict))
    
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    print(f"Extracting theta power ({theta_band[0]}-{theta_band[1]} Hz)...")
    
    for event_type in group_averages:
        for size in group_averages[event_type]:
            avg_spec = group_averages[event_type][size]['avg_spectrogram']
            if avg_spec.size > 0:
                # Extract theta power: average across theta frequencies
                theta_power = np.mean(avg_spec[:, theta_mask, :], axis=1)  # (n_channels, n_times)
                theta_data[event_type][size]['theta_power'] = theta_power
                theta_data[event_type][size]['window_times'] = group_averages[event_type][size]['window_times']
                theta_data[event_type][size]['n_events'] = group_averages[event_type][size]['n_events']
    
    return dict(theta_data)


def get_colorblind_friendly_colors():
    """
    Define colorblind-friendly color palettes for NM (warm) and ITI (cool) events.
    
    Returns:
    --------
    colors : Dict
        Color specifications for different event types and sizes
    """
    return {
        'NM': {
            1: {'color': '#D73027', 'name': 'Dark Red'},      # Dark red
            2: {'color': '#FC8D59', 'name': 'Orange'},        # Orange  
            3: {'color': '#FEE08B', 'name': 'Light Orange'}   # Light orange
        },
        'ITI': {
            1: {'color': '#4575B4', 'name': 'Dark Blue'},     # Dark blue
            2: {'color': '#74ADD1', 'name': 'Medium Blue'},   # Medium blue
            3: {'color': '#ABD9E9', 'name': 'Light Blue'}     # Light blue
        }
    }

def get_line_styles():
    """
    Define line styles for different event sizes.
    
    Returns:
    --------
    styles : Dict
        Line style specifications for different sizes
    """
    return {
        1: {'linestyle': '-', 'linewidth': 2.5, 'name': 'Solid'},
        2: {'linestyle': '--', 'linewidth': 2.0, 'name': 'Dashed'}, 
        3: {'linestyle': ':', 'linewidth': 3.0, 'name': 'Dotted'}
    }

def report_missing_groups(group_averages: Dict):
    """
    Report missing event types and sizes for transparency.
    
    Parameters:
    -----------
    group_averages : Dict
        Group averages dictionary
    """
    print("\n=== Event Group Report ===")
    all_event_types = ['NM', 'ITI']
    all_sizes = [1, 2, 3]
    
    for event_type in all_event_types:
        print(f"\n{event_type} Events:")
        if event_type in group_averages:
            for size in all_sizes:
                if size in group_averages[event_type]:
                    n_events = group_averages[event_type][size]['n_events']
                    print(f"  Size {size}: {n_events} events ✓")
                else:
                    print(f"  Size {size}: No events found ✗")
        else:
            print(f"  No {event_type} events found at all ✗")
    print("=" * 27)

def plot_average_spectrograms(group_averages: Dict, 
                            freqs: np.ndarray,
                            channels: List[int],
                            roi_channels: Optional[List[int]] = None,
                            save_path: Optional[str] = None):
    """
    Plot average spectrograms for each group and channel/ROI with colorblind-friendly design.
    
    Parameters:
    -----------
    group_averages : Dict
        Output from compute_group_averages
    freqs : np.ndarray
        Frequency vector
    channels : List[int]
        List of channel indices
    roi_channels : Optional[List[int]]
        Indices of channels to include in ROI analysis
    save_path : Optional[str]
        Path to save figures
    """
    if roi_channels is None:
        roi_channels = list(range(len(channels)))
    
    # Create save directory if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Report missing groups
    report_missing_groups(group_averages)
    
    colors = get_colorblind_friendly_colors()
    
    # Plot ROI average first (most important)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Average Spectrograms - ROI Average ({len(roi_channels)} channels)', fontsize=16, fontweight='bold')
    
    for row, event_type in enumerate(['NM', 'ITI']):
        for col, size in enumerate([1, 2, 3]):
            ax = axes[row, col]
            
            if event_type in group_averages and size in group_averages[event_type]:
                data = group_averages[event_type][size]
                spectrogram = np.mean(data['avg_spectrogram'][roi_channels, :, :], axis=0)
                times = data['window_times']
                
                # Use colorblind-friendly colormap
                if event_type == 'NM':
                    cmap = 'Reds'
                else:
                    cmap = 'Blues'
                
                im = ax.pcolormesh(times, freqs, np.log10(spectrogram + 1e-12), 
                                 shading='auto', cmap=cmap)
                ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=2)
                
                # Color-coded title
                title_color = colors[event_type][size]['color']
                ax.set_title(f'{event_type} Size {size} (n={data["n_events"]})', 
                           color=title_color, fontweight='bold', fontsize=12)
                
                if col == 0:
                    ax.set_ylabel('Frequency (Hz)', fontsize=11)
                if row == 1:
                    ax.set_xlabel('Time (s)', fontsize=11)
                    
                # Improved colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Log₁₀ Power (μV²)', fontsize=10)
                
            else:
                ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, style='italic')
                ax.set_title(f'{event_type} Size {size}', fontsize=12)
                ax.set_facecolor('#f0f0f0')
            
            # Grid for better readability
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/spectrogram_roi_average.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot individual channels (first 4 channels as examples)
    n_channels_to_plot = min(4, len(channels))
    for ch_idx in range(n_channels_to_plot):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Average Spectrograms - Channel {channels[ch_idx]}', fontsize=14)
        
        for row, event_type in enumerate(['NM', 'ITI']):
            for col, size in enumerate([1, 2, 3]):
                ax = axes[row, col]
                
                if event_type in group_averages and size in group_averages[event_type]:
                    data = group_averages[event_type][size]
                    spectrogram = data['avg_spectrogram'][ch_idx, :, :]
                    times = data['window_times']
                    
                    im = ax.pcolormesh(times, freqs, np.log10(spectrogram + 1e-12), 
                                     shading='auto', cmap='inferno')
                    ax.axvline(0, color='white', linestyle='--', alpha=0.7)
                    ax.set_title(f'{event_type} Size {size} (n={data["n_events"]})')
                    ax.set_ylabel('Frequency (Hz)') if col == 0 else None
                    ax.set_xlabel('Time (s)') if row == 1 else None
                    plt.colorbar(im, ax=ax, label='Log Power')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{event_type} Size {size}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/spectrogram_ch{channels[ch_idx]}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_theta_timecourses(theta_data: Dict,
                         channels: List[int],
                         roi_channels: Optional[List[int]] = None,
                         save_path: Optional[str] = None):
    """
    Plot theta power time-courses with colorblind-friendly colors and line styles.
    
    Note: This function plots averages for single sessions. For multi-session 
    variability, use plot_theta_timecourses_with_variability().
    
    Parameters:
    -----------
    theta_data : Dict
        Output from extract_theta_power_timecourse
    channels : List[int]
        List of channel indices
    roi_channels : Optional[List[int]]
        Indices of channels to include in ROI analysis
    save_path : Optional[str]
        Path to save figures
    """
    if roi_channels is None:
        roi_channels = list(range(len(channels)))
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    colors = get_colorblind_friendly_colors()
    line_styles = get_line_styles()
    
    # Plot ROI average first (most important)
    plt.figure(figsize=(14, 8))
    plt.title(f'Theta Power Time-courses - ROI Average ({len(roi_channels)} channels)', 
             fontsize=16, fontweight='bold', pad=20)
    
    for event_type in ['NM', 'ITI']:
        for size in [1, 2, 3]:
            if event_type in theta_data and size in theta_data[event_type]:
                data = theta_data[event_type][size]
                theta_power = np.mean(data['theta_power'][roi_channels, :], axis=0)
                times = data['window_times']
                
                color_info = colors[event_type][size]
                style_info = line_styles[size]
                
                plt.plot(times, theta_power, 
                       color=color_info['color'],
                       linestyle=style_info['linestyle'],
                       linewidth=style_info['linewidth'],
                       label=f'{event_type} Size {size} (n={data["n_events"]}) - {style_info["name"]}',
                       alpha=0.9)
    
    plt.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Event Onset')
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Theta Power (μV²)', fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/theta_timecourse_roi_average.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot individual channels (first 4 as examples)
    n_channels_to_plot = min(4, len(channels))
    for ch_idx in range(n_channels_to_plot):
        plt.figure(figsize=(14, 8))
        plt.title(f'Theta Power Time-courses - Channel {channels[ch_idx]}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        for event_type in ['NM', 'ITI']:
            for size in [1, 2, 3]:
                if event_type in theta_data and size in theta_data[event_type]:
                    data = theta_data[event_type][size]
                    theta_power = data['theta_power'][ch_idx, :]
                    times = data['window_times']
                    
                    color_info = colors[event_type][size]
                    style_info = line_styles[size]
                    
                    plt.plot(times, theta_power, 
                           color=color_info['color'],
                           linestyle=style_info['linestyle'],
                           linewidth=style_info['linewidth'],
                           label=f'{event_type} Size {size} (n={data["n_events"]}) - {style_info["name"]}',
                           alpha=0.9)
        
        plt.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Event Onset')
        plt.xlabel('Time (s)', fontsize=13)
        plt.ylabel('Theta Power (μV²)', fontsize=13)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/theta_timecourse_ch{channels[ch_idx]}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_theta_timecourses_with_variability(multi_session_theta_data: List[Dict],
                                          channels: List[int],
                                          roi_channels: Optional[List[int]] = None,
                                          save_path: Optional[str] = None,
                                          plot_type: str = 'sem'):
    """
    Plot theta power time-courses with variability across sessions.
    
    Parameters:
    -----------
    multi_session_theta_data : List[Dict]
        List of theta_data dictionaries from multiple sessions
    channels : List[int]
        List of channel indices
    roi_channels : Optional[List[int]]
        Indices of channels to include in ROI analysis
    save_path : Optional[str]
        Path to save figures
    plot_type : str
        Type of variability plot: 'sem', 'std', 'boxplot', 'violin'
    """
    if roi_channels is None:
        roi_channels = list(range(len(channels)))
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    colors = get_colorblind_friendly_colors()
    line_styles = get_line_styles()
    
    # Aggregate data across sessions
    aggregated_data = defaultdict(lambda: defaultdict(list))
    window_times = None
    
    for session_theta in multi_session_theta_data:
        for event_type in ['NM', 'ITI']:
            for size in [1, 2, 3]:
                if event_type in session_theta and size in session_theta[event_type]:
                    data = session_theta[event_type][size]
                    roi_theta = np.mean(data['theta_power'][roi_channels, :], axis=0)
                    aggregated_data[event_type][size].append(roi_theta)
                    
                    if window_times is None:
                        window_times = data['window_times']
    
    if window_times is None:
        print("No valid data found across sessions")
        return
    
    # Create plots based on plot_type
    if plot_type in ['sem', 'std']:
        plot_theta_with_error_bands(aggregated_data, window_times, colors, line_styles, 
                                   plot_type, roi_channels, save_path)
    elif plot_type == 'boxplot':
        plot_theta_boxplots(aggregated_data, window_times, colors, roi_channels, save_path)
    elif plot_type == 'violin':
        plot_theta_violin_plots(aggregated_data, window_times, colors, roi_channels, save_path)


def plot_theta_with_error_bands(aggregated_data: Dict, window_times: np.ndarray, 
                               colors: Dict, line_styles: Dict, error_type: str,
                               roi_channels: List[int], save_path: Optional[str]):
    """
    Plot theta time-courses with error bands (SEM or STD).
    """
    plt.figure(figsize=(14, 8))
    plt.title(f'Theta Power Time-courses with {error_type.upper()} - ROI Average ({len(roi_channels)} channels)', 
             fontsize=16, fontweight='bold', pad=20)
    
    for event_type in ['NM', 'ITI']:
        for size in [1, 2, 3]:
            if size in aggregated_data[event_type] and aggregated_data[event_type][size]:
                data_array = np.array(aggregated_data[event_type][size])  # (n_sessions, n_timepoints)
                
                mean_theta = np.mean(data_array, axis=0)
                if error_type == 'sem':
                    error = np.std(data_array, axis=0) / np.sqrt(data_array.shape[0])
                else:  # std
                    error = np.std(data_array, axis=0)
                
                color_info = colors[event_type][size]
                style_info = line_styles[size]
                
                # Plot mean line
                plt.plot(window_times, mean_theta,
                        color=color_info['color'],
                        linestyle=style_info['linestyle'],
                        linewidth=style_info['linewidth'],
                        label=f'{event_type} Size {size} (n_sessions={data_array.shape[0]})',
                        alpha=0.9)
                
                # Plot error band
                plt.fill_between(window_times, 
                               mean_theta - error,
                               mean_theta + error,
                               color=color_info['color'],
                               alpha=0.2)
    
    plt.axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Event Onset')
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Theta Power (μV²)', fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/theta_timecourse_with_{error_type}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_theta_boxplots(aggregated_data: Dict, window_times: np.ndarray, 
                       colors: Dict, roi_channels: List[int], save_path: Optional[str]):
    """
    Plot theta power as boxplots at key time points.
    """
    # Select key time points (baseline, event, post-event)
    time_points = [-1.0, 0.0, 1.0]  # seconds
    time_indices = [np.argmin(np.abs(window_times - t)) for t in time_points]
    
    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 6))
    fig.suptitle(f'Theta Power Distribution at Key Time Points - ROI Average ({len(roi_channels)} channels)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, (time_point, time_idx) in enumerate(zip(time_points, time_indices)):
        ax = axes[ax_idx]
        
        box_data = []
        box_labels = []
        box_colors = []
        
        for event_type in ['NM', 'ITI']:
            for size in [1, 2, 3]:
                if size in aggregated_data[event_type] and aggregated_data[event_type][size]:
                    data_array = np.array(aggregated_data[event_type][size])
                    theta_values = data_array[:, time_idx]
                    
                    box_data.append(theta_values)
                    box_labels.append(f'{event_type}\nSize {size}')
                    box_colors.append(colors[event_type][size]['color'])
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f't = {time_point:.1f}s', fontsize=12, fontweight='bold')
        ax.set_ylabel('Theta Power (μV²)' if ax_idx == 0 else '', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/theta_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_theta_violin_plots(aggregated_data: Dict, window_times: np.ndarray, 
                          colors: Dict, roi_channels: List[int], save_path: Optional[str]):
    """
    Plot theta power as violin plots at key time points.
    """
    import matplotlib.patches as mpatches
    
    # Select key time points
    time_points = [-1.0, 0.0, 1.0]
    time_indices = [np.argmin(np.abs(window_times - t)) for t in time_points]
    
    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 6))
    fig.suptitle(f'Theta Power Distribution (Violin Plots) - ROI Average ({len(roi_channels)} channels)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, (time_point, time_idx) in enumerate(zip(time_points, time_indices)):
        ax = axes[ax_idx]
        
        positions = []
        violin_data = []
        violin_colors = []
        labels = []
        
        pos = 1
        for event_type in ['NM', 'ITI']:
            for size in [1, 2, 3]:
                if size in aggregated_data[event_type] and aggregated_data[event_type][size]:
                    data_array = np.array(aggregated_data[event_type][size])
                    theta_values = data_array[:, time_idx]
                    
                    positions.append(pos)
                    violin_data.append(theta_values)
                    violin_colors.append(colors[event_type][size]['color'])
                    labels.append(f'{event_type}\nSize {size}')
                    pos += 1
        
        if violin_data:
            parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
            
            for pc, color in zip(parts['bodies'], violin_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        
        ax.set_title(f't = {time_point:.1f}s', fontsize=12, fontweight='bold')
        ax.set_ylabel('Theta Power (μV²)' if ax_idx == 0 else '', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/theta_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_analysis_results(results: Dict, save_path: str):
    """
    Save analysis results to disk.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing all analysis results
    save_path : str
        Path to save the results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results (excluding large tfr_full to save space)
    results_to_save = results.copy()
    if 'tfr_full' in results_to_save:
        del results_to_save['tfr_full']  # Too large, can be recomputed
    
    with open(f'{save_path}/analysis_results.pkl', 'wb') as f:
        pickle.dump(results_to_save, f)
    
    # Save summary statistics
    summary = {
        'session_id': results['session_id'],
        'parameters': results['parameters'],
        'n_channels': len(results['channels']),
        'n_freqs': len(results['freqs']),
        'recording_duration': results['times'][-1] - results['times'][0]
    }
    
    if 'group_averages' in results:
        summary['group_stats'] = {}
        for event_type in results['group_averages']:
            for size in results['group_averages'][event_type]:
                key = f'{event_type}_size_{size}'
                summary['group_stats'][key] = {
                    'n_events': results['group_averages'][event_type][size]['n_events'],
                    'spectrogram_shape': results['group_averages'][event_type][size]['avg_spectrogram'].shape
                }
    
    with open(f'{save_path}/analysis_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Analysis results saved to {save_path}")


def process_single_session(session_data: Dict,
                         session_id: str,
                         sfreq: float = 200.0,
                         freqs: Optional[np.ndarray] = None,
                         n_cycles: int = 7,
                         channels: Optional[List[int]] = None,
                         roi_channels: Optional[List[int]] = None,
                         window_duration: float = 4.0,
                         theta_band: Tuple[float, float] = (4, 8),
                         save_path: Optional[str] = None,
                         plot_results: bool = True) -> Dict:
    """
    Complete processing pipeline for a single session.
    
    Parameters:
    -----------
    session_data : Dict
        Session data dictionary
    session_id : str
        Identifier for the session
    sfreq : float
        Sampling frequency in Hz
    freqs : Optional[np.ndarray]
        Frequency vector for analysis
    n_cycles : int
        Number of cycles for Morlet wavelets
    channels : Optional[List[int]]
        Channel indices to analyze
    roi_channels : Optional[List[int]]
        ROI channel indices
    window_duration : float
        Event window duration in seconds
    theta_band : Tuple[float, float]
        Theta frequency band
    save_path : Optional[str]
        Path to save results
    plot_results : bool
        Whether to generate plots
    
    Returns:
    --------
    results : Dict
        Complete analysis results
    """
    print(f"\n=== Processing session: {session_id} ===")
    
    if freqs is None:
        freqs = np.arange(4, 50, 1)
    
    if channels is None:
        channels = list(range(session_data['eeg'].shape[0]))
    
    if roi_channels is None:
        roi_channels = list(range(len(channels)))
    
    # Step 1: Compute full session spectrogram
    print("Step 1: Computing full session Morlet spectrogram...")
    freqs_out, tfr_full, times, used_channels = compute_full_session_morlet(
        session_data, sfreq, freqs, n_cycles, channels
    )
    
    # Step 2: Extract event windows
    print("Step 2: Extracting event windows...")
    nm_windows = extract_event_windows(
        tfr_full, times, session_data['nm_peak_times'], 
        session_data['nm_sizes'], window_duration, 'NM'
    )
    
    iti_windows = extract_event_windows(
        tfr_full, times, session_data['iti_peak_times'], 
        session_data['iti_sizes'], window_duration, 'ITI'
    )
    
    # Step 3: Organize by groups
    print("Step 3: Organizing windows by groups...")
    organized_windows = organize_windows_by_groups(nm_windows, iti_windows)
    
    # Step 4: Compute averages
    print("Step 4: Computing group averages...")
    group_averages = compute_group_averages(organized_windows)
    
    # Step 5: Extract theta time-courses
    print("Step 5: Extracting theta power time-courses...")
    theta_data = extract_theta_power_timecourse(group_averages, freqs_out, theta_band)
    
    # Compile results
    results = {
        'session_id': session_id,
        'freqs': freqs_out,
        'times': times,
        'channels': used_channels,
        'roi_channels': roi_channels,
        'tfr_full': tfr_full,
        'nm_windows': nm_windows,
        'iti_windows': iti_windows,
        'organized_windows': organized_windows,
        'group_averages': group_averages,
        'theta_data': theta_data,
        'parameters': {
            'sfreq': sfreq,
            'n_cycles': n_cycles,
            'window_duration': window_duration,
            'theta_band': theta_band
        }
    }
    
    # Step 6: Save results
    if save_path:
        print("Step 6: Saving results...")
        session_save_path = f"{save_path}/{session_id}"
        save_analysis_results(results, session_save_path)
    
    # Step 7: Generate plots
    if plot_results:
        print("Step 7: Generating plots...")
        plot_save_path = f"{save_path}/{session_id}" if save_path else None
        
        # Generate improved plots with colorblind-friendly design
        plot_average_spectrograms(
            group_averages, freqs_out, used_channels, roi_channels, plot_save_path
        )
        
        plot_theta_timecourses(
            theta_data, used_channels, roi_channels, plot_save_path
        )
    
    print(f"=== Session {session_id} processing complete! ===")
    return results


# Example usage and testing functions
def test_pipeline_on_sample():
    """
    Test the pipeline on a sample session.
    """
    try:
        # Load first session
        session_data = load_session_data('../all_eeg_data.pkl', session_index=0)
        
        # Create session ID
        session_id = f"{session_data.get('rat_id', 'unknown')}_{session_data.get('session_date', 'unknown')}"
        
        # Test with subset of channels for faster processing
        test_channels = list(range(8))  # First 8 channels
        roi_channels = [0, 1, 2, 3]     # First 4 as ROI
        
        # Run pipeline
        results = process_single_session(
            session_data=session_data,
            session_id=session_id,
            channels=test_channels,
            roi_channels=roi_channels,
            save_path='test_results',
            plot_results=True
        )
        
        return results
        
    except Exception as e:
        print(f"Error in test pipeline: {e}")
        return None


if __name__ == "__main__":
    print("Spectral Analysis Pipeline for EEG Near-Mistake Events")
    print("Use test_pipeline_on_sample() to test the pipeline")