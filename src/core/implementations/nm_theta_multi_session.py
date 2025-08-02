#!/usr/bin/env python3
"""
Multi-Session NM Theta Analysis

This script analyzes theta oscillations around Near Mistake (NM) events across
multiple sessions from the same rat. It applies the ROI-based analysis to each
session and then averages the results across sessions.

Usage:
    python nm_theta_multi_session.py

Author: Generated for EEG near-mistake multi-session analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings
import gc

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our modules
from implementations.nm_theta_single_basic import analyze_session_nm_theta_roi, load_session_data
from utils.electrode_utils import get_channels, load_electrode_mappings


def load_all_sessions_for_rat(pkl_path: str, rat_id: Union[str, int]) -> List[Tuple[int, Dict]]:
    """
    Load all sessions for a specific rat from all_eeg_data.pkl.
    
    Parameters:
    -----------
    pkl_path : str
        Path to all_eeg_data.pkl
    rat_id : Union[str, int]
        Rat ID to filter sessions for
    
    Returns:
    --------
    sessions : List[Tuple[int, Dict]]
        List of (session_index, session_data) tuples for the specified rat
    """
    print(f"Loading all sessions for rat {rat_id} from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Convert rat_id to string for comparison
    target_rat_id = str(rat_id)
    rat_sessions = []
    
    for session_idx, session_data in enumerate(all_data):
        session_rat_id = str(session_data.get('rat_id', 'unknown'))
        if session_rat_id == target_rat_id:
            rat_sessions.append((session_idx, session_data))
    
    print(f"Found {len(rat_sessions)} sessions for rat {rat_id}")
    for idx, (session_idx, session_data) in enumerate(rat_sessions):
        session_date = session_data.get('session_date', 'unknown')
        n_nm_events = len(session_data.get('nm_peak_times', []))
        eeg_shape = session_data.get('eeg', np.array([])).shape
        print(f"  Session {idx+1}: Index {session_idx}, Date: {session_date}, "
              f"EEG shape: {eeg_shape}, NM events: {n_nm_events}")
    
    if not rat_sessions:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    return rat_sessions


def analyze_rat_multi_session(rat_id: Union[str, int],
                             roi_or_channels: Union[str, List[int]],
                             pkl_path: str = 'data/processed/all_eeg_data.pkl',
                             freq_range: Tuple[float, float] = (3, 8),
                             n_freqs: int = 30,
                             window_duration: float = 1.0,
                             n_cycles_factor: float = 3.0,
                             save_path: str = None,
                             mapping_df: Optional[pd.DataFrame] = None,
                             show_plots: bool = True,
                             use_baseline_normalization: bool = False) -> Dict:
    """
    Analyze NM theta oscillations across all sessions for a single rat.
    
    Parameters:
    -----------
    rat_id : Union[str, int]
        Rat identifier
    roi_or_channels : Union[str, List[int]]
        Either ROI name (e.g., 'frontal', 'hippocampus') or list of channel numbers (1-32, as in electrode mappings)
    pkl_path : str
        Path to all_eeg_data.pkl file
    freq_range : Tuple[float, float]
        Frequency range for analysis (default: 3-8 Hz)
    n_freqs : int
        Number of logarithmically spaced frequencies (default: 30)
    window_duration : float
        Event window duration (default: 1.0s = ±0.5s)
    n_cycles_factor : float
        Factor for adaptive n_cycles (default: 3.0)
    save_path : str, optional
        Directory to save results. If None, uses f'nm_multi_session_rat_{rat_id}'
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe. If None, loads from default CSV
    show_plots : bool
        Whether to display final averaged plots (default: True)
    
    Returns:
    --------
    results : Dict
        Multi-session analysis results with averaged spectrograms
    """
    
    print("=" * 70)
    print(f"MULTI-SESSION NM THETA ANALYSIS - RAT {rat_id}")
    print("=" * 70)
    
    # Add ROI mapping verification for multi-session
    print(f"\n🔍 ROI MAPPING VERIFICATION:")
    if isinstance(roi_or_channels, str):
        from core.electrode_utils import ROI_MAP
        electrode_numbers = ROI_MAP.get(roi_or_channels, [])
        print(f"✓ ROI specification: '{roi_or_channels}'")
        print(f"✓ Expected electrode numbers: {electrode_numbers}")
        print(f"✓ This mapping will be consistent across all sessions for rat {rat_id}")
    else:
        print(f"✓ Custom channel specification: {roi_or_channels}")
    print("=" * 70)
    
    if save_path is None:
        save_path = f'nm_multi_session_rat_{rat_id}'
    
    # Step 1: Load all sessions for this rat
    rat_sessions = load_all_sessions_for_rat(pkl_path, rat_id)
    
    # Step 2: Analyze each session
    print(f"\nAnalyzing {len(rat_sessions)} sessions...")
    session_results = []
    session_metadata = []
    
    for session_idx, (orig_session_idx, session_data) in enumerate(rat_sessions):
        print(f"\n--- Processing Session {session_idx + 1}/{len(rat_sessions)} ---")
        session_date = session_data.get('session_date', 'unknown')
        print(f"Session {orig_session_idx}: Date {session_date}")
        
        try:
            # Run ROI analysis for this session
            session_save_path = os.path.join(save_path, f'session_{orig_session_idx}')
            
            result = analyze_session_nm_theta_roi(
                session_data=session_data,
                roi_or_channels=roi_or_channels,
                freq_range=freq_range,
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                save_path=session_save_path,
                mapping_df=mapping_df,
                show_plots=False,  # Don't plot individual sessions
                use_baseline_normalization=use_baseline_normalization  # Pass through the parameter
            )
            
            session_results.append(result)
            
            # Store metadata
            metadata = {
                'original_session_index': orig_session_idx,
                'session_date': session_date,
                'rat_id': session_data.get('rat_id'),
                'roi_channels': result['roi_channels'],
                'total_nm_events': sum(data['n_events'] for data in result['normalized_windows'].values()),
                'nm_sizes': list(result['normalized_windows'].keys())
            }
            session_metadata.append(metadata)
            
            print(f"✓ Session {session_idx + 1} completed successfully")
            print(f"  ROI channels: {result['roi_channels']}")
            print(f"  NM events analyzed: {metadata['total_nm_events']}")
            
            # Verify mapping consistency across sessions
            if session_idx == 0:
                first_session_channels = result['roi_channels']
                print(f"  📌 Reference ROI channels: {sorted(first_session_channels)}")
            else:
                current_channels = result['roi_channels']
                if sorted(current_channels) == sorted(first_session_channels):
                    print(f"  ✅ ROI channels match reference session")
                else:
                    print(f"  ⚠️  ROI channels differ from reference: {sorted(current_channels)} vs {sorted(first_session_channels)}")
            
        except Exception as e:
            print(f"❌ Error processing session {session_idx + 1}: {e}")
            print("Continuing with remaining sessions...")
            continue
    
    if not session_results:
        raise ValueError("No sessions were successfully analyzed")
        
    print(f"\n✓ Successfully analyzed {len(session_results)}/{len(rat_sessions)} sessions")
    
    # Step 3: Average results across sessions
    print("\nAveraging results across sessions...")
    averaged_results = average_session_results(session_results, session_metadata, rat_id)
    
    # Step 4: Save multi-session results
    print("Saving multi-session results...")
    save_multi_session_results(averaged_results, save_path)
    
    # Step 5: Plot multi-session results
    if show_plots:
        print("Plotting multi-session results...")
        plot_multi_session_results(averaged_results, save_path)
    else:
        print("Skipping plots (show_plots=False)")
    
    print("=" * 70)
    print("MULTI-SESSION ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return averaged_results


def analyze_rat_multi_session_memory_efficient(rat_id: Union[str, int],
                                              roi_or_channels: Union[str, List[int]],
                                              pkl_path: str = 'data/processed/all_eeg_data.pkl',
                                              freq_range: Tuple[float, float] = (3, 8),
                                              n_freqs: int = 30,
                                              window_duration: float = 1.0,
                                              n_cycles_factor: float = 3.0,
                                              save_path: str = None,
                                              mapping_df: Optional[pd.DataFrame] = None,
                                              show_plots: bool = True,
                                              use_baseline_normalization: bool = False) -> Dict:
    """
    Memory-efficient multi-session analysis that processes sessions one by one.
    
    Parameters:
    -----------
    rat_id : Union[str, int]
        Rat identifier
    roi_or_channels : Union[str, List[int]]
        Either ROI name (e.g., 'frontal', 'hippocampus') or list of channel numbers (1-32)
    pkl_path : str
        Path to all_eeg_data.pkl file
    freq_range : Tuple[float, float]
        Frequency range for analysis (default: 3-8 Hz)
    n_freqs : int
        Number of logarithmically spaced frequencies (default: 30)
    window_duration : float
        Event window duration (default: 1.0s = ±0.5s)
    n_cycles_factor : float
        Factor for adaptive n_cycles (default: 3.0)
    save_path : str, optional
        Directory to save results. If None, uses f'nm_multi_session_memory_efficient_rat_{rat_id}'
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe. If None, loads from default CSV
    show_plots : bool
        Whether to display final plots (default: True)
    
    Returns:
    --------
    results : Dict
        Final aggregated analysis results
    """
    
    print("=" * 80)
    print(f"MEMORY-EFFICIENT MULTI-SESSION NM THETA ANALYSIS - RAT {rat_id}")
    print("=" * 80)
    
    if save_path is None:
        save_path = f'results/multi_session/rat_{rat_id}_memory_efficient'
    
    # Step 1: Load all data once and find sessions for this rat
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Find sessions for this rat
    target_rat_id = str(rat_id)
    session_indices = []
    rat_sessions = []
    
    for session_idx, session_data in enumerate(all_data):
        session_rat_id = str(session_data.get('rat_id', 'unknown'))
        if session_rat_id == target_rat_id:
            session_indices.append(session_idx)
            rat_sessions.append(session_data)
    
    if not session_indices:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    print(f"Found {len(session_indices)} sessions for rat {rat_id}")
    
    # Step 2: Process each session individually
    print(f"\nProcessing {len(session_indices)} sessions individually...")
    session_results = []
    session_metadata = []
    
    # Session processing with lightweight resilience
    failed_sessions = []
    max_retries = 2
    
    for session_idx, (orig_session_idx, session_data) in enumerate(zip(session_indices, rat_sessions)):
        print(f"\n--- Session {session_idx + 1}/{len(session_indices)} (Index {orig_session_idx}) ---")
        session_date = session_data.get('session_date', 'unknown')
        print(f"Session {orig_session_idx}: Date {session_date}")
        
        session_success = False
        
        # Try processing with retries
        for attempt in range(max_retries + 1):
            try:
                # Clean memory before each attempt (especially important for retries)
                if attempt > 0:
                    gc.collect()
                    print(f"  🔄 Retry {attempt} for session {orig_session_idx}")
                
                # Run ROI analysis for this session
                session_save_path = os.path.join(save_path, f'session_{orig_session_idx}')
                
                result = analyze_session_nm_theta_roi(
                    session_data=session_data,
                    roi_or_channels=roi_or_channels,
                    freq_range=freq_range,
                    n_freqs=n_freqs,
                    window_duration=window_duration,
                    n_cycles_factor=n_cycles_factor,
                    save_path=session_save_path,
                    mapping_df=mapping_df,
                    show_plots=False,  # Don't plot individual sessions
                    use_baseline_normalization=use_baseline_normalization  # Pass through the parameter
                )
                
                session_results.append(result)
                
                # Store metadata
                metadata = {
                    'original_session_index': orig_session_idx,
                    'session_date': session_date,
                    'rat_id': session_data.get('rat_id'),
                    'roi_channels': result['roi_channels'],
                    'total_nm_events': sum(data['n_events'] for data in result['normalized_windows'].values()),
                    'nm_sizes': list(result['normalized_windows'].keys())
                }
                session_metadata.append(metadata)
                
                print(f"✓ Session {session_idx + 1} completed successfully")
                print(f"  ROI channels: {result['roi_channels']}")
                print(f"  NM events analyzed: {metadata['total_nm_events']}")
                
                session_success = True
                break
                
            except Exception as e:
                print(f"❌ Session {orig_session_idx} attempt {attempt + 1} failed: {e}")
                
                # Force cleanup on failure
                gc.collect()
                
                if attempt == max_retries:
                    print(f"❌ Session {orig_session_idx} failed after {max_retries + 1} attempts")
                    failed_sessions.append(orig_session_idx)
        
        if not session_success:
            print(f"❌ Session {orig_session_idx} permanently failed - continuing with remaining sessions")
    
    # Report session processing results
    print(f"\n📊 Session processing summary:")
    print(f"  Total sessions: {len(session_indices)}")
    print(f"  Successful: {len(session_results)}")
    print(f"  Failed: {len(failed_sessions)}")
    if failed_sessions:
        print(f"  Failed session indices: {failed_sessions}")
    print(f"  Success rate: {len(session_results)/len(session_indices)*100:.1f}%")
    
    # Clean up loaded data
    del all_data
    gc.collect()
    
    if not session_results:
        raise ValueError("No sessions were successfully processed!")
    
    print(f"\n✓ Successfully processed {len(session_results)}/{len(session_indices)} sessions")
    
    # Step 3: Average results across sessions
    print("\nAveraging results across sessions...")
    aggregated_results = average_session_results(session_results, session_metadata, rat_id)
    
    # Step 4: Save results
    print("\nSaving aggregated results...")
    save_multi_session_results(aggregated_results, save_path)
    
    # Step 5: Plot results
    if show_plots:
        print("\nPlotting final results...")
        plot_multi_session_results(aggregated_results, save_path)
    else:
        print("Skipping plots (show_plots=False)")
    
    print("=" * 80)
    print("MEMORY-EFFICIENT MULTI-SESSION ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return aggregated_results


def average_session_results(session_results: List[Dict], 
                          session_metadata: List[Dict],
                          rat_id: Union[str, int]) -> Dict:
    """
    Average the normalized windows across sessions for each NM size.
    
    Parameters:
    -----------
    session_results : List[Dict]
        List of session analysis results
    session_metadata : List[Dict]
        List of session metadata
    rat_id : Union[str, int]
        Rat identifier
    
    Returns:
    --------
    averaged_results : Dict
        Averaged results across sessions
    """
    
    # Get common parameters from first session
    freqs = session_results[0]['freqs']
    roi_channels = session_results[0]['roi_channels']
    roi_specification = session_results[0]['roi_specification']
    
    # Collect all NM sizes across sessions
    all_nm_sizes = set()
    for result in session_results:
        all_nm_sizes.update(result['normalized_windows'].keys())
    all_nm_sizes = sorted(list(all_nm_sizes))
    
    print(f"NM sizes found across sessions: {all_nm_sizes}")
    
    # Average windows for each NM size
    averaged_windows = {}
    
    for nm_size in all_nm_sizes:
        print(f"Averaging windows for NM size {nm_size}...")
        
        # Collect windows from all sessions that have this NM size
        size_windows = []
        size_sessions = []
        total_events = 0
        
        for session_idx, result in enumerate(session_results):
            if nm_size in result['normalized_windows']:
                windows = result['normalized_windows'][nm_size]['windows']
                size_windows.append(windows)
                size_sessions.append(session_idx)
                total_events += windows.shape[0]
                print(f"  Session {session_idx + 1}: {windows.shape[0]} events")
        
        if size_windows:
            # Validate window shapes before concatenation
            print(f"  Validating {len(size_windows)} window arrays for shape consistency...")
            
            # Check all window arrays have the same shape (except first dimension)
            shapes = [windows.shape for windows in size_windows]
            freq_time_shapes = [shape[1:] for shape in shapes]  # (n_freqs, n_times) parts
            unique_ft_shapes = list(set(freq_time_shapes))
            
            if len(unique_ft_shapes) > 1:
                print(f"  ❌ ERROR: Inconsistent window shapes found!")
                print(f"     (n_freqs, n_times) shape counts: {[(shape, freq_time_shapes.count(shape)) for shape in unique_ft_shapes]}")
                for idx, (windows, session_idx) in enumerate(zip(size_windows, size_sessions)):
                    print(f"     Session {session_idx + 1}: shape {windows.shape}")
                
                # Try to find common shape and exclude outliers
                ft_shape_counts = [(shape, freq_time_shapes.count(shape)) for shape in unique_ft_shapes]
                most_common_ft_shape = max(ft_shape_counts, key=lambda x: x[1])[0]
                print(f"     Most common (n_freqs, n_times) shape: {most_common_ft_shape}")
                
                # Filter to only use windows with the most common shape
                valid_windows = []
                valid_sessions = []
                valid_total_events = 0
                
                for windows, session_idx in zip(size_windows, size_sessions):
                    if windows.shape[1:] == most_common_ft_shape:
                        valid_windows.append(windows)
                        valid_sessions.append(session_idx)
                        valid_total_events += windows.shape[0]
                
                print(f"     Using {len(valid_windows)}/{len(size_windows)} sessions with (n_freqs, n_times) shape {most_common_ft_shape}")
                size_windows = valid_windows
                size_sessions = valid_sessions
                total_events = valid_total_events
                
                if len(size_windows) == 0:
                    print(f"  ❌ No valid windows found for NM size {nm_size}")
                    continue
            else:
                print(f"  ✓ All window arrays have consistent (n_freqs, n_times) shape: {unique_ft_shapes[0]}")
            
            # Concatenate all windows across sessions
            all_windows = np.concatenate(size_windows, axis=0)  # (total_events, n_freqs, n_times)
            
            # Get window times from first session (should be consistent)
            window_times = session_results[0]['normalized_windows'][list(session_results[0]['normalized_windows'].keys())[0]]['window_times']
            
            averaged_windows[nm_size] = {
                'windows': all_windows,
                'window_times': window_times,
                'n_events': total_events,
                'n_sessions': len(size_sessions),
                'contributing_sessions': size_sessions,
                'avg_spectrogram': np.mean(all_windows, axis=0)  # Average across all events from all sessions
            }
            
            print(f"  ✓ NM size {nm_size}: {total_events} total events from {len(size_sessions)} sessions")
        else:
            print(f"  ⚠ NM size {nm_size}: No data found")
    
    # Compile averaged results
    averaged_results = {
        'rat_id': rat_id,
        'roi_specification': roi_specification,
        'roi_channels': roi_channels,
        'frequencies': freqs,
        'averaged_windows': averaged_windows,
        'session_metadata': session_metadata,
        'n_sessions_analyzed': len(session_results),
        'analysis_parameters': {
            'frequency_range': (freqs[0], freqs[-1]),
            'frequency_step': freqs[1] - freqs[0] if len(freqs) > 1 else 0.5,
            'n_frequencies': len(freqs),
            'window_duration': window_times[-1] - window_times[0] if 'window_times' in locals() else 1.0,
            'normalization': 'per-channel z-score, averaged across ROI channels, then across sessions'
        }
    }
    
    return averaged_results


def save_multi_session_results(results: Dict, save_path: str):
    """Save multi-session analysis results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(save_path, 'multi_session_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = {
        'rat_id': results['rat_id'],
        'roi_specification': results['roi_specification'],
        'roi_channels': results['roi_channels'],
        'n_sessions': results['n_sessions_analyzed'],
        'frequency_range': f"{results['frequencies'][0]:.1f}-{results['frequencies'][-1]:.1f} Hz",
        'nm_sizes': list(results['averaged_windows'].keys()),
        'total_events_per_size': {size: data['n_events'] for size, data in results['averaged_windows'].items()},
        'sessions_per_size': {size: data['n_sessions'] for size, data in results['averaged_windows'].items()}
    }
    
    summary_file = os.path.join(save_path, 'multi_session_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Session NM Theta Analysis Summary - Rat {results['rat_id']}\n")
        f.write("=" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Multi-session results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def plot_multi_session_results(results: Dict, save_path: str = None):
    """Plot averaged spectrograms across sessions."""
    print("Generating multi-session plots...")
    
    averaged_windows = results['averaged_windows']
    freqs = results['frequencies']
    roi_channels = results['roi_channels']
    rat_id = results['rat_id']
    
    nm_sizes = sorted(averaged_windows.keys())
    n_sizes = len(nm_sizes)
    
    if n_sizes == 0:
        print("No data to plot!")
        return
    
    # Create figure with subplots for each NM size
    fig, axes = plt.subplots(2, n_sizes, figsize=(5*n_sizes, 10))
    if n_sizes == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Multi-Session NM Theta Analysis - Rat {rat_id}\n'
                 f'ROI: {len(roi_channels)} channels {roi_channels} | '
                 f'{results["n_sessions_analyzed"]} sessions', 
                 fontsize=16, fontweight='bold')
    
    # Color scaling across all plots, centered at 0
    all_spectrograms = [data['avg_spectrogram'] for data in averaged_windows.values()]
    vmin = min(spec.min() for spec in all_spectrograms)
    vmax = max(spec.max() for spec in all_spectrograms)
    # Center colormap at 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    for i, nm_size in enumerate(nm_sizes):
        data = averaged_windows[nm_size]
        avg_spectrogram = data['avg_spectrogram']
        window_times = data['window_times']
        n_events = data['n_events']
        n_sessions = data['n_sessions']
        
        # Top row: Average spectrogram with diverging colormap centered at 0
        im1 = axes[0, i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        axes[0, i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        # Set y-axis ticks to show actual frequency values for spectrogram
        freq_step = max(1, len(freqs) // 10)
        freq_ticks = freqs[::freq_step]
        axes[0, i].set_yticks(freq_ticks)
        axes[0, i].set_yticklabels([f'{f:.1f}' for f in freq_ticks])
        
        axes[0, i].set_title(f'NM Size {nm_size}\n{n_events} events, {n_sessions} sessions', 
                            fontsize=12, fontweight='bold')
        axes[0, i].set_ylabel('Frequency (Hz)', fontsize=10)
        axes[0, i].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0, i])
        cbar1.set_label('Z-score', fontsize=9)
        
        # Bottom row: Frequency profile at event time
        event_idx = np.argmin(np.abs(window_times))
        freq_profile = avg_spectrogram[:, event_idx]
        
        axes[1, i].plot(freqs, freq_profile, 'o-', linewidth=2, markersize=4, color='blue')
        axes[1, i].axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Set x-axis ticks to show actual frequency values for frequency profile
        axes[1, i].set_xticks(freq_ticks)
        axes[1, i].set_xticklabels([f'{f:.1f}' for f in freq_ticks])
        
        axes[1, i].set_xlabel('Frequency (Hz)', fontsize=10)
        axes[1, i].set_ylabel('Z-score at t=0', fontsize=10)
        axes[1, i].set_title(f'Event Profile', fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plot_file = os.path.join(save_path, 'multi_session_spectrograms.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Multi-session plot saved: {plot_file}")
    
    plt.show()


def main():
    """
    Main function to run multi-session NM theta analysis.
    """
    try:
        # Analysis parameters
        rat_id = 10501  # Example rat ID
        roi_specification = 'frontal'  # Can also be a list like [10, 11, 12] (channel numbers 1-32)
        freq_range = (2, 10)      # Extended theta range
        n_freqs = 30              # 30 log-spaced frequencies across 2-10 Hz
        window_duration = 1.0
        n_cycles_factor = 3.0
        
        print(f"Starting multi-session analysis for rat {rat_id}")
        print(f"ROI specification: {roi_specification}")
        print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        
        # Run multi-session analysis
        results = analyze_rat_multi_session(
            rat_id=rat_id,
            roi_or_channels=roi_specification,
            pkl_path='data/processed/all_eeg_data.pkl',
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor
        )
        
        # Print final summary
        print(f"\n🎉 Multi-session analysis completed for rat {rat_id}!")
        print(f"✓ ROI: {roi_specification} -> channels {results['roi_channels']}")
        print(f"✓ Sessions analyzed: {results['n_sessions_analyzed']}")
        print(f"✓ NM sizes found: {[float(key) for key in results['averaged_windows'].keys()]}")
        
        for nm_size, data in results['averaged_windows'].items():
            print(f"  - NM size {nm_size}: {data['n_events']} events from {data['n_sessions']} sessions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-session analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)