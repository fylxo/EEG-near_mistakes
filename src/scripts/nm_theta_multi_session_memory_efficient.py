#!/usr/bin/env python3
"""
Memory-Efficient Multi-Session NM Theta Analysis

This script analyzes theta oscillations around Near Mistake (NM) events across
multiple sessions from the same rat using a memory-efficient approach. It processes
sessions one by one, saves intermediate results, and then aggregates the final
averaged spectrograms to avoid memory overflow issues.

Key Features:
- Processes sessions individually to minimize memory usage
- Saves intermediate results immediately after processing each session
- Loads only required session data, not entire dataset
- Explicit memory cleanup between sessions
- Final aggregation loads only averaged spectrograms

Usage:
    python nm_theta_multi_session_memory_efficient.py --rat_id 10501 --roi frontal

Author: Generated for EEG near-mistake multi-session analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
import argparse
import gc
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

# Import from our modules
from nm_theta_analysis import analyze_session_nm_theta_roi
from electrode_utils import get_channels, load_electrode_mappings


def extract_session_from_loaded_data(all_data: List[Dict], session_index: int) -> Dict:
    """
    Extract a single session from already loaded data.
    
    Parameters:
    -----------
    all_data : List[Dict]
        All loaded session data
    session_index : int
        Index of the specific session to extract
    
    Returns:
    --------
    session_data : Dict
        Single session data dictionary
    """
    print(f"Extracting session {session_index} from loaded data")
    
    if session_index >= len(all_data):
        raise IndexError(f"Session index {session_index} out of range. Dataset has {len(all_data)} sessions.")
    
    session_data = all_data[session_index].copy()
    
    print(f"✓ Session {session_index} extracted successfully")
    session_date = session_data.get('session_date', 'unknown')
    rat_id = session_data.get('rat_id', 'unknown')
    eeg_shape = session_data.get('eeg', np.array([])).shape
    n_nm_events = len(session_data.get('nm_peak_times', []))
    print(f"  Rat ID: {rat_id}, Date: {session_date}")
    print(f"  EEG shape: {eeg_shape}, NM events: {n_nm_events}")
    
    return session_data


def find_rat_sessions_from_loaded_data(all_data: List[Dict], rat_id: Union[str, int]) -> Tuple[List[int], List[Dict]]:
    """
    Find all sessions for a specific rat from already loaded data.
    
    Parameters:
    -----------
    all_data : List[Dict]
        All loaded session data
    rat_id : Union[str, int]
        Rat ID to search for
    
    Returns:
    --------
    session_indices : List[int]
        List of session indices for the specified rat
    rat_sessions : List[Dict]
        List of session data for the specified rat
    """
    print(f"Finding all sessions for rat {rat_id} in loaded data")
    
    # Convert rat_id to string for comparison
    target_rat_id = str(rat_id)
    session_indices = []
    rat_sessions = []
    
    for session_idx, session_data in enumerate(all_data):
        session_rat_id = str(session_data.get('rat_id', 'unknown'))
        if session_rat_id == target_rat_id:
            session_indices.append(session_idx)
            rat_sessions.append(session_data)
            session_date = session_data.get('session_date', 'unknown')
            n_nm_events = len(session_data.get('nm_peak_times', []))
            eeg_shape = session_data.get('eeg', np.array([])).shape
            print(f"  Found session {session_idx}: Date {session_date}, "
                  f"EEG shape {eeg_shape}, NM events: {n_nm_events}")
    
    if not session_indices:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    print(f"✓ Found {len(session_indices)} sessions for rat {rat_id}")
    return session_indices, rat_sessions


def process_single_session_and_save(session_index: int,
                                   session_data: Dict,
                                   roi_or_channels: Union[str, List[int]],
                                   freq_range: Tuple[float, float],
                                   n_freqs: int,
                                   window_duration: float,
                                   n_cycles_factor: float,
                                   session_save_dir: str,
                                   mapping_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Process a single session and save results immediately to disk.
    
    Parameters:
    -----------
    session_index : int
        Index of session to process
    session_data : Dict
        Single session data dictionary
    roi_or_channels : Union[str, List[int]]
        ROI name or channel numbers
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Cycles factor for spectrograms
    session_save_dir : str
        Directory to save this session's results
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe
    
    Returns:
    --------
    session_summary : Dict
        Summary of session analysis (not full results)
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING SESSION {session_index}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Process the session (data already loaded)
        print(f"Running ROI analysis for session {session_index}...")
        
        result = analyze_session_nm_theta_roi(
            session_data=session_data,
            roi_or_channels=roi_or_channels,
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path=session_save_dir,
            mapping_df=mapping_df,
            show_plots=False  # Don't plot individual sessions
        )
        
        # Step 3: Save results immediately
        os.makedirs(session_save_dir, exist_ok=True)
        results_file = os.path.join(session_save_dir, 'session_results.pkl')
        
        with open(results_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"✓ Session {session_index} results saved to {results_file}")
        
        # Step 4: Create summary for return (minimal memory usage)
        session_summary = {
            'session_index': session_index,
            'rat_id': session_data.get('rat_id'),
            'session_date': session_data.get('session_date', 'unknown'),
            'roi_channels': result['roi_channels'],
            'roi_specification': result['roi_specification'],
            'total_nm_events': sum(data['n_events'] for data in result['normalized_windows'].values()),
            'nm_sizes': list(result['normalized_windows'].keys()),
            'results_file': results_file,
            'save_dir': session_save_dir
        }
        
        print(f"✓ Session {session_index} completed successfully")
        print(f"  ROI channels: {result['roi_channels']}")
        print(f"  Total NM events: {session_summary['total_nm_events']}")
        print(f"  NM sizes: {session_summary['nm_sizes']}")
        
        # Step 5: Explicit memory cleanup
        del result
        gc.collect()
        
        return session_summary
        
    except Exception as e:
        print(f"❌ Error processing session {session_index}: {e}")
        import traceback
        traceback.print_exc()
        
        # Even on error, clean up memory
        gc.collect()
        raise


def aggregate_session_results(session_summaries: List[Dict], 
                            rat_id: Union[str, int]) -> Dict:
    """
    Load saved session results and aggregate them into final averaged results.
    
    Parameters:
    -----------
    session_summaries : List[Dict]
        List of session summaries with file paths
    rat_id : Union[str, int]
        Rat identifier
    
    Returns:
    --------
    aggregated_results : Dict
        Final aggregated results with averaged spectrograms
    """
    print(f"\n{'='*60}")
    print(f"AGGREGATING RESULTS FROM {len(session_summaries)} SESSIONS")
    print(f"{'='*60}")
    
    # Load first session to get common parameters
    first_summary = session_summaries[0]
    with open(first_summary['results_file'], 'rb') as f:
        first_result = pickle.load(f)
    
    freqs = first_result['freqs']
    roi_channels = first_result['roi_channels']
    roi_specification = first_result['roi_specification']
    
    # Collect all NM sizes across sessions
    all_nm_sizes = set()
    for summary in session_summaries:
        all_nm_sizes.update(summary['nm_sizes'])
    all_nm_sizes = sorted(list(all_nm_sizes))
    
    print(f"NM sizes found across sessions: {all_nm_sizes}")
    print(f"Common parameters: {len(freqs)} frequencies, ROI: {roi_channels}")
    
    # Aggregate results for each NM size
    aggregated_windows = {}
    
    for nm_size in all_nm_sizes:
        print(f"\nAggregating NM size {nm_size}...")
        
        size_spectrograms = []
        size_sessions = []
        total_events = 0
        
        # Load each session's results for this NM size
        for session_idx, summary in enumerate(session_summaries):
            if nm_size in summary['nm_sizes']:
                print(f"  Loading session {summary['session_index']} results...")
                
                with open(summary['results_file'], 'rb') as f:
                    session_result = pickle.load(f)
                
                if nm_size in session_result['normalized_windows']:
                    windows_data = session_result['normalized_windows'][nm_size]
                    windows = windows_data['windows']  # (n_events, n_freqs, n_times)
                    
                    # Get average spectrogram for this session and NM size
                    session_avg = np.mean(windows, axis=0)  # (n_freqs, n_times)
                    size_spectrograms.append(session_avg)
                    size_sessions.append(session_idx)
                    total_events += windows.shape[0]
                    
                    print(f"    Session {summary['session_index']}: {windows.shape[0]} events")
                
                # Clean up after loading
                del session_result
                gc.collect()
        
        if size_spectrograms:
            # Average spectrograms across sessions
            all_session_avg = np.mean(size_spectrograms, axis=0)  # (n_freqs, n_times)
            
            # Get window times from first session (should be consistent)
            window_times = first_result['normalized_windows'][list(first_result['normalized_windows'].keys())[0]]['window_times']
            
            aggregated_windows[nm_size] = {
                'avg_spectrogram': all_session_avg,
                'window_times': window_times,
                'total_events': total_events,
                'n_sessions': len(size_sessions),
                'contributing_sessions': size_sessions
            }
            
            print(f"  ✓ NM size {nm_size}: {total_events} total events from {len(size_sessions)} sessions")
        else:
            print(f"  ⚠ NM size {nm_size}: No data found")
    
    # Clean up first result
    del first_result
    gc.collect()
    
    # Compile final aggregated results
    aggregated_results = {
        'rat_id': rat_id,
        'roi_specification': roi_specification,
        'roi_channels': roi_channels,
        'frequencies': freqs,
        'aggregated_windows': aggregated_windows,
        'session_summaries': session_summaries,
        'n_sessions_analyzed': len(session_summaries),
        'analysis_parameters': {
            'frequency_range': (freqs[0], freqs[-1]),
            'n_frequencies': len(freqs),
            'window_duration': window_times[-1] - window_times[0] if 'window_times' in locals() else 1.0,
            'normalization': 'per-channel z-score, averaged across ROI channels, then across sessions'
        }
    }
    
    print(f"✓ Aggregation completed successfully")
    return aggregated_results


def plot_aggregated_results(results: Dict, save_path: str = None):
    """Plot final aggregated spectrograms."""
    print(f"\n{'='*60}")
    print("PLOTTING AGGREGATED RESULTS")
    print(f"{'='*60}")
    
    aggregated_windows = results['aggregated_windows']
    freqs = results['frequencies']
    roi_channels = results['roi_channels']
    rat_id = results['rat_id']
    
    nm_sizes = sorted(aggregated_windows.keys())
    n_sizes = len(nm_sizes)
    
    if n_sizes == 0:
        print("No data to plot!")
        return
    
    # Create figure with subplots for each NM size
    fig, axes = plt.subplots(2, n_sizes, figsize=(5*n_sizes, 10))
    if n_sizes == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Memory-Efficient Multi-Session NM Theta Analysis - Rat {rat_id}\n'
                 f'ROI: {len(roi_channels)} channels {roi_channels} | '
                 f'{results["n_sessions_analyzed"]} sessions', 
                 fontsize=16, fontweight='bold')
    
    # Color scaling across all plots, centered at 0
    all_spectrograms = [data['avg_spectrogram'] for data in aggregated_windows.values()]
    vmin = min(spec.min() for spec in all_spectrograms)
    vmax = max(spec.max() for spec in all_spectrograms)
    # Center colormap at 0
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs
    
    for i, nm_size in enumerate(nm_sizes):
        data = aggregated_windows[nm_size]
        avg_spectrogram = data['avg_spectrogram']
        window_times = data['window_times']
        total_events = data['total_events']
        n_sessions = data['n_sessions']
        
        # Top row: Average spectrogram with diverging colormap centered at 0
        im1 = axes[0, i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        axes[0, i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        axes[0, i].set_title(f'NM Size {nm_size}\n{total_events} events, {n_sessions} sessions', 
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
        axes[1, i].set_xlabel('Frequency (Hz)', fontsize=10)
        axes[1, i].set_ylabel('Z-score at t=0', fontsize=10)
        axes[1, i].set_title(f'Event Profile', fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plot_file = os.path.join(save_path, 'memory_efficient_multi_session_spectrograms.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
    
    plt.show()


def save_aggregated_results(results: Dict, save_path: str):
    """Save final aggregated results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(save_path, 'aggregated_multi_session_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = {
        'rat_id': results['rat_id'],
        'roi_specification': results['roi_specification'],
        'roi_channels': results['roi_channels'],
        'n_sessions': results['n_sessions_analyzed'],
        'frequency_range': f"{results['frequencies'][0]:.1f}-{results['frequencies'][-1]:.1f} Hz",
        'nm_sizes': list(results['aggregated_windows'].keys()),
        'total_events_per_size': {size: data['total_events'] for size, data in results['aggregated_windows'].items()},
        'sessions_per_size': {size: data['n_sessions'] for size, data in results['aggregated_windows'].items()}
    }
    
    summary_file = os.path.join(save_path, 'memory_efficient_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Memory-Efficient Multi-Session NM Theta Analysis - Rat {results['rat_id']}\n")
        f.write("=" * 70 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Aggregated results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def analyze_rat_multi_session_memory_efficient(rat_id: Union[str, int],
                                              roi_or_channels: Union[str, List[int]],
                                              pkl_path: str = 'data/processed/all_eeg_data.pkl',
                                              freq_range: Tuple[float, float] = (3, 8),
                                              n_freqs: int = 30,
                                              window_duration: float = 1.0,
                                              n_cycles_factor: float = 3.0,
                                              save_path: str = None,
                                              mapping_df: Optional[pd.DataFrame] = None,
                                              show_plots: bool = True) -> Dict:
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
    
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    # Step 2: Process each session individually
    print(f"\nProcessing {len(session_indices)} sessions individually...")
    session_summaries = []
    
    for i, (session_index, session_data) in enumerate(zip(session_indices, rat_sessions)):
        print(f"\n--- Session {i+1}/{len(session_indices)} (Index {session_index}) ---")
        
        session_save_dir = os.path.join(save_path, f'session_{session_index}')
        
        try:
            summary = process_single_session_and_save(
                session_index=session_index,
                session_data=session_data,
                roi_or_channels=roi_or_channels,
                freq_range=freq_range,
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                session_save_dir=session_save_dir,
                mapping_df=mapping_df
            )
            session_summaries.append(summary)
            
        except Exception as e:
            print(f"❌ Skipping session {session_index} due to error: {e}")
            continue
    
    # Clean up loaded data
    del all_data
    gc.collect()
    
    if not session_summaries:
        raise ValueError("No sessions were successfully processed!")
    
    print(f"\n✓ Successfully processed {len(session_summaries)}/{len(session_indices)} sessions")
    
    # Step 3: Aggregate results from saved files
    print("\nAggregating results from saved session files...")
    aggregated_results = aggregate_session_results(session_summaries, rat_id)
    
    # Step 4: Save aggregated results
    print("\nSaving final aggregated results...")
    save_aggregated_results(aggregated_results, save_path)
    
    # Step 5: Plot results
    if show_plots:
        print("\nPlotting final results...")
        plot_aggregated_results(aggregated_results, save_path)
    else:
        print("Skipping plots (show_plots=False)")
    
    print("=" * 80)
    print("MEMORY-EFFICIENT MULTI-SESSION ANALYSIS COMPLETE!")
    print("=" * 80)
    
    return aggregated_results


def main(rat_id: str = None,
         roi: str = 'frontal',
         pkl_path: str = 'data/processed/all_eeg_data.pkl',
         freq_min: float = 2.0,
         freq_max: float = 8.0,
         n_freqs: int = 30,
         window_duration: float = 1.0,
         save_path: str = None,
         show_plots: bool = True):
    """
    Main function with direct parameter support.
    
    Parameters:
    -----------
    rat_id : str, optional
        Rat ID to analyze. If None, will use command line argument.
    roi : str
        ROI specification (e.g., frontal, hippocampus)
    pkl_path : str
        Path to all_eeg_data.pkl
    freq_min : float
        Minimum frequency (Hz)
    freq_max : float
        Maximum frequency (Hz)
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration (s)
    save_path : str, optional
        Save directory. If None, uses default naming.
    show_plots : bool
        Whether to display plots
    
    Returns:
    --------
    success : bool
        True if analysis completed successfully, False otherwise
    """
    
    # If rat_id is not provided, fall back to command line argument
    if rat_id is None:
        parser = argparse.ArgumentParser(description='Memory-efficient multi-session NM theta analysis')
        parser.add_argument('--rat_id', type=str, required=True, help='Rat ID to analyze')
        parser.add_argument('--roi', type=str, default='frontal', help='ROI specification (e.g., frontal, hippocampus)')
        parser.add_argument('--pkl_path', type=str, default='data/processed/all_eeg_data.pkl', 
                           help='Path to all_eeg_data.pkl')
        parser.add_argument('--freq_min', type=float, default=2.0, help='Minimum frequency (Hz)')
        parser.add_argument('--freq_max', type=float, default=8.0, help='Maximum frequency (Hz)')
        parser.add_argument('--n_freqs', type=int, default=30, help='Number of frequencies')
        parser.add_argument('--window_duration', type=float, default=1.0, help='Event window duration (s)')
        parser.add_argument('--save_path', type=str, default=None, help='Save directory')
        parser.add_argument('--no_plots', action='store_true', help='Skip plotting')
        
        args = parser.parse_args()
        
        rat_id = args.rat_id
        roi = args.roi
        pkl_path = args.pkl_path
        freq_min = args.freq_min
        freq_max = args.freq_max
        n_freqs = args.n_freqs
        window_duration = args.window_duration
        save_path = args.save_path
        show_plots = not args.no_plots
    
    try:
        print(f"Starting memory-efficient analysis for rat {rat_id}")
        print(f"ROI: {roi}")
        print(f"Frequency range: {freq_min}-{freq_max} Hz")
        print(f"Data file: {pkl_path}")
        
        results = analyze_rat_multi_session_memory_efficient(
            rat_id=rat_id,
            roi_or_channels=roi,
            pkl_path=pkl_path,
            freq_range=(freq_min, freq_max),
            n_freqs=n_freqs,
            window_duration=window_duration,
            save_path=save_path,
            show_plots=show_plots
        )
        
        # Print final summary
        print(f"\n🎉 Memory-efficient analysis completed for rat {rat_id}!")
        print(f"✓ ROI: {roi} -> channels {results['roi_channels']}")
        print(f"✓ Sessions analyzed: {results['n_sessions_analyzed']}")
        print(f"✓ NM sizes found: {list(results['aggregated_windows'].keys())}")
        
        for nm_size, data in results['aggregated_windows'].items():
            print(f"  - NM size {nm_size}: {data['total_events']} events from {data['n_sessions']} sessions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in memory-efficient analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Example usage with direct parameters
    success = main(
        rat_id="9592",  # Change this to your desired rat ID
        roi="frontal",
        pkl_path="data/processed/all_eeg_data.pkl",
        freq_min=2.0,
        freq_max=8.0,
        n_freqs=30,
        window_duration=1.0,
        save_path=None,  # Will use default naming
        show_plots=True
    )
    exit(0 if success else 1)