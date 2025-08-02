#!/usr/bin/env python3
"""
Parallel NM Theta Analysis

This script combines single-session and multi-session NM theta analysis with parallelization
for improved performance. It uses threading/multiprocessing at multiple levels:
- Channel-level parallelization in spectrogram computation
- Session-level parallelization in multi-session analysis

Features:
- Single session analysis with parallel channel processing
- Multi-session analysis with memory-efficient parallel processing
- Threading and multiprocessing options
- Memory management and cleanup
- Same API as existing scripts but with parallelism options

Usage:
    # Single session with parallel channels
    python nm_theta_parallel.py --mode single --session_index 0 --roi frontal --n_jobs 4

    # Multi-session with parallel processing
    python nm_theta_parallel.py --mode multi --rat_id 10501 --roi frontal --n_jobs 4

Author: Generated for EEG near-mistake parallel analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
import argparse
import gc
import time
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import warnings
import mne

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from existing modules
from implementations.nm_theta_single_basic import (
    extract_nm_event_windows,
    normalize_windows,
    save_roi_results,
    plot_roi_theta_results,
    get_electrode_numbers_from_channels
)
from utils.electrode_utils import get_channels, load_electrode_mappings, ROI_MAP
from eeg_analysis_package.time_frequency import morlet_spectrogram


def process_single_channel_threaded(args):
    """
    Worker function for parallel channel processing using threading.
    
    Parameters:
    -----------
    args : tuple
        (channel_idx, eeg_channel_data, freqs, n_cycles, sfreq)
    
    Returns:
    --------
    channel_results : dict
        Results for this channel including normalized power and metadata
    """
    ch_idx, eeg_channel, freqs, n_cycles, sfreq = args
    
    try:
        # OPTIMIZED: Process all frequencies at once with adaptive n_cycles
        data = eeg_channel[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
        power = mne.time_frequency.tfr_array_morlet(
            data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, 
            output='power', zero_mean=True
        )
        channel_power = power[0, 0, :, :]  # (n_freqs, n_times)
        
        # Validate time axis alignment
        original_time_length = len(eeg_channel)
        if original_time_length != channel_power.shape[1]:
            print(f"Warning: Time axis mismatch for channel {ch_idx}")
            print(f"  Original signal length: {original_time_length}")
            print(f"  Spectrogram time axis: {channel_power.shape[1]}")
            print(f"  Difference: {original_time_length - channel_power.shape[1]} samples")
        
        # Compute per-channel statistics for normalization
        ch_mean = np.mean(channel_power, axis=1)  # Mean per frequency
        ch_std = np.std(channel_power, axis=1)    # Std per frequency
        ch_std = np.maximum(ch_std, 1e-12)        # Avoid division by zero
        
        # Z-score normalize this channel
        normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
        
        return {
            'channel_idx': ch_idx,
            'power': channel_power,
            'normalized_power': normalized_power,
            'mean': ch_mean,
            'std': ch_std,
            'power_range': (channel_power.min(), channel_power.max()),
            'zscore_range': (normalized_power.min(), normalized_power.max()),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'channel_idx': ch_idx,
            'power': None,
            'normalized_power': None,
            'mean': None,
            'std': None,
            'power_range': None,
            'zscore_range': None,
            'success': False,
            'error': str(e)
        }


def compute_roi_theta_spectrogram_parallel(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    sfreq: float = 200.0,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    n_cycles_factor: float = 3.0,
    n_jobs: Optional[int] = None,
    method: str = 'threading'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel version of compute_roi_theta_spectrogram with channel-level parallelization.
    
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
        Number of logarithmically spaced frequencies
    n_cycles_factor : float
        Factor for determining number of cycles per frequency
    n_jobs : Optional[int]
        Number of parallel jobs (None = auto-detect)
    method : str
        'threading' or 'multiprocessing'
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    roi_power : np.ndarray
        Average ROI power matrix (n_freqs, n_times)
    channel_powers : np.ndarray
        Individual channel powers (n_channels, n_freqs, n_times)
    """
    print(f"🔄 PARALLEL CHANNEL PROCESSING ({method})")
    print(f"   Channels: {len(roi_channels)}, Jobs: {n_jobs or 'auto'}")
    
    # Verification
    print(f"📊 PROCESSING VERIFICATION:")
    print(f"   Using {len(roi_channels)} channels: {sorted(roi_channels)}")
    print(f"   EEG data shape: {eeg_data.shape}")
    print(f"   Each channel will be z-score normalized individually, then averaged")
    
    # Create logarithmically spaced frequency vector
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    # Print exact frequencies being used
    print(f"📊 Using {n_freqs} logarithmically spaced frequencies:")
    print(f"   Range: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    print(f"   Frequencies: {[f'{f:.2f}' for f in freqs]}")
    print(f"   N-cycles: {[f'{nc:.1f}' for nc in n_cycles]}")
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(roi_channels))
    
    # Prepare arguments for parallel processing
    channel_args = []
    for ch_idx in roi_channels:
        eeg_channel = eeg_data[ch_idx, :]
        channel_args.append((ch_idx, eeg_channel, freqs, n_cycles, sfreq))
    
    # Process channels in parallel
    start_time = time.time()
    
    print(f"🚀 Starting parallel processing with {n_jobs} workers...")
    
    if method == 'threading':
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_single_channel_threaded, channel_args))
    elif method == 'multiprocessing':
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_single_channel_threaded, channel_args))
    else:
        raise ValueError(f"Unknown parallelization method: {method}. Use 'threading' or 'multiprocessing'")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Parallel processing time: {processing_time:.2f}s")
    
    # Process results and handle any failures
    successful_results = []
    failed_channels = []
    
    for result in results:
        if result['success']:
            successful_results.append(result)
            print(f"  ✅ Channel {result['channel_idx']}: "
                  f"Power range: {result['power_range'][0]:.2e} - {result['power_range'][1]:.2e}, "
                  f"Z-score range: {result['zscore_range'][0]:.2f} - {result['zscore_range'][1]:.2f}")
        else:
            failed_channels.append(result['channel_idx'])
            print(f"  ❌ Channel {result['channel_idx']} failed: {result['error']}")
    
    if not successful_results:
        raise ValueError("All channels failed to process!")
    
    if failed_channels:
        print(f"⚠️ Warning: {len(failed_channels)} channels failed: {failed_channels}")
        print(f"Continuing with {len(successful_results)} successful channels")
    
    # Combine successful results
    channel_powers = []
    normalized_powers = []
    
    for result in successful_results:
        channel_powers.append(result['power'])
        normalized_powers.append(result['normalized_power'])
    
    # Convert to arrays and compute ROI average
    channel_powers = np.array(channel_powers)  # (n_channels, n_freqs, n_times)
    normalized_powers = np.array(normalized_powers)
    roi_power = np.mean(normalized_powers, axis=0)  # Average across channels
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    print(f"Used {len(successful_results)}/{len(roi_channels)} channels successfully")
    
    return freqs, roi_power, channel_powers


def analyze_session_nm_theta_roi_parallel(
    session_data: Dict,
    roi_or_channels: Union[str, List[int]],
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    save_path: str = 'nm_theta_results',
    mapping_df: Optional[pd.DataFrame] = None,
    show_plots: bool = True,
    show_frequency_profiles: bool = False,
    n_jobs: Optional[int] = None,
    parallel_method: str = 'threading'
) -> Dict:
    """
    Parallel version of analyze_session_nm_theta_roi with channel-level parallelization.
    
    Parameters:
    -----------
    session_data : Dict
        Session data dictionary
    roi_or_channels : Union[str, List[int]]
        Either ROI name or list of channel numbers (1-32)
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Factor for adaptive n_cycles
    save_path : str
        Directory to save results
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe
    show_plots : bool
        Whether to display plots
    show_frequency_profiles : bool
        Whether to display ROI frequency profiles plot
    n_jobs : Optional[int]
        Number of parallel jobs
    parallel_method : str
        'threading' or 'multiprocessing'
    
    Returns:
    --------
    results : Dict
        Complete analysis results
    """
    print("=" * 60)
    print("PARALLEL NM THETA ROI ANALYSIS (SINGLE SESSION)")
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
        
        # Handle rat_id type conversion
        lookup_rat_id = rat_id
        if rat_id not in mapping_df.index:
            if isinstance(rat_id, str) and rat_id.isdigit():
                lookup_rat_id = int(rat_id)
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
    
    # Step 2: Compute ROI theta spectrogram with parallelization
    print(f"Step 2: Computing ROI theta spectrogram with parallel processing")
    print(f"   Parallelization method: {parallel_method}")
    print(f"   Number of jobs: {n_jobs or 'auto'}")
    
    freqs, roi_power, channel_powers = compute_roi_theta_spectrogram_parallel(
        session_data['eeg'],
        roi_channels,
        sfreq=200.0,
        freq_range=freq_range,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor,
        n_jobs=n_jobs,
        method=parallel_method
    )
    
    # Step 3: ROI power is already normalized per channel
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
    print("PARALLEL ROI ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'freqs': freqs,
        'roi_power': roi_power,
        'channel_powers': channel_powers,
        'roi_channels': roi_channels,
        'normalized_windows': normalized_windows,
        'roi_specification': roi_or_channels,
        'parallel_method': parallel_method,
        'n_jobs_used': n_jobs or min(cpu_count(), len(roi_channels))
    }


def process_single_session_parallel_threaded(args):
    """
    Worker function for parallel multi-session processing using threading.
    
    Parameters:
    -----------
    args : tuple
        (session_data, roi_or_channels, freq_range, n_freqs, window_duration, 
         n_cycles_factor, session_save_dir, mapping_df, session_idx, parallel_method, channel_n_jobs)
    
    Returns:
    --------
    session_summary : dict
        Summary of session analysis results
    """
    (session_data, roi_or_channels, freq_range, n_freqs, window_duration, 
     n_cycles_factor, session_save_dir, mapping_df, session_idx, parallel_method, channel_n_jobs) = args
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SESSION {session_idx} IN PARALLEL")
    print(f"{'='*60}")
    
    try:
        # Process the session with parallel channel processing
        print(f"Running parallel ROI analysis for session {session_idx}...")
        
        result = analyze_session_nm_theta_roi_parallel(
            session_data=session_data,
            roi_or_channels=roi_or_channels,
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path=session_save_dir,
            mapping_df=mapping_df,
            show_plots=False,  # Don't plot individual sessions
            show_frequency_profiles=False,
            n_jobs=channel_n_jobs,
            parallel_method=parallel_method
        )
        
        # Save results immediately
        os.makedirs(session_save_dir, exist_ok=True)
        results_file = os.path.join(session_save_dir, 'session_results.pkl')
        
        with open(results_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"✓ Session {session_idx} results saved to {results_file}")
        
        # Create summary for return (minimal memory usage)
        session_summary = {
            'session_index': session_idx,
            'rat_id': session_data.get('rat_id'),
            'session_date': session_data.get('session_date', 'unknown'),
            'roi_channels': result['roi_channels'],
            'roi_specification': result['roi_specification'],
            'total_nm_events': sum(data['n_events'] for data in result['normalized_windows'].values()),
            'nm_sizes': list(result['normalized_windows'].keys()),
            'results_file': results_file,
            'save_dir': session_save_dir,
            'parallel_method': parallel_method,
            'success': True,
            'error': None
        }
        
        print(f"✓ Session {session_idx} completed successfully")
        print(f"  ROI channels: {result['roi_channels']}")
        print(f"  Total NM events: {session_summary['total_nm_events']}")
        print(f"  NM sizes: {session_summary['nm_sizes']}")
        
        # Explicit memory cleanup
        del result
        gc.collect()
        
        return session_summary
        
    except Exception as e:
        print(f"❌ Error processing session {session_idx}: {e}")
        import traceback
        traceback.print_exc()
        
        # Even on error, clean up memory
        gc.collect()
        
        return {
            'session_index': session_idx,
            'rat_id': session_data.get('rat_id'),
            'session_date': session_data.get('session_date', 'unknown'),
            'roi_channels': [],
            'roi_specification': roi_or_channels,
            'total_nm_events': 0,
            'nm_sizes': [],
            'results_file': None,
            'save_dir': session_save_dir,
            'parallel_method': parallel_method,
            'success': False,
            'error': str(e)
        }


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


def aggregate_session_results_parallel(session_summaries: List[Dict], 
                                      rat_id: Union[str, int]) -> Dict:
    """
    Load saved session results and aggregate them into final averaged results.
    Same as the memory-efficient version but adapted for parallel results.
    
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
    print(f"AGGREGATING PARALLEL RESULTS FROM {len(session_summaries)} SESSIONS")
    print(f"{'='*60}")
    
    # Filter successful sessions
    successful_summaries = [s for s in session_summaries if s['success']]
    failed_summaries = [s for s in session_summaries if not s['success']]
    
    if failed_summaries:
        print(f"⚠️ Warning: {len(failed_summaries)} sessions failed:")
        for failed in failed_summaries:
            print(f"  Session {failed['session_index']}: {failed['error']}")
    
    if not successful_summaries:
        raise ValueError("No sessions were successfully analyzed!")
    
    print(f"📊 Aggregating {len(successful_summaries)} successful sessions...")
    
    # Load first session to get common parameters
    first_summary = successful_summaries[0]
    with open(first_summary['results_file'], 'rb') as f:
        first_result = pickle.load(f)
    
    freqs = first_result['freqs']
    roi_channels = first_result['roi_channels']
    roi_specification = first_result['roi_specification']
    
    # Collect all NM sizes across sessions
    all_nm_sizes = set()
    for summary in successful_summaries:
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
        session_window_times = []
        total_events = 0
        
        # Load each session's results for this NM size
        for session_idx, summary in enumerate(successful_summaries):
            if nm_size in summary['nm_sizes']:
                print(f"  Loading session {summary['session_index']} results...")
                
                with open(summary['results_file'], 'rb') as f:
                    session_result = pickle.load(f)
                
                if nm_size in session_result['normalized_windows']:
                    windows_data = session_result['normalized_windows'][nm_size]
                    windows = windows_data['windows']  # (n_events, n_freqs, n_times)
                    window_times = windows_data['window_times']
                    
                    # Get average spectrogram for this session and NM size
                    session_avg = np.mean(windows, axis=0)  # (n_freqs, n_times)
                    size_spectrograms.append(session_avg)
                    size_sessions.append(session_idx)
                    session_window_times.append(window_times)
                    total_events += windows.shape[0]
                    
                    print(f"    Session {summary['session_index']}: {windows.shape[0]} events, shape {session_avg.shape}")
                
                # Clean up after loading
                del session_result
                gc.collect()
        
        if size_spectrograms:
            # Validate spectrogram shapes before averaging
            print(f"  Validating {len(size_spectrograms)} spectrograms for shape consistency...")
            
            # Check all spectrograms have the same shape
            shapes = [spec.shape for spec in size_spectrograms]
            unique_shapes = list(set(shapes))
            
            if len(unique_shapes) > 1:
                print(f"  ❌ ERROR: Inconsistent spectrogram shapes found!")
                print(f"     Shape counts: {[(shape, shapes.count(shape)) for shape in unique_shapes]}")
                
                # Use most common shape
                shape_counts = [(shape, shapes.count(shape)) for shape in unique_shapes]
                most_common_shape = max(shape_counts, key=lambda x: x[1])[0]
                print(f"     Most common shape: {most_common_shape}")
                
                # Filter to only use spectrograms with the most common shape
                valid_spectrograms = []
                valid_sessions = []
                
                for spec, session_idx in zip(size_spectrograms, size_sessions):
                    if spec.shape == most_common_shape:
                        valid_spectrograms.append(spec)
                        valid_sessions.append(session_idx)
                
                print(f"     Using {len(valid_spectrograms)}/{len(size_spectrograms)} sessions with shape {most_common_shape}")
                size_spectrograms = valid_spectrograms
                size_sessions = valid_sessions
                
                if len(size_spectrograms) == 0:
                    print(f"  ❌ No valid spectrograms found for NM size {nm_size}")
                    continue
            else:
                print(f"  ✓ All spectrograms have consistent shape: {unique_shapes[0]}")
            
            # Use first session's window times (should be consistent)
            window_times = session_window_times[0] if session_window_times else np.linspace(-0.5, 0.5, 200)
            
            # Average spectrograms across sessions
            all_session_avg = np.mean(size_spectrograms, axis=0)  # (n_freqs, n_times)
            
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
        'session_summaries': successful_summaries,
        'failed_sessions': failed_summaries,
        'n_sessions_analyzed': len(successful_summaries),
        'n_sessions_failed': len(failed_summaries),
        'parallel_method': successful_summaries[0]['parallel_method'] if successful_summaries else 'unknown',
        'analysis_parameters': {
            'frequency_range': (freqs[0], freqs[-1]),
            'n_frequencies': len(freqs),
            'window_duration': window_times[-1] - window_times[0] if 'window_times' in locals() else 1.0,
            'normalization': 'per-channel z-score, averaged across ROI channels, then across sessions'
        }
    }
    
    print(f"✓ Parallel aggregation completed successfully")
    return aggregated_results


def analyze_rat_multi_session_parallel(
    rat_id: Union[str, int],
    roi_or_channels: Union[str, List[int]],
    pkl_path: str = 'data/processed/all_eeg_data.pkl',
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 30,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    save_path: str = None,
    mapping_df: Optional[pd.DataFrame] = None,
    show_plots: bool = True,
    session_n_jobs: Optional[int] = None,
    channel_n_jobs: Optional[int] = None,
    parallel_method: str = 'threading'
) -> Dict:
    """
    Parallel multi-session analysis with both session-level and channel-level parallelization.
    
    Parameters:
    -----------
    rat_id : Union[str, int]
        Rat identifier
    roi_or_channels : Union[str, List[int]]
        Either ROI name or list of channel numbers (1-32)
    pkl_path : str
        Path to all_eeg_data.pkl file
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Factor for adaptive n_cycles
    save_path : str, optional
        Directory to save results
    mapping_df : Optional[pd.DataFrame]
        Electrode mapping dataframe
    show_plots : bool
        Whether to display final plots
    session_n_jobs : Optional[int]
        Number of parallel jobs for session processing
    channel_n_jobs : Optional[int]
        Number of parallel jobs for channel processing within each session
    parallel_method : str
        'threading' or 'multiprocessing'
    
    Returns:
    --------
    results : Dict
        Final aggregated analysis results
    """
    
    print("=" * 80)
    print(f"PARALLEL MULTI-SESSION NM THETA ANALYSIS - RAT {rat_id}")
    print(f"  Method: {parallel_method}")
    print(f"  Session parallelism: {session_n_jobs or 'auto'} jobs")
    print(f"  Channel parallelism: {channel_n_jobs or 'auto'} jobs per session")
    print("=" * 80)
    
    if save_path is None:
        save_path = f'results/multi_session/rat_{rat_id}_parallel_{parallel_method}'
    
    # Step 1: Load all data and find sessions for this rat
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    if session_n_jobs is None:
        session_n_jobs = min(cpu_count(), len(session_indices))
    
    if channel_n_jobs is None:
        channel_n_jobs = 2  # Conservative default for channel-level parallelism
    
    print(f"Will process {len(session_indices)} sessions with {session_n_jobs} parallel workers")
    print(f"Each session will use {channel_n_jobs} workers for channel processing")
    
    # Step 2: Process sessions in parallel
    print(f"\nProcessing {len(session_indices)} sessions in parallel...")
    
    # Prepare arguments for parallel session processing
    session_args = []
    for i, (session_index, session_data) in enumerate(zip(session_indices, rat_sessions)):
        session_save_dir = os.path.join(save_path, f'session_{session_index}')
        args = (
            session_data, roi_or_channels, freq_range, n_freqs, window_duration,
            n_cycles_factor, session_save_dir, mapping_df, session_index, 
            parallel_method, channel_n_jobs
        )
        session_args.append(args)
    
    # Process sessions in parallel
    start_time = time.time()
    
    print(f"🚀 Starting parallel session processing with {session_n_jobs} workers...")
    
    if parallel_method == 'threading':
        with ThreadPoolExecutor(max_workers=session_n_jobs) as executor:
            session_summaries = list(executor.map(process_single_session_parallel_threaded, session_args))
    elif parallel_method == 'multiprocessing':
        with ProcessPoolExecutor(max_workers=session_n_jobs) as executor:
            session_summaries = list(executor.map(process_single_session_parallel_threaded, session_args))
    else:
        raise ValueError(f"Unknown parallelization method: {parallel_method}")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Total parallel session processing time: {processing_time:.2f}s")
    
    # Clean up loaded data
    del all_data
    gc.collect()
    
    successful_sessions = [s for s in session_summaries if s['success']]
    failed_sessions = [s for s in session_summaries if not s['success']]
    
    print(f"\n✓ Successfully processed {len(successful_sessions)}/{len(session_indices)} sessions")
    if failed_sessions:
        print(f"❌ Failed sessions: {[s['session_index'] for s in failed_sessions]}")
    
    if not successful_sessions:
        raise ValueError("No sessions were successfully processed!")
    
    # Step 3: Aggregate results from saved files
    print("\nAggregating results from saved session files...")
    aggregated_results = aggregate_session_results_parallel(session_summaries, rat_id)
    
    # Step 4: Save aggregated results
    print("\nSaving final aggregated results...")
    save_aggregated_results_parallel(aggregated_results, save_path)
    
    # Step 5: Plot results
    if show_plots:
        print("\nPlotting final results...")
        plot_aggregated_results_parallel(aggregated_results, save_path)
    else:
        print("Skipping plots (show_plots=False)")
    
    # Add timing information
    aggregated_results['processing_time'] = processing_time
    aggregated_results['session_n_jobs'] = session_n_jobs
    aggregated_results['channel_n_jobs'] = channel_n_jobs
    
    print("=" * 80)
    print("PARALLEL MULTI-SESSION ANALYSIS COMPLETE!")
    print(f"  Total processing time: {processing_time:.2f}s")
    print(f"  Sessions processed: {len(successful_sessions)}")
    print(f"  Parallelization: {parallel_method}")
    print("=" * 80)
    
    return aggregated_results


def save_aggregated_results_parallel(results: Dict, save_path: str):
    """Save final aggregated parallel results."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(save_path, 'parallel_multi_session_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = {
        'rat_id': results['rat_id'],
        'roi_specification': results['roi_specification'],
        'roi_channels': results['roi_channels'],
        'n_sessions_successful': results['n_sessions_analyzed'],
        'n_sessions_failed': results['n_sessions_failed'],
        'frequency_range': f"{results['frequencies'][0]:.1f}-{results['frequencies'][-1]:.1f} Hz",
        'nm_sizes': list(results['aggregated_windows'].keys()),
        'total_events_per_size': {size: data['total_events'] for size, data in results['aggregated_windows'].items()},
        'sessions_per_size': {size: data['n_sessions'] for size, data in results['aggregated_windows'].items()},
        'parallel_method': results['parallel_method'],
        'processing_time': results.get('processing_time', 'unknown'),
        'session_n_jobs': results.get('session_n_jobs', 'unknown'),
        'channel_n_jobs': results.get('channel_n_jobs', 'unknown')
    }
    
    summary_file = os.path.join(save_path, 'parallel_multi_session_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Parallel Multi-Session NM Theta Analysis - Rat {results['rat_id']}\n")
        f.write("=" * 70 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Parallel aggregated results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def plot_aggregated_results_parallel(results: Dict, save_path: str = None):
    """Plot final aggregated spectrograms from parallel processing."""
    print("Generating parallel multi-session plots...")
    
    aggregated_windows = results['aggregated_windows']
    freqs = results['frequencies']
    roi_channels = results['roi_channels']
    rat_id = results['rat_id']
    parallel_method = results['parallel_method']
    
    nm_sizes = sorted(aggregated_windows.keys())
    n_sizes = len(nm_sizes)
    
    if n_sizes == 0:
        print("No data to plot!")
        return
    
    # Create figure with subplots for each NM size
    fig, axes = plt.subplots(2, n_sizes, figsize=(5*n_sizes, 10))
    if n_sizes == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Parallel Multi-Session NM Theta Analysis - Rat {rat_id}\n'
                 f'ROI: {len(roi_channels)} channels {roi_channels} | '
                 f'{results["n_sessions_analyzed"]} sessions | Method: {parallel_method}', 
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
        
        # Top row: Average spectrogram
        im1 = axes[0, i].pcolormesh(
            window_times, freqs, avg_spectrogram,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        axes[0, i].axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        
        # Set y-axis ticks
        freq_step = max(1, len(freqs) // 10)
        freq_ticks = freqs[::freq_step]
        axes[0, i].set_yticks(freq_ticks)
        axes[0, i].set_yticklabels([f'{f:.1f}' for f in freq_ticks])
        
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
        
        axes[1, i].set_xticks(freq_ticks)
        axes[1, i].set_xticklabels([f'{f:.1f}' for f in freq_ticks])
        
        axes[1, i].set_xlabel('Frequency (Hz)', fontsize=10)
        axes[1, i].set_ylabel('Z-score at t=0', fontsize=10)
        axes[1, i].set_title(f'Event Profile', fontsize=11)
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plot_file = os.path.join(save_path, f'parallel_{parallel_method}_multi_session_spectrograms.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Parallel plot saved: {plot_file}")
    
    plt.show()


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


def main():
    """
    Main function with optional argument parsing and direct parameter configuration.
    
    You can either:
    1. Modify the parameters directly in the code below (recommended)
    2. Use command line arguments with --use_args flag
    """
    
    # =============================================================================
    # DIRECT PARAMETER CONFIGURATION (MODIFY THESE AS NEEDED)
    # =============================================================================
    
    # Analysis mode: 'single' or 'multi'
    mode = 'multi'  # Change to 'multi' for multi-session analysis
    
    # Data file
    pkl_path = 'data/processed/all_eeg_data.pkl'
    
    # ROI specification - can be:
    # - ROI name: 'frontal', 'hippocampus', etc.
    # - List of channel numbers: [2, 3, 5] (1-32 electrode numbers)
    # - List of channel indices: [1, 2, 4] (0-31 channel indices)
    roi_or_channels = 'motor'  # or [2, 3, 5] for custom channels
    
    # Frequency analysis parameters
    freq_range = (1.0, 5.0)  # (min_freq, max_freq) in Hz
    n_freqs = 25           # Number of logarithmically spaced frequencies
    window_duration = 1.0    # Event window duration in seconds (±0.5s)
    n_cycles_factor = 3.0    # Factor for adaptive n_cycles
    
    # Parallelization parameters
    parallel_method = 'threading'  # 'threading' or 'multiprocessing'
    channel_n_jobs = None          # None = auto-detect, or specify number (e.g., 4)
    session_n_jobs = None          # For multi-session mode
    
    # Single session parameters (only used if mode = 'single')
    session_index = 0  # Which session to analyze (0-based index)
    
    # Multi-session parameters (only used if mode = 'multi')
    rat_id = '9151'  # Rat ID for multi-session analysis
    
    # Output parameters
    save_path = None   # None = auto-generate, or specify path
    show_plots = True  # Whether to display plots
    
    # =============================================================================
    # OPTIONAL COMMAND LINE ARGUMENT PARSING
    # =============================================================================
    
    # Check if user wants to use command line arguments
    use_command_line = '--use_args' in sys.argv or len(sys.argv) > 1 and not any(arg.endswith('.py') for arg in sys.argv)
    
    if use_command_line:
        print("📋 Using command line arguments...")
        parser = argparse.ArgumentParser(description='Parallel NM Theta Analysis')
        parser.add_argument('--use_args', action='store_true',
                           help='Flag to indicate using command line arguments')
        parser.add_argument('--mode', choices=['single', 'multi'], default=mode,
                           help='Analysis mode: single session or multi-session')
        
        # Common parameters
        parser.add_argument('--pkl_path', type=str, default=pkl_path,
                           help='Path to all_eeg_data.pkl')
        parser.add_argument('--roi', type=str, default=str(roi_or_channels),
                           help='ROI specification (e.g., frontal, hippocampus) or custom channels as comma-separated list')
        parser.add_argument('--freq_min', type=float, default=freq_range[0],
                           help='Minimum frequency (Hz)')
        parser.add_argument('--freq_max', type=float, default=freq_range[1],
                           help='Maximum frequency (Hz)')
        parser.add_argument('--n_freqs', type=int, default=n_freqs,
                           help='Number of frequencies')
        parser.add_argument('--window_duration', type=float, default=window_duration,
                           help='Event window duration (s)')
        parser.add_argument('--save_path', type=str, default=save_path,
                           help='Save directory')
        parser.add_argument('--no_plots', action='store_true',
                           help='Skip plotting')
        
        # Parallelization parameters
        parser.add_argument('--parallel_method', choices=['threading', 'multiprocessing'], 
                           default=parallel_method, help='Parallelization method')
        parser.add_argument('--session_n_jobs', type=int, default=session_n_jobs,
                           help='Number of parallel jobs for session processing (multi-session mode)')
        parser.add_argument('--channel_n_jobs', type=int, default=channel_n_jobs,
                           help='Number of parallel jobs for channel processing')
        
        # Single session parameters
        parser.add_argument('--session_index', type=int, default=session_index,
                           help='Session index for single session mode')
        
        # Multi-session parameters
        parser.add_argument('--rat_id', type=str, default=rat_id,
                           help='Rat ID for multi-session mode')
        
        args = parser.parse_args()
        
        # Override parameters with command line arguments
        mode = args.mode
        pkl_path = args.pkl_path
        freq_range = (args.freq_min, args.freq_max)
        n_freqs = args.n_freqs
        window_duration = args.window_duration
        save_path = args.save_path
        show_plots = not args.no_plots
        parallel_method = args.parallel_method
        session_n_jobs = args.session_n_jobs
        channel_n_jobs = args.channel_n_jobs
        session_index = args.session_index
        rat_id = args.rat_id
        
        # Parse ROI
        try:
            # Try to parse as comma-separated channel numbers
            roi_channels = [int(x.strip()) for x in args.roi.split(',')]
            roi_or_channels = roi_channels
            print(f"Using custom channels: {roi_channels}")
        except ValueError:
            # Use as ROI name
            roi_or_channels = args.roi
            print(f"Using ROI: {args.roi}")
    else:
        print("📋 Using direct parameter configuration from code...")
        print(f"   Mode: {mode}")
        print(f"   ROI: {roi_or_channels}")
        print(f"   Parallel method: {parallel_method}")
        if mode == 'single':
            print(f"   Session index: {session_index}")
        else:
            print(f"   Rat ID: {rat_id}")
    
    # =============================================================================
    
    try:
        if use_command_line:
            # Use variables from args
            if args.mode == 'single':
                print(f"🚀 Starting parallel single session analysis...")
                print(f"  Session index: {args.session_index}")
                print(f"  ROI: {roi_or_channels}")
                print(f"  Parallel method: {args.parallel_method}")
                print(f"  Channel jobs: {args.channel_n_jobs or 'auto'}")
                
                # Load session data
                session_data = load_session_data(args.pkl_path, args.session_index)
                
                # Set save path
                save_path = args.save_path or f'results/single_session/session_{args.session_index}_parallel_{args.parallel_method}'
                
                # Run analysis
                results = analyze_session_nm_theta_roi_parallel(
                    session_data=session_data,
                    roi_or_channels=roi_or_channels,
                    freq_range=freq_range,
                    n_freqs=args.n_freqs,
                    window_duration=args.window_duration,
                    save_path=save_path,
                    show_plots=show_plots,
                    n_jobs=args.channel_n_jobs,
                    parallel_method=args.parallel_method
                )
                
                print(f"\n🎉 Parallel single session analysis completed!")
                print(f"✓ ROI: {roi_or_channels} -> channels {results['roi_channels']}")
                print(f"✓ Parallel method: {results['parallel_method']}")
                print(f"✓ Jobs used: {results['n_jobs_used']}")
                print(f"✓ NM sizes analyzed: {list(results['normalized_windows'].keys())}")
            
            elif args.mode == 'multi':
                if args.rat_id is None:
                    print("❌ Error: --rat_id is required for multi-session mode")
                    return False
                
                print(f"🚀 Starting parallel multi-session analysis...")
                print(f"  Rat ID: {args.rat_id}")
                print(f"  ROI: {roi_or_channels}")
                print(f"  Parallel method: {args.parallel_method}")
                print(f"  Session jobs: {args.session_n_jobs or 'auto'}")
                print(f"  Channel jobs: {args.channel_n_jobs or 'auto'}")
                
                # Set save path
                save_path = args.save_path or f'results/multi_session/rat_{args.rat_id}_parallel_{args.parallel_method}'
                
                # Run analysis
                results = analyze_rat_multi_session_parallel(
                    rat_id=args.rat_id,
                    roi_or_channels=roi_or_channels,
                    pkl_path=args.pkl_path,
                    freq_range=freq_range,
                    n_freqs=args.n_freqs,
                    window_duration=args.window_duration,
                    save_path=save_path,
                    show_plots=show_plots,
                    session_n_jobs=args.session_n_jobs,
                    channel_n_jobs=args.channel_n_jobs,
                    parallel_method=args.parallel_method
                )
                
                print(f"\n🎉 Parallel multi-session analysis completed!")
                print(f"✓ Rat ID: {args.rat_id}")
                print(f"✓ ROI: {roi_or_channels} -> channels {results['roi_channels']}")
                print(f"✓ Sessions analyzed: {results['n_sessions_analyzed']}")
                print(f"✓ Parallel method: {results['parallel_method']}")
                print(f"✓ Processing time: {results['processing_time']:.2f}s")
                print(f"✓ NM sizes found: {[float(key) for key in results['aggregated_windows'].keys()]}")
            return True
        else:
            # Use directly set variables
            if mode == 'single':
                print(f"🚀 Starting parallel single session analysis...")
                print(f"  Session index: {session_index}")
                print(f"  ROI: {roi_or_channels}")
                print(f"  Parallel method: {parallel_method}")
                print(f"  Channel jobs: {channel_n_jobs or 'auto'}")
                
                # Load session data
                session_data = load_session_data(pkl_path, session_index)
                
                # Set save path
                save_dir = save_path or f'results/single_session/session_{session_index}_parallel_{parallel_method}'
                
                # Run analysis
                results = analyze_session_nm_theta_roi_parallel(
                    session_data=session_data,
                    roi_or_channels=roi_or_channels,
                    freq_range=freq_range,
                    n_freqs=n_freqs,
                    window_duration=window_duration,
                    save_path=save_dir,
                    show_plots=show_plots,
                    n_jobs=channel_n_jobs,
                    parallel_method=parallel_method
                )
                
                print(f"\n🎉 Parallel single session analysis completed!")
                print(f"✓ ROI: {roi_or_channels} -> channels {results['roi_channels']}")
                print(f"✓ Parallel method: {results['parallel_method']}")
                print(f"✓ Jobs used: {results['n_jobs_used']}")
                print(f"✓ NM sizes analyzed: {list(results['normalized_windows'].keys())}")
            
            elif mode == 'multi':
                if rat_id is None:
                    print("❌ Error: rat_id is required for multi-session mode")
                    return False
                
                print(f"🚀 Starting parallel multi-session analysis...")
                print(f"  Rat ID: {rat_id}")
                print(f"  ROI: {roi_or_channels}")
                print(f"  Parallel method: {parallel_method}")
                print(f"  Session jobs: {session_n_jobs or 'auto'}")
                print(f"  Channel jobs: {channel_n_jobs or 'auto'}")
                
                # Set save path
                save_dir = save_path or f'results/multi_session/rat_{rat_id}_parallel_{parallel_method}'
                
                # Run analysis
                results = analyze_rat_multi_session_parallel(
                    rat_id=rat_id,
                    roi_or_channels=roi_or_channels,
                    pkl_path=pkl_path,
                    freq_range=freq_range,
                    n_freqs=n_freqs,
                    window_duration=window_duration,
                    save_path=save_dir,
                    show_plots=show_plots,
                    session_n_jobs=session_n_jobs,
                    channel_n_jobs=channel_n_jobs,
                    parallel_method=parallel_method
                )
                
                print(f"\n🎉 Parallel multi-session analysis completed!")
                print(f"✓ Rat ID: {rat_id}")
                print(f"✓ ROI: {roi_or_channels} -> channels {results['roi_channels']}")
                print(f"✓ Sessions analyzed: {results['n_sessions_analyzed']}")
                print(f"✓ Parallel method: {results['parallel_method']}")
                print(f"✓ Processing time: {results['processing_time']:.2f}s")
                print(f"✓ NM sizes found: {[float(key) for key in results['aggregated_windows'].keys()]}")
            return True
    except Exception as e:
        print(f"❌ Error in parallel analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)