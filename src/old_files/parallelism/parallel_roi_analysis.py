#!/usr/bin/env python3
"""
Parallel ROI Analysis Module

This module contains parallel versions of key functions from nm_theta_analysis.py
for performance testing and validation.

Key parallelization targets:
1. Channel processing in compute_roi_theta_spectrogram()
2. Frequency processing within each channel
3. Multi-session processing
"""

import numpy as np
import time
import pickle
from typing import List, Tuple, Dict, Optional, Union
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import os
import sys
from functools import partial
import mne

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from eeg_analysis_package.time_frequency import morlet_spectrogram
    from nm_theta_single_basic import (
        compute_global_statistics,
        extract_nm_event_windows,
        analyze_session_nm_theta_roi
    )
    from electrode_utils import get_channels, load_electrode_mappings
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Make sure you're running from the project root directory")


def process_single_channel_parallel(args):
    """
    Worker function for parallel channel processing.
    
    Parameters:
    -----------
    args : tuple
        (channel_idx, eeg_channel_data, freqs, n_cycles, sfreq)
    
    Returns:
    --------
    channel_results : dict
        Results for this channel including power, stats, and metadata
    """
    ch_idx, eeg_channel, freqs, n_cycles, sfreq = args
    
    # OPTIMIZED: Process all frequencies at once with adaptive n_cycles
    data = eeg_channel[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles, 
        output='power', zero_mean=True
    )
    channel_power = power[0, 0, :, :]  # (n_freqs, n_times)
    
    # Compute per-channel statistics
    ch_mean = np.mean(channel_power, axis=1)
    ch_std = np.std(channel_power, axis=1)
    ch_std = np.maximum(ch_std, 1e-12)
    
    # Z-score normalization
    normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
    
    return {
        'channel_idx': ch_idx,
        'power': channel_power,
        'normalized_power': normalized_power,
        'mean': ch_mean,
        'std': ch_std,
        'power_range': (channel_power.min(), channel_power.max()),
        'zscore_range': (normalized_power.min(), normalized_power.max())
    }


def process_frequency_batch_parallel(args):
    """
    Worker function for parallel frequency processing.
    
    Parameters:
    -----------
    args : tuple
        (freq_batch, cycles_batch, eeg_channel, sfreq)
    
    Returns:
    --------
    power_batch : np.ndarray
        Power for this frequency batch (n_freqs_batch, n_times)
    """
    freq_batch, cycles_batch, eeg_channel, sfreq = args
    
    # OPTIMIZED: Process frequency batch at once
    data = eeg_channel[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=sfreq, freqs=freq_batch, n_cycles=cycles_batch, 
        output='power', zero_mean=True
    )
    return power[0, 0, :, :]  # (n_freqs_batch, n_times)


def compute_roi_theta_spectrogram_parallel_channels(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    sfreq: float = 200.0,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    n_cycles_factor: float = 3.0,
    n_jobs: Optional[int] = None,
    method: str = 'multiprocessing'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel version of compute_roi_theta_spectrogram with channel-level parallelization.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data (n_channels, n_samples)
    roi_channels : List[int]
        List of channel indices
    sfreq : float
        Sampling frequency
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    n_freqs : int
        Number of frequencies
    n_cycles_factor : float
        Cycles factor
    n_jobs : Optional[int]
        Number of parallel jobs (None = auto-detect)
    method : str
        'multiprocessing' or 'threading'
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    roi_power : np.ndarray
        Average ROI power
    channel_powers : np.ndarray
        Individual channel powers
    """
    print(f"🔄 PARALLEL CHANNEL PROCESSING ({method})")
    print(f"   Channels: {len(roi_channels)}, Jobs: {n_jobs or 'auto'}")
    
    # Setup
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(roi_channels))
    
    # Prepare arguments for parallel processing
    channel_args = []
    for ch_idx in roi_channels:
        eeg_channel = eeg_data[ch_idx, :]
        channel_args.append((ch_idx, eeg_channel, freqs, n_cycles, sfreq))
    
    # Process channels in parallel
    start_time = time.time()
    
    if method == 'multiprocessing':
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_single_channel_parallel, channel_args))
    elif method == 'threading':
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(process_single_channel_parallel, channel_args))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Parallel processing time: {processing_time:.2f}s")
    
    # Combine results
    channel_powers = []
    normalized_powers = []
    
    for result in results:
        channel_powers.append(result['power'])
        normalized_powers.append(result['normalized_power'])
        print(f"  Channel {result['channel_idx']}: "
              f"Power range: {result['power_range'][0]:.2e} - {result['power_range'][1]:.2e}, "
              f"Z-score range: {result['zscore_range'][0]:.2f} - {result['zscore_range'][1]:.2f}")
    
    # Convert to arrays and compute ROI average
    channel_powers = np.array(channel_powers)  # (n_channels, n_freqs, n_times)
    normalized_powers = np.array(normalized_powers)
    roi_power = np.mean(normalized_powers, axis=0)  # Average across channels
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    
    return freqs, roi_power, channel_powers


def compute_roi_theta_spectrogram_parallel_frequencies(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    sfreq: float = 200.0,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    n_cycles_factor: float = 3.0,
    n_jobs: Optional[int] = None,
    batch_size: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel version with frequency-level parallelization within each channel.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data (n_channels, n_samples)
    roi_channels : List[int]
        List of channel indices
    sfreq : float
        Sampling frequency
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    n_freqs : int
        Number of frequencies
    n_cycles_factor : float
        Cycles factor
    n_jobs : Optional[int]
        Number of parallel jobs
    batch_size : int
        Number of frequencies to process in each batch
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    roi_power : np.ndarray
        Average ROI power
    channel_powers : np.ndarray
        Individual channel powers
    """
    print(f"🔄 PARALLEL FREQUENCY PROCESSING")
    print(f"   Channels: {len(roi_channels)}, Batch size: {batch_size}")
    
    # Setup
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    channel_powers = []
    normalized_powers = []
    
    start_time = time.time()
    
    for i, ch_idx in enumerate(roi_channels):
        print(f"Processing channel {ch_idx} ({i+1}/{len(roi_channels)})")
        eeg_channel = eeg_data[ch_idx, :]
        
        # Create frequency batches
        freq_batches = []
        cycles_batches = []
        
        for j in range(0, len(freqs), batch_size):
            end_idx = min(j + batch_size, len(freqs))
            freq_batches.append(freqs[j:end_idx])
            cycles_batches.append(n_cycles[j:end_idx])
        
        # Prepare arguments for parallel processing
        batch_args = [(freq_batch, cycles_batch, eeg_channel, sfreq) 
                     for freq_batch, cycles_batch in zip(freq_batches, cycles_batches)]
        
        # Process frequency batches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            batch_results = list(executor.map(process_frequency_batch_parallel, batch_args))
        
        # Combine batch results
        channel_power = np.concatenate(batch_results, axis=0)
        
        # Normalize
        ch_mean = np.mean(channel_power, axis=1)
        ch_std = np.std(channel_power, axis=1)
        ch_std = np.maximum(ch_std, 1e-12)
        normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
        
        channel_powers.append(channel_power)
        normalized_powers.append(normalized_power)
        
        print(f"  Channel {ch_idx} power range: {channel_power.min():.2e} - {channel_power.max():.2e}")
        print(f"  Channel {ch_idx} z-score range: {normalized_power.min():.2f} - {normalized_power.max():.2f}")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Parallel frequency processing time: {processing_time:.2f}s")
    
    # Convert to arrays and compute ROI average
    channel_powers = np.array(channel_powers)
    normalized_powers = np.array(normalized_powers)
    roi_power = np.mean(normalized_powers, axis=0)
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    
    return freqs, roi_power, channel_powers


def process_single_session_parallel(args):
    """
    Worker function for parallel multi-session processing.
    
    Parameters:
    -----------
    args : tuple
        (session_data, roi_or_channels, freq_range, n_freqs, window_duration, 
         n_cycles_factor, mapping_df, session_idx)
    
    Returns:
    --------
    session_result : dict
        Analysis results for this session
    """
    (session_data, roi_or_channels, freq_range, n_freqs, 
     window_duration, n_cycles_factor, mapping_df, session_idx) = args
    
    print(f"🔄 Processing session {session_idx} in parallel...")
    
    try:
        result = analyze_session_nm_theta_roi(
            session_data=session_data,
            roi_or_channels=roi_or_channels,
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path=None,  # Don't save individual sessions in parallel test
            mapping_df=mapping_df,
            show_plots=False
        )
        
        return {
            'session_idx': session_idx,
            'success': True,
            'result': result,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ Error in session {session_idx}: {e}")
        return {
            'session_idx': session_idx,
            'success': False,
            'result': None,
            'error': str(e)
        }


def analyze_rat_multi_session_parallel(
    rat_sessions: List[Tuple[int, Dict]],
    roi_or_channels: Union[str, List[int]],
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    mapping_df: Optional = None,
    n_jobs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Dict:
    """
    Parallel version of multi-session analysis.
    
    Parameters:
    -----------
    rat_sessions : List[Tuple[int, Dict]]
        List of (session_index, session_data) tuples
    roi_or_channels : Union[str, List[int]]
        ROI specification
    freq_range : Tuple[float, float]
        Frequency range
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Cycles factor
    mapping_df : Optional
        Electrode mapping dataframe
    n_jobs : Optional[int]
        Number of parallel jobs
    batch_size : Optional[int]
        Number of sessions to process in each batch
    
    Returns:
    --------
    results : Dict
        Combined multi-session results
    """
    print(f"🔄 PARALLEL MULTI-SESSION ANALYSIS")
    print(f"   Sessions: {len(rat_sessions)}, Jobs: {n_jobs or 'auto'}")
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(rat_sessions))
    
    if batch_size is None:
        batch_size = max(1, len(rat_sessions) // n_jobs)
    
    # Prepare arguments for parallel processing
    session_args = []
    for session_idx, session_data in rat_sessions:
        args = (session_data, roi_or_channels, freq_range, n_freqs, 
                window_duration, n_cycles_factor, mapping_df, session_idx)
        session_args.append(args)
    
    # Process sessions in parallel
    start_time = time.time()
    
    successful_results = []
    failed_sessions = []
    
    # Process in batches to control memory usage
    for i in range(0, len(session_args), batch_size):
        batch_args = session_args[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(session_args)-1)//batch_size + 1}")
        
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(batch_args))) as executor:
            batch_results = list(executor.map(process_single_session_parallel, batch_args))
        
        # Collect results
        for result in batch_results:
            if result['success']:
                successful_results.append(result['result'])
            else:
                failed_sessions.append(result['session_idx'])
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Parallel multi-session processing time: {processing_time:.2f}s")
    print(f"   ✅ Successful: {len(successful_results)}, ❌ Failed: {len(failed_sessions)}")
    
    if failed_sessions:
        print(f"   Failed sessions: {failed_sessions}")
    
    # Combine results (simplified version for testing)
    combined_results = {
        'successful_sessions': len(successful_results),
        'failed_sessions': len(failed_sessions),
        'processing_time': processing_time,
        'session_results': successful_results,
        'roi_or_channels': roi_or_channels,
        'freq_range': freq_range,
        'n_freqs': n_freqs
    }
    
    return combined_results


if __name__ == "__main__":
    print("Parallel ROI Analysis Module")
    print("Available functions:")
    print("- compute_roi_theta_spectrogram_parallel_channels()")
    print("- compute_roi_theta_spectrogram_parallel_frequencies()")
    print("- analyze_rat_multi_session_parallel()")
    print(f"Detected {cpu_count()} CPU cores")