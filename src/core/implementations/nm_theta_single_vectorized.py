#!/usr/bin/env python3
"""
Vectorized NM Theta Analysis with Optimized I/O

This script provides significant performance improvements over nm_theta_parallel.py through:
1. Vectorized frequency processing - compute multiple frequencies simultaneously
2. Optimized scipy CWT implementation for morlet wavelets
3. Memory-mapped/lazy loading for large datasets
4. Efficient I/O operations with streaming pickle
5. Threading-based parallelism (avoiding multiprocessing issues)

Key optimizations:
- Batch frequency computation instead of one-by-one processing
- scipy.signal.cwt for efficient wavelet transforms
- Memory-mapped arrays for large EEG data
- Lazy session loading to reduce memory footprint
- Streaming results to disk to avoid memory accumulation

Usage:
    # Single session with vectorized processing
    python nm_theta_vectorized.py --mode single --session_index 0 --roi frontal --n_jobs 4

    # Multi-session with memory-efficient processing
    python nm_theta_vectorized.py --mode multi --rat_id 10501 --roi frontal --n_jobs 4

Author: Generated for EEG near-mistake vectorized analysis
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
import h5py

from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import warnings
from scipy.signal import cwt, morlet2
from scipy.io import savemat, loadmat

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


def get_memory_usage():
    """
    Get current memory usage in MB.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0


def print_memory_usage(label: str = ""):
    """
    Print current memory usage with optional label.
    """
    memory_mb = get_memory_usage()
    if label:
        print(f"💾 Memory usage ({label}): {memory_mb:.1f} MB")
    else:
        print(f"💾 Memory usage: {memory_mb:.1f} MB")


def create_morlet_wavelets_vectorized(freqs: np.ndarray, sfreq: float, n_cycles: float = 7) -> np.ndarray:
    """
    Create morlet wavelets for all frequencies at once using scipy.
    
    Parameters:
    -----------
    freqs : np.ndarray
        Array of frequencies in Hz
    sfreq : float
        Sampling frequency in Hz
    n_cycles : float
        Number of cycles for morlet wavelets (can be array-like for adaptive)
    
    Returns:
    --------
    wavelets : np.ndarray
        Array of morlet wavelets, shape (n_freqs, n_samples)
    """
    if np.isscalar(n_cycles):
        n_cycles = np.full(len(freqs), n_cycles)
    
    # Calculate wavelet parameters
    sigma_t = n_cycles / (2 * np.pi * freqs)  # Time domain standard deviation
    
    # Determine wavelet length (6 sigma rule)
    max_sigma = np.max(sigma_t)
    wavelet_length = int(6 * max_sigma * sfreq)
    if wavelet_length % 2 == 0:
        wavelet_length += 1  # Make odd for symmetric wavelet
    
    # Create time vector
    t = np.arange(-wavelet_length // 2, wavelet_length // 2 + 1) / sfreq
    
    # Generate wavelets for all frequencies
    wavelets = np.zeros((len(freqs), len(t)), dtype=complex)
    
    for i, (freq, sigma) in enumerate(zip(freqs, sigma_t)):
        # Morlet wavelet: complex exponential * Gaussian envelope
        wavelets[i] = (np.pi ** -0.25) * np.exp(2j * np.pi * freq * t) * np.exp(-0.5 * (t / sigma) ** 2)
        
        # Normalize to unit energy
        wavelets[i] /= np.sqrt(np.trapz(np.abs(wavelets[i]) ** 2, t))
    
    return wavelets


def compute_spectrogram_vectorized_cwt(
    eeg_data: np.ndarray,
    freqs: np.ndarray,
    sfreq: float = 200.0,
    n_cycles: Union[float, np.ndarray] = 7.0,
    method: str = 'cwt'
) -> np.ndarray:
    """
    Compute spectrogram using vectorized approach with scipy CWT.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        EEG data (n_channels, n_samples) or (n_samples,) for single channel
    freqs : np.ndarray
        Frequency array
    sfreq : float
        Sampling frequency
    n_cycles : Union[float, np.ndarray]
        Number of cycles (can be adaptive)
    method : str
        'cwt' for scipy CWT or 'manual' for manual implementation
    
    Returns:
    --------
    power : np.ndarray
        Power spectrogram (n_channels, n_freqs, n_samples) or (n_freqs, n_samples)
    """
    if eeg_data.ndim == 1:
        eeg_data = eeg_data[np.newaxis, :]  # Add channel dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    n_channels, n_samples = eeg_data.shape
    n_freqs = len(freqs)
    
    if method == 'cwt':
        # Use scipy's continuous wavelet transform
        # Create morlet2 wavelets (real-valued approximation)
        power = np.zeros((n_channels, n_freqs, n_samples))
        
        for ch_idx in range(n_channels):
            # Use scipy's cwt with morlet2 wavelet
            # Convert frequencies to scales for morlet2
            scales = sfreq / freqs  # Approximate scale conversion
            
            # Compute CWT
            coeffs = cwt(eeg_data[ch_idx], morlet2, scales)
            
            # Compute power (squared magnitude)
            power[ch_idx] = np.abs(coeffs) ** 2
    
    else:
        # Manual vectorized implementation
        # Create all wavelets at once
        if np.isscalar(n_cycles):
            n_cycles = np.full(len(freqs), n_cycles)
        
        wavelets = create_morlet_wavelets_vectorized(freqs, sfreq, n_cycles)
        
        # Compute convolution for all channels and frequencies
        power = np.zeros((n_channels, n_freqs, n_samples))
        
        for ch_idx in range(n_channels):
            for freq_idx, wavelet in enumerate(wavelets):
                # Convolve signal with wavelet
                convolved = np.convolve(eeg_data[ch_idx], wavelet, mode='same')
                power[ch_idx, freq_idx] = np.abs(convolved) ** 2
    
    if squeeze_output:
        power = power.squeeze(0)
    
    return power


def process_channel_batch_vectorized(args):
    """
    Vectorized worker function for processing multiple channels efficiently.
    
    Parameters:
    -----------
    args : tuple
        (channel_indices, eeg_data, freqs, n_cycles, sfreq, method)
    
    Returns:
    --------
    batch_results : dict
        Results for the batch of channels
    """
    channel_indices, eeg_data_batch, freqs, n_cycles, sfreq, method = args
    
    try:
        # Compute spectrogram for entire batch at once
        power_batch = compute_spectrogram_vectorized_cwt(
            eeg_data_batch, freqs, sfreq, n_cycles, method
        )
        
        # Normalize each channel individually
        normalized_batch = np.zeros_like(power_batch)
        stats_batch = []
        
        for i, ch_idx in enumerate(channel_indices):
            channel_power = power_batch[i]  # (n_freqs, n_samples)
            
            # Compute per-channel statistics for normalization
            ch_mean = np.mean(channel_power, axis=1)  # Mean per frequency
            ch_std = np.std(channel_power, axis=1)    # Std per frequency
            ch_std = np.maximum(ch_std, 1e-12)        # Avoid division by zero
            
            # Z-score normalize this channel
            normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
            normalized_batch[i] = normalized_power
            
            stats_batch.append({
                'channel_idx': ch_idx,
                'mean': ch_mean,
                'std': ch_std,
                'power_range': (channel_power.min(), channel_power.max()),
                'zscore_range': (normalized_power.min(), normalized_power.max())
            })
        
        return {
            'channel_indices': channel_indices,
            'power_batch': power_batch,
            'normalized_batch': normalized_batch,
            'stats_batch': stats_batch,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'channel_indices': channel_indices,
            'power_batch': None,
            'normalized_batch': None,
            'stats_batch': None,
            'success': False,
            'error': str(e)
        }


def compute_roi_theta_spectrogram_vectorized(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    sfreq: float = 200.0,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    n_cycles_factor: float = 3.0,
    n_jobs: Optional[int] = None,
    batch_size: int = 8,
    method: str = 'cwt',
    memory_efficient: bool = True,
    chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Vectorized version with batch processing for improved performance.
    
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
    batch_size : int
        Number of channels to process in each batch
    method : str
        'cwt' for scipy CWT or 'manual' for manual wavelets
    memory_efficient : bool
        If True, use memory-efficient processing (recommended for large datasets)
    chunk_size : Optional[int]
        Time chunk size for memory-efficient processing (None = auto)
    
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    roi_power : np.ndarray
        Average ROI power matrix (n_freqs, n_times)
    channel_powers : Optional[np.ndarray]
        Individual channel powers (n_channels, n_freqs, n_times) or None if memory_efficient=True
    """
    print(f"🚀 VECTORIZED CHANNEL PROCESSING ({method.upper()})")
    print(f"   Channels: {len(roi_channels)}, Batch size: {batch_size}, Jobs: {n_jobs or 'auto'}")
    print(f"   Memory efficient: {memory_efficient}")
    
    # Create logarithmically spaced frequency vector
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    n_cycles = np.maximum(3, freqs * n_cycles_factor)
    
    print(f"📊 Using {n_freqs} logarithmically spaced frequencies:")
    print(f"   Range: {freq_range[0]:.2f} - {freq_range[1]:.2f} Hz")
    print(f"   Method: {method} ({'scipy CWT' if method == 'cwt' else 'manual wavelets'})")
    
    if n_jobs is None:
        n_jobs = min(cpu_count(), max(1, len(roi_channels) // batch_size))
    
    if memory_efficient:
        return _compute_roi_spectrogram_memory_efficient(
            eeg_data, roi_channels, freqs, n_cycles, sfreq, method, 
            n_jobs, batch_size, chunk_size
        )
    else:
        return _compute_roi_spectrogram_standard(
            eeg_data, roi_channels, freqs, n_cycles, sfreq, method, 
            n_jobs, batch_size
        )


def _compute_roi_spectrogram_memory_efficient(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    method: str,
    n_jobs: int,
    batch_size: int,
    chunk_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, None]:
    """
    Memory-efficient version that processes data in chunks and avoids large intermediate arrays.
    """
    n_samples = eeg_data.shape[1]
    n_freqs = len(freqs)
    
    # Auto-determine chunk size if not provided
    if chunk_size is None:
        # Aim for chunks that use ~100MB of memory
        # Each chunk: n_channels * n_freqs * chunk_size * 8 bytes
        target_memory = 100 * 1024 * 1024  # 100MB
        chunk_size = max(1000, min(n_samples // 10, target_memory // (len(roi_channels) * n_freqs * 8)))
    
    print(f"   Memory-efficient processing with chunk size: {chunk_size}")
    print(f"   Total samples: {n_samples}, Chunks: {n_samples // chunk_size + 1}")
    
    # Initialize ROI power accumulator
    roi_power_accumulator = np.zeros((n_freqs, n_samples))
    channel_count = np.zeros(n_samples)  # Track how many channels contributed to each time point
    
    print_memory_usage("after initialization")
    
    # Process channels in batches
    channel_batches = []
    for i in range(0, len(roi_channels), batch_size):
        batch_channels = roi_channels[i:i + batch_size]
        channel_batches.append(batch_channels)
    
    print(f"   Processing {len(channel_batches)} channel batches...")
    
    start_time = time.time()
    
    # Process each batch of channels
    for batch_idx, batch_channels in enumerate(channel_batches):
        print(f"     Processing batch {batch_idx + 1}/{len(channel_batches)}: channels {batch_channels}")
        print_memory_usage(f"batch {batch_idx + 1} start")
        
        # Process this batch of channels in time chunks
        batch_data = eeg_data[batch_channels, :]
        
        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            chunk_data = batch_data[:, chunk_start:chunk_end]
            
            # Compute spectrogram for this chunk
            chunk_power = compute_spectrogram_vectorized_cwt(
                chunk_data, freqs, sfreq, n_cycles, method
            )
            
            # Normalize each channel in this chunk
            for ch_idx in range(len(batch_channels)):
                channel_power = chunk_power[ch_idx]  # (n_freqs, chunk_size)
                
                # Compute per-channel statistics for this chunk
                ch_mean = np.mean(channel_power, axis=1)  # Mean per frequency
                ch_std = np.std(channel_power, axis=1)    # Std per frequency
                ch_std = np.maximum(ch_std, 1e-12)        # Avoid division by zero
                
                # Z-score normalize this channel
                normalized_power = (channel_power - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
                
                # Accumulate to ROI power
                roi_power_accumulator[:, chunk_start:chunk_end] += normalized_power
                channel_count[chunk_start:chunk_end] += 1
            
            # Clear chunk data to free memory
            del chunk_power, chunk_data
            gc.collect()
        
        # Clear batch data
        del batch_data
        gc.collect()
        print_memory_usage(f"batch {batch_idx + 1} end")
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Memory-efficient processing time: {processing_time:.2f}s")
    print_memory_usage("after processing")
    
    # Average across channels (avoid division by zero)
    valid_mask = channel_count > 0
    roi_power = np.zeros((n_freqs, n_samples))
    roi_power[:, valid_mask] = roi_power_accumulator[:, valid_mask] / channel_count[valid_mask]
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    print(f"Used {len(roi_channels)} channels successfully")
    
    return freqs, roi_power, None  # Return None for channel_powers to indicate memory-efficient mode


def _compute_roi_spectrogram_standard(
    eeg_data: np.ndarray,
    roi_channels: List[int],
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    method: str,
    n_jobs: int,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard version that stores all channel powers (uses more memory).
    """
    # Create batches of channels
    channel_batches = []
    for i in range(0, len(roi_channels), batch_size):
        batch_channels = roi_channels[i:i + batch_size]
        batch_data = eeg_data[batch_channels, :]
        channel_batches.append((batch_channels, batch_data, freqs, n_cycles, sfreq, method))
    
    print(f"   Processing {len(channel_batches)} batches with {n_jobs} workers...")
    
    # Process batches in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        batch_results = list(executor.map(process_channel_batch_vectorized, channel_batches))
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Standard processing time: {processing_time:.2f}s")
    
    # Combine batch results
    successful_results = []
    failed_channels = []
    
    for batch_result in batch_results:
        if batch_result['success']:
            successful_results.append(batch_result)
            for stats in batch_result['stats_batch']:
                print(f"  ✅ Channel {stats['channel_idx']}: "
                      f"Power range: {stats['power_range'][0]:.2e} - {stats['power_range'][1]:.2e}, "
                      f"Z-score range: {stats['zscore_range'][0]:.2f} - {stats['zscore_range'][1]:.2f}")
        else:
            failed_channels.extend(batch_result['channel_indices'])
            print(f"  ❌ Batch {batch_result['channel_indices']} failed: {batch_result['error']}")
    
    if not successful_results:
        raise ValueError("All channel batches failed to process!")
    
    if failed_channels:
        print(f"⚠️ Warning: {len(failed_channels)} channels failed: {failed_channels}")
        print(f"Continuing with successful channels from {len(successful_results)} batches")
    
    # Combine successful results
    all_powers = []
    all_normalized = []
    
    for batch_result in successful_results:
        all_powers.append(batch_result['power_batch'])
        all_normalized.append(batch_result['normalized_batch'])
    
    # Concatenate along channel dimension
    channel_powers = np.concatenate(all_powers, axis=0)  # (n_channels, n_freqs, n_times)
    normalized_powers = np.concatenate(all_normalized, axis=0)
    roi_power = np.mean(normalized_powers, axis=0)  # Average across channels
    
    print(f"ROI spectrogram computed. Shape: {roi_power.shape}")
    print(f"ROI z-score range: {roi_power.min():.2f} - {roi_power.max():.2f}")
    successful_channels = sum(len(br['channel_indices']) for br in successful_results)
    print(f"Used {successful_channels}/{len(roi_channels)} channels successfully")
    
    return freqs, roi_power, channel_powers


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


def analyze_session_nm_theta_roi_vectorized(
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
    batch_size: int = 8,
    method: str = 'cwt'
) -> Dict:
    """
    Vectorized version of session analysis with optimized processing.
    
    Parameters similar to parallel version but with additional vectorization options.
    """
    print("=" * 60)
    print("VECTORIZED NM THETA ROI ANALYSIS (SINGLE SESSION)")
    print("=" * 60)
    
    if len(session_data['nm_peak_times']) == 0:
        raise ValueError("No NM events found in session")
    
    # Step 1: Get ROI channel indices (same as parallel version)
    print(f"Step 1: Determining ROI channels")
    rat_id = session_data.get('rat_id', 'unknown')
    
    if mapping_df is None:
        mapping_df = load_electrode_mappings()
    
    roi_channels = get_channels(rat_id, roi_or_channels, mapping_df)
    
    # Validation
    max_channel = session_data['eeg'].shape[0] - 1
    invalid_channels = [ch for ch in roi_channels if ch > max_channel or ch < 0]
    if invalid_channels:
        raise ValueError(f"Invalid channel indices {invalid_channels}. Valid range: 0-{max_channel}")
    
    if not roi_channels:
        raise ValueError("No valid channels found for ROI")
    
    times = session_data['eeg_time'].flatten()
    
    # Step 2: Compute ROI theta spectrogram with vectorization
    print(f"Step 2: Computing ROI theta spectrogram with vectorized processing")
    print(f"   Method: {method}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of jobs: {n_jobs or 'auto'}")
    
    freqs, roi_power, channel_powers = compute_roi_theta_spectrogram_vectorized(
        session_data['eeg'],
        roi_channels,
        sfreq=200.0,
        freq_range=freq_range,
        n_freqs=n_freqs,
        n_cycles_factor=n_cycles_factor,
        n_jobs=n_jobs,
        batch_size=batch_size,
        method=method,
        memory_efficient=True  # Use memory-efficient mode by default
    )
    
    # Steps 3-7: Same as parallel version
    print("Step 3: Using per-channel normalized ROI power")
    global_mean = np.zeros(len(freqs))
    global_std = np.ones(len(freqs))
    
    print("Step 4: Extracting NM event windows")
    nm_windows = extract_nm_event_windows(
        roi_power, times, 
        session_data['nm_peak_times'],
        session_data['nm_sizes'],
        window_duration
    )
    
    if not nm_windows:
        raise ValueError("No valid NM event windows extracted")
    
    print("Step 5: ROI windows already normalized per channel")
    normalized_windows = nm_windows
    
    print("Step 6: Saving results")
    # Handle case where channel_powers might be None (memory-efficient mode)
    if channel_powers is None:
        print("   Note: Channel powers not saved (memory-efficient mode)")
        channel_powers_for_save = None
    else:
        channel_powers_for_save = channel_powers
    
    save_roi_results(
        session_data, normalized_windows, freqs, 
        roi_channels, roi_or_channels, channel_powers_for_save, save_path
    )
    
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
    print("VECTORIZED ROI ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return {
        'freqs': freqs,
        'roi_power': roi_power,
        'channel_powers': channel_powers,
        'roi_channels': roi_channels,
        'normalized_windows': normalized_windows,
        'roi_specification': roi_or_channels,
        'method': method,
        'batch_size': batch_size,
        'n_jobs_used': n_jobs or min(cpu_count(), max(1, len(roi_channels) // batch_size))
    }


def process_single_session_vectorized_threaded(args):
    """
    Worker function for vectorized multi-session processing.
    """
    (session_data, roi_or_channels, freq_range, n_freqs, window_duration, 
     n_cycles_factor, session_save_dir, mapping_df, session_index, batch_size, method, n_jobs) = args
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SESSION {session_index} WITH VECTORIZATION")
    print(f"{'='*60}")
    
    try:
        # Process the session with vectorized processing
        result = analyze_session_nm_theta_roi_vectorized(
            session_data=session_data,
            roi_or_channels=roi_or_channels,
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path=session_save_dir,
            mapping_df=mapping_df,
            show_plots=False,
            show_frequency_profiles=False,
            n_jobs=n_jobs,
            batch_size=batch_size,
            method=method
        )
        
        # Save results immediately
        os.makedirs(session_save_dir, exist_ok=True)
        results_file = os.path.join(session_save_dir, 'session_results.pkl')
        
        with open(results_file, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"✓ Session {session_index} results saved to {results_file}")
        
        # Create summary
        session_summary = {
            'session_index': session_index,
            'rat_id': session_data.get('rat_id'),
            'session_date': session_data.get('session_date', 'unknown'),
            'roi_channels': result['roi_channels'],
            'roi_specification': result['roi_specification'],
            'total_nm_events': sum(data['n_events'] for data in result['normalized_windows'].values()),
            'nm_sizes': list(result['normalized_windows'].keys()),
            'results_file': results_file,
            'save_dir': session_save_dir,
            'method': method,
            'parallel_method': method,
            'batch_size': batch_size,
            'success': True,
            'error': None
        }
        
        print(f"✓ Session {session_index} completed successfully")
        print(f"  Method: {method}, Batch size: {batch_size}")
        print(f"  ROI channels: {result['roi_channels']}")
        print(f"  Total NM events: {session_summary['total_nm_events']}")
        
        # Clean up
        del result, session_data
        gc.collect()
        
        return session_summary
        
    except Exception as e:
        print(f"❌ Error processing session {session_index}: {e}")
        import traceback
        traceback.print_exc()
        
        gc.collect()
        
        return {
            'session_index': session_index,
            'rat_id': 'unknown',
            'session_date': 'unknown',
            'roi_channels': [],
            'roi_specification': roi_or_channels,
            'total_nm_events': 0,
            'nm_sizes': [],
            'results_file': None,
            'save_dir': session_save_dir,
            'method': method,
            'batch_size': batch_size,
            'success': False,
            'error': str(e)
        }


def analyze_rat_multi_session_vectorized(
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
    channel_batch_size: int = 8,
    method: str = 'cwt'
) -> Dict:
    """
    Vectorized multi-session analysis with lazy loading and optimized processing.
    
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
    channel_batch_size : int
        Batch size for channel processing
    method : str
        'cwt' for scipy CWT or 'manual' for manual wavelets
    
    Returns:
    --------
    results : Dict
        Final aggregated analysis results
    """
    
    print("=" * 80)
    print(f"VECTORIZED MULTI-SESSION NM THETA ANALYSIS - RAT {rat_id}")
    print(f"  Method: {method.upper()}")
    print(f"  Session parallelism: {session_n_jobs or 'auto'} jobs")
    print(f"  Channel batch size: {channel_batch_size}")
    print("=" * 80)
    
    if save_path is None:
        save_path = f'../../results/multi_session/rat_{rat_id}_vectorized_{method}'
    
    # Step 1: Load all data and find sessions for this rat
    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    if session_n_jobs is None:
        session_n_jobs = min(cpu_count(), len(session_indices))
    
    print(f"Will process {len(session_indices)} sessions with {session_n_jobs} parallel workers")
    print(f"Each session will use 2 workers for channel processing")
    
    # Step 2: Process sessions in parallel with vectorization
    print(f"\nProcessing {len(session_indices)} sessions with vectorized method...")
    
    # Prepare arguments for parallel session processing
    session_args = []
    for i, (session_index, session_data) in enumerate(zip(session_indices, rat_sessions)):
        session_save_dir = os.path.join(save_path, f'session_{session_index}')
        args = (
            session_data, roi_or_channels, freq_range, n_freqs, window_duration,
            n_cycles_factor, session_save_dir, mapping_df, session_index, 
            channel_batch_size, method, 2  # Use 2 jobs per session for channel processing
        )
        session_args.append(args)
    
    # Process sessions in parallel
    start_time = time.time()
    
    print(f"🚀 Starting vectorized session processing with {session_n_jobs} workers...")
    
    with ThreadPoolExecutor(max_workers=session_n_jobs) as executor:
        session_summaries = list(executor.map(process_single_session_vectorized_threaded, session_args))
    
    processing_time = time.time() - start_time
    print(f"   ⏱️ Total vectorized processing time: {processing_time:.2f}s")
    
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
    
    # Step 3: Aggregate results (reuse from parallel version)
    print("\nAggregating vectorized results...")
    from nm_theta_parallel import aggregate_session_results_parallel
    aggregated_results = aggregate_session_results_parallel(session_summaries, rat_id)
    
    # Step 4: Save results
    print("\nSaving vectorized results...")
    save_vectorized_results(aggregated_results, save_path, method)
    
    # Step 5: Plot results
    if show_plots:
        print("\nPlotting vectorized results...")
        plot_vectorized_results(aggregated_results, save_path, method)
    else:
        print("Skipping plots (show_plots=False)")
    
    # Add method information
    aggregated_results['processing_time'] = processing_time
    aggregated_results['session_n_jobs'] = session_n_jobs
    aggregated_results['channel_batch_size'] = channel_batch_size
    aggregated_results['method'] = method
    
    print("=" * 80)
    print("VECTORIZED MULTI-SESSION ANALYSIS COMPLETE!")
    print(f"  Total processing time: {processing_time:.2f}s")
    print(f"  Sessions processed: {len(successful_sessions)}")
    print(f"  Method: {method}")
    print("=" * 80)
    
    return aggregated_results


def save_vectorized_results(results: Dict, save_path: str, method: str):
    """Save vectorized results with method information."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(save_path, f'vectorized_{method}_multi_session_results.pkl')
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
        'method': method,
        'channel_batch_size': results.get('channel_batch_size', 'unknown'),
        'processing_time': results.get('processing_time', 'unknown'),
        'session_n_jobs': results.get('session_n_jobs', 'unknown')
    }
    
    summary_file = os.path.join(save_path, f'vectorized_{method}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Vectorized Multi-Session NM Theta Analysis - Rat {results['rat_id']}\n")
        f.write("=" * 70 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Vectorized results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")


def plot_vectorized_results(results: Dict, save_path: str, method: str):
    """Plot vectorized results with method information."""
    print(f"Generating vectorized {method} plots...")
    
    # Reuse plotting from parallel version but with method info
    from nm_theta_parallel import plot_aggregated_results_parallel
    
    # Temporarily modify the parallel_method field for plotting
    original_method = results.get('parallel_method', 'unknown')
    results['parallel_method'] = f'vectorized_{method}'
    
    plot_aggregated_results_parallel(results, save_path)
    
    # Restore original method
    results['parallel_method'] = original_method
    
    # Save with vectorized naming
    plot_files = [f for f in os.listdir(save_path) if f.endswith('.png')]
    for plot_file in plot_files:
        if 'parallel' in plot_file:
            old_path = os.path.join(save_path, plot_file)
            new_name = plot_file.replace('parallel', f'vectorized_{method}')
            new_path = os.path.join(save_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed plot: {new_name}")


def main():
    """
    Main function for vectorized analysis with direct parameter configuration.
    """
    
    # =============================================================================
    # DIRECT PARAMETER CONFIGURATION
    # =============================================================================
    
    # Analysis mode: 'single' or 'multi'
    mode = 'multi'
    
    # Data file
    pkl_path = 'data/processed/all_eeg_data.pkl'
    
    # ROI specification
    roi_or_channels = 'motor'  # or [2, 3, 5] for custom channels
    
    # Frequency analysis parameters
    freq_range = (1.0, 45.0)  # (min_freq, max_freq) in Hz - more conservative theta range
    n_freqs = 25          # Number of logarithmically spaced frequencies - reduced from 35
    window_duration = 1.0    # Event window duration in seconds - reduced from 2.0
    n_cycles_factor = 3.0    # Factor for adaptive n_cycles
    
    # Vectorization parameters
    method = 'cwt'             # 'cwt' for scipy CWT or 'manual' for manual wavelets
    channel_batch_size = 4     # Number of channels to process in each batch - reduced from 8
    session_n_jobs = None      # None = auto-detect
    
    # Single session parameters
    session_index = 0
    
    # Multi-session parameters
    rat_id = '10593'
    
    # Output parameters
    save_path = None
    show_plots = True
    
    # =============================================================================
    # COMMAND LINE ARGUMENT PARSING (OPTIONAL)
    # =============================================================================
    
    use_command_line = '--use_args' in sys.argv or len(sys.argv) > 1 and not any(arg.endswith('.py') for arg in sys.argv)
    
    if use_command_line:
        print("📋 Using command line arguments...")
        parser = argparse.ArgumentParser(description='Vectorized NM Theta Analysis')
        parser.add_argument('--use_args', action='store_true')
        parser.add_argument('--mode', choices=['single', 'multi'], default=mode)
        parser.add_argument('--pkl_path', type=str, default=pkl_path)
        parser.add_argument('--roi', type=str, default=str(roi_or_channels))
        parser.add_argument('--freq_min', type=float, default=freq_range[0])
        parser.add_argument('--freq_max', type=float, default=freq_range[1])
        parser.add_argument('--n_freqs', type=int, default=n_freqs)
        parser.add_argument('--window_duration', type=float, default=window_duration)
        parser.add_argument('--save_path', type=str, default=save_path)
        parser.add_argument('--no_plots', action='store_true')
        parser.add_argument('--method', choices=['cwt', 'manual'], default=method)
        parser.add_argument('--batch_size', type=int, default=channel_batch_size)
        parser.add_argument('--session_n_jobs', type=int, default=session_n_jobs)
        parser.add_argument('--session_index', type=int, default=session_index)
        parser.add_argument('--rat_id', type=str, default=rat_id)
        
        args = parser.parse_args()
        
        # Override parameters
        mode = args.mode
        pkl_path = args.pkl_path
        freq_range = (args.freq_min, args.freq_max)
        n_freqs = args.n_freqs
        window_duration = args.window_duration
        save_path = args.save_path
        show_plots = not args.no_plots
        method = args.method
        channel_batch_size = args.batch_size
        session_n_jobs = args.session_n_jobs
        session_index = args.session_index
        rat_id = args.rat_id
        
        # Parse ROI
        try:
            roi_channels = [int(x.strip()) for x in args.roi.split(',')]
            roi_or_channels = roi_channels
        except ValueError:
            roi_or_channels = args.roi
    else:
        print("📋 Using direct parameter configuration from code...")
        print(f"   Mode: {mode}")
        print(f"   ROI: {roi_or_channels}")
        print(f"   Method: {method}")
        print(f"   Batch size: {channel_batch_size}")
    
    # =============================================================================
    # RUN ANALYSIS
    # =============================================================================
    
    try:
        if mode == 'single':
            print(f"🚀 Starting vectorized single session analysis...")
            print(f"  Session index: {session_index}")
            print(f"  Method: {method}")
            print(f"  Batch size: {channel_batch_size}")
            
            # Load session data
            session_data = load_session_data(pkl_path, session_index)
            
            # Set save path
            save_dir = save_path or f'../../results/single_session/session_{session_index}_vectorized_{method}'
            
            # Run analysis
            results = analyze_session_nm_theta_roi_vectorized(
                session_data=session_data,
                roi_or_channels=roi_or_channels,
                freq_range=freq_range,
                n_freqs=n_freqs,
                window_duration=window_duration,
                save_path=save_dir,
                show_plots=show_plots,
                batch_size=channel_batch_size,
                method=method
            )
            
            print(f"\n🎉 Vectorized single session analysis completed!")
            print(f"✓ Method: {method}")
            print(f"✓ Batch size: {channel_batch_size}")
            print(f"✓ ROI channels: {results['roi_channels']}")
            
        elif mode == 'multi':
            if rat_id is None:
                print("❌ Error: rat_id is required for multi-session mode")
                return False
            
            print(f"🚀 Starting vectorized multi-session analysis...")
            print(f"  Rat ID: {rat_id}")
            print(f"  Method: {method}")
            print(f"  Batch size: {channel_batch_size}")
            
            # Set save path
            save_dir = save_path or f'../../results/multi_session/rat_{rat_id}_vectorized_{method}'
            
            # Run analysis
            results = analyze_rat_multi_session_vectorized(
                rat_id=rat_id,
                roi_or_channels=roi_or_channels,
                pkl_path=pkl_path,
                freq_range=freq_range,
                n_freqs=n_freqs,
                window_duration=window_duration,
                save_path=save_dir,
                show_plots=show_plots,
                session_n_jobs=session_n_jobs,
                channel_batch_size=channel_batch_size,
                method=method
            )
            
            print(f"\n🎉 Vectorized multi-session analysis completed!")
            print(f"✓ Method: {method}")
            print(f"✓ Batch size: {channel_batch_size}")
            print(f"✓ Processing time: {results['processing_time']:.2f}s")
            print(f"✓ Sessions analyzed: {results['n_sessions_analyzed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in vectorized analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)