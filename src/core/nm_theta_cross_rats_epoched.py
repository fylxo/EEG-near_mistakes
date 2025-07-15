#!/usr/bin/env python3
"""
Cross-Rats NM Theta Analysis - Epoched Approach

This script follows the paper's methodology:
1. Extract individual event epochs first (-2 to +2 seconds)
2. Compute spectrogram on each individual epoch
3. Clip to analysis window (-1 to +1 seconds)
4. Average across events

Author: Generated for EEG near-mistake analysis comparison
"""

import os
import sys
import pickle
import json
import gc
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict
import mne

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import AnalysisConfig, DataConfig, PlottingConfig

# Parula colormap data (abbreviated for brevity)
PARULA_DATA = [
    [0.2422, 0.1504, 0.6603], [0.2444, 0.1534, 0.6728], [0.2464, 0.1569, 0.6847],
    [0.2484, 0.1607, 0.6961], [0.2503, 0.1648, 0.7071], [0.2522, 0.1689, 0.7179],
    [0.254, 0.1732, 0.7286], [0.2558, 0.1773, 0.7393], [0.2576, 0.1814, 0.7501],
    [0.2594, 0.1854, 0.761], [0.2611, 0.1893, 0.7719], [0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937], [0.2661, 0.2011, 0.8043], [0.2676, 0.2052, 0.8148],
    [0.2691, 0.2094, 0.8249], [0.2704, 0.2138, 0.8346], [0.2717, 0.2184, 0.8439],
    [0.2729, 0.2231, 0.8528], [0.274, 0.228, 0.8612], [0.2749, 0.233, 0.8692],
    [0.2758, 0.2382, 0.8767], [0.2766, 0.2435, 0.884], [0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973], [0.2788, 0.2598, 0.9035], [0.2794, 0.2653, 0.9094],
    [0.2798, 0.2708, 0.915], [0.2802, 0.2764, 0.9204], [0.2806, 0.2819, 0.9255],
    [0.2809, 0.2875, 0.9305], [0.2811, 0.293, 0.9352], [0.2813, 0.2985, 0.9397],
    [0.2814, 0.304, 0.9441], [0.2814, 0.3095, 0.9483], [0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563], [0.2809, 0.3259, 0.96], [0.2807, 0.3313, 0.9636],
    [0.2803, 0.3367, 0.967], [0.2798, 0.3421, 0.9702], [0.2791, 0.3475, 0.9733],
    [0.2784, 0.3529, 0.9763], [0.2776, 0.3583, 0.9791], [0.2766, 0.3638, 0.9817],
    [0.2754, 0.3693, 0.984], [0.2741, 0.3748, 0.9862], [0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898], [0.2691, 0.3916, 0.9912], [0.267, 0.3973, 0.9924],
    [0.2647, 0.403, 0.9935], [0.2621, 0.4088, 0.9946], [0.2591, 0.4145, 0.9955],
    [0.2556, 0.4203, 0.9965], [0.2517, 0.4261, 0.9974], [0.2473, 0.4319, 0.9983],
    [0.2424, 0.4378, 0.9991], [0.2369, 0.4437, 0.9996], [0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985], [0.2189, 0.462, 0.9968], [0.2128, 0.4682, 0.9948],
    [0.2066, 0.4743, 0.9926], [0.2006, 0.4803, 0.9906], [0.195, 0.4861, 0.9887],
    [0.1903, 0.4919, 0.9867], [0.1869, 0.4975, 0.9844], [0.1847, 0.503, 0.9819],
    [0.1831, 0.5084, 0.9793], [0.1818, 0.5138, 0.9766], [0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709], [0.1785, 0.5296, 0.9677], [0.1778, 0.5349, 0.9641],
    [0.1773, 0.5401, 0.9602], [0.1768, 0.5452, 0.956], [0.1764, 0.5504, 0.9516],
    [0.1755, 0.5554, 0.9473], [0.174, 0.5605, 0.9432], [0.1716, 0.5655, 0.9393],
    [0.1686, 0.5705, 0.9357], [0.1649, 0.5755, 0.9323], [0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254], [0.154, 0.5902, 0.9218], [0.1513, 0.595, 0.9182],
    [0.1492, 0.5997, 0.9147], [0.1475, 0.6043, 0.9113], [0.1461, 0.6089, 0.908],
    [0.1446, 0.6135, 0.905], [0.1429, 0.618, 0.9022], [0.1408, 0.6226, 0.8998],
    [0.1383, 0.6272, 0.8975], [0.1354, 0.6317, 0.8953], [0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891], [0.1253, 0.6453, 0.8887], [0.1219, 0.6497, 0.8862],
    [0.1185, 0.6541, 0.8834], [0.1152, 0.6584, 0.8804], [0.1119, 0.6627, 0.877],
    [0.1085, 0.6669, 0.8734], [0.1048, 0.671, 0.8695], [0.1009, 0.675, 0.8653],
    [0.0964, 0.6789, 0.8609], [0.0914, 0.6828, 0.8562], [0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462], [0.0713, 0.6938, 0.8409], [0.0628, 0.6972, 0.8355],
    [0.0535, 0.7006, 0.8299], [0.0433, 0.7039, 0.8242], [0.0328, 0.7071, 0.8183],
    [0.0234, 0.7103, 0.8124], [0.0155, 0.7133, 0.8064], [0.0091, 0.7163, 0.8003],
    [0.0046, 0.7192, 0.7941], [0.0019, 0.722, 0.7878], [0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752], [0.0046, 0.7301, 0.7688], [0.0094, 0.7327, 0.7623],
    [0.0162, 0.7352, 0.7558], [0.0253, 0.7376, 0.7492], [0.0369, 0.74, 0.7426],
    [0.0504, 0.7423, 0.7359], [0.0638, 0.7446, 0.7292], [0.077, 0.7468, 0.7224],
    [0.0899, 0.7489, 0.7156], [0.1023, 0.751, 0.7088], [0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695], [0.1354, 0.7572, 0.6881], [0.1448, 0.7593, 0.6812],
    [0.1532, 0.7614, 0.6741], [0.1609, 0.7635, 0.6671], [0.1678, 0.7656, 0.6599],
    [0.1741, 0.7678, 0.6527], [0.1799, 0.7699, 0.6454], [0.1853, 0.7721, 0.6379],
    [0.1905, 0.7743, 0.6303], [0.1954, 0.7765, 0.6225], [0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065], [0.2118, 0.7828, 0.5983], [0.2178, 0.7849, 0.5899],
    [0.2244, 0.7869, 0.5813], [0.2318, 0.7887, 0.5725], [0.2401, 0.7905, 0.5636],
    [0.2491, 0.7922, 0.5546], [0.2589, 0.7937, 0.5454], [0.2695, 0.7951, 0.536],
    [0.2809, 0.7964, 0.5266], [0.2929, 0.7975, 0.517], [0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975], [0.3301, 0.8002, 0.4876], [0.3424, 0.8009, 0.4774],
    [0.3548, 0.8016, 0.4669], [0.3671, 0.8021, 0.4563], [0.3795, 0.8026, 0.4454],
    [0.3921, 0.8029, 0.4344], [0.405, 0.8031, 0.4233], [0.4184, 0.803, 0.4122],
    [0.4322, 0.8028, 0.4013], [0.4463, 0.8024, 0.3904], [0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691], [0.4899, 0.8002, 0.3586], [0.5044, 0.7993, 0.348],
    [0.5187, 0.7982, 0.3374], [0.5329, 0.797, 0.3267], [0.547, 0.7957, 0.3159],
    [0.5609, 0.7943, 0.305], [0.5748, 0.7929, 0.2941], [0.5886, 0.7913, 0.2833],
    [0.6024, 0.7896, 0.2726], [0.6161, 0.7878, 0.2622], [0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423], [0.6567, 0.7818, 0.2329], [0.6701, 0.7796, 0.2239],
    [0.6833, 0.7773, 0.2155], [0.6963, 0.775, 0.2075], [0.7091, 0.7727, 0.1998],
    [0.7218, 0.7703, 0.1924], [0.7344, 0.7679, 0.1852], [0.7468, 0.7654, 0.1782],
    [0.759, 0.7629, 0.1717], [0.771, 0.7604, 0.1658], [0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157], [0.806, 0.7529, 0.1546], [0.8172, 0.7505, 0.1535],
    [0.8281, 0.7481, 0.1536], [0.8389, 0.7457, 0.1546], [0.8495, 0.7435, 0.1564],
    [0.86, 0.7413, 0.1587], [0.8703, 0.7392, 0.1615], [0.8804, 0.7372, 0.165],
    [0.8903, 0.7353, 0.1695], [0.9, 0.7336, 0.1749], [0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189], [0.9272, 0.7298, 0.1973], [0.9357, 0.729, 0.2061],
    [0.944, 0.7285, 0.2151], [0.9523, 0.7284, 0.2237], [0.9606, 0.7285, 0.2312],
    [0.9689, 0.7292, 0.2373], [0.977, 0.7304, 0.2418], [0.9842, 0.733, 0.2446],
    [0.99, 0.7365, 0.2429], [0.9946, 0.7407, 0.2394], [0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309], [0.9972, 0.7569, 0.2267], [0.9971, 0.7626, 0.2224],
    [0.9969, 0.7683, 0.2181], [0.9966, 0.774, 0.2138], [0.9962, 0.7798, 0.2095],
    [0.9957, 0.7856, 0.2053], [0.9949, 0.7915, 0.2012], [0.9938, 0.7974, 0.1974],
    [0.9923, 0.8034, 0.1939], [0.9906, 0.8095, 0.1906], [0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846], [0.9835, 0.828, 0.1817], [0.9807, 0.8342, 0.1787],
    [0.9778, 0.8404, 0.1757], [0.9748, 0.8467, 0.1726], [0.972, 0.8529, 0.1695],
    [0.9694, 0.8591, 0.1665], [0.9671, 0.8654, 0.1636], [0.9651, 0.8716, 0.1608],
    [0.9634, 0.8778, 0.1582], [0.9619, 0.884, 0.1557], [0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507], [0.9596, 0.9023, 0.148], [0.9595, 0.9084, 0.145],
    [0.9597, 0.9143, 0.1418], [0.9601, 0.9203, 0.1382], [0.9608, 0.9262, 0.1344],
    [0.9618, 0.932, 0.1304], [0.9629, 0.9379, 0.1261], [0.9642, 0.9437, 0.1216],
    [0.9657, 0.9494, 0.1168], [0.9674, 0.9552, 0.1116], [0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001], [0.973, 0.9724, 0.0938], [0.9749, 0.9782, 0.0872],
    [0.9769, 0.9839, 0.0805]
]

# Create Parula colormap
PARULA_COLORMAP = ListedColormap(PARULA_DATA)


def extract_event_epochs(eeg_data: np.ndarray, event_times: np.ndarray, 
                        sfreq: float = 200.0, epoch_duration: float = 4.0,
                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract individual event epochs from continuous EEG data.
    
    Parameters:
    -----------
    eeg_data : np.ndarray
        Continuous EEG data (n_channels, n_samples)
    event_times : np.ndarray
        Event times in seconds
    sfreq : float
        Sampling frequency in Hz
    epoch_duration : float
        Total epoch duration in seconds (e.g., 4.0 for -2 to +2 seconds)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    epochs : np.ndarray
        Epoched data (n_events, n_channels, n_samples_per_epoch)
    epoch_times : np.ndarray
        Time vector for each epoch (relative to event onset)
    """
    half_duration = epoch_duration / 2.0
    samples_per_epoch = int(epoch_duration * sfreq)
    half_samples = int(half_duration * sfreq)
    
    if verbose:
        print(f"📊 Extracting event epochs:")
        print(f"   Epoch duration: ±{half_duration} seconds ({samples_per_epoch} samples)")
        print(f"   Total events: {len(event_times)}")
    
    # Create time vector for epochs (relative to event onset)
    epoch_times = np.linspace(-half_duration, half_duration, samples_per_epoch)
    
    # Convert event times to samples
    event_samples = (event_times * sfreq).astype(int)
    
    # Initialize epoch container
    n_channels = eeg_data.shape[0]
    epochs = []
    valid_events = []
    
    # Extract epochs
    for i, event_sample in enumerate(event_samples):
        start_sample = event_sample - half_samples
        end_sample = event_sample + half_samples
        
        # Check if epoch is within bounds
        if start_sample >= 0 and end_sample < eeg_data.shape[1]:
            epoch = eeg_data[:, start_sample:end_sample]
            epochs.append(epoch)
            valid_events.append(i)
        else:
            if verbose:
                print(f"   Skipping event {i} (out of bounds)")
    
    if len(epochs) == 0:
        raise ValueError("No valid epochs extracted")
    
    epochs = np.array(epochs)  # Shape: (n_valid_events, n_channels, n_samples_per_epoch)
    
    if verbose:
        print(f"   Valid epochs extracted: {len(epochs)}")
        print(f"   Epoch shape: {epochs.shape}")
    
    return epochs, epoch_times


def compute_epoch_spectrograms(epochs: np.ndarray, roi_channels: List[int],
                             sfreq: float = 200.0, freq_range: Tuple[float, float] = (3, 8),
                             n_freqs: int = 60, n_cycles: float = 5.0,
                             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrograms for individual epochs using paper's methodology.
    
    Parameters:
    -----------
    epochs : np.ndarray
        Epoched data (n_events, n_channels, n_samples_per_epoch)
    roi_channels : List[int]
        List of channel indices to include in ROI
    sfreq : float
        Sampling frequency in Hz
    freq_range : Tuple[float, float]
        Frequency range (low, high) in Hz
    n_freqs : int
        Number of logarithmically spaced frequencies
    n_cycles : float
        Number of cycles for Morlet wavelets
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    freqs : np.ndarray
        Frequency vector
    epoch_spectrograms : np.ndarray
        Spectrograms for each epoch (n_events, n_freqs, n_times)
    epoch_times : np.ndarray
        Time vector for spectrograms (relative to event onset)
    """
    n_events, n_channels, n_samples_per_epoch = epochs.shape
    
    if verbose:
        print(f"📊 Computing epoch spectrograms:")
        print(f"   {n_events} events, {len(roi_channels)} ROI channels")
        print(f"   Frequency range: {freq_range[0]}-{freq_range[1]} Hz ({n_freqs} freqs)")
        print(f"   n_cycles: {n_cycles}")
    
    # Create frequency vector
    freqs = np.geomspace(freq_range[0], freq_range[1], n_freqs)
    n_cycles_array = np.full(len(freqs), n_cycles)
    
    # Initialize containers
    epoch_spectrograms = []
    
    # Process each epoch individually
    for epoch_idx in range(n_events):
        if verbose and epoch_idx % 50 == 0:
            print(f"   Processing epoch {epoch_idx+1}/{n_events}")
        
        # Extract ROI data for this epoch
        epoch_data = epochs[epoch_idx]  # Shape: (n_channels, n_samples_per_epoch)
        roi_data = epoch_data[roi_channels, :]  # Shape: (n_roi_channels, n_samples_per_epoch)
        
        # Compute spectrograms for each ROI channel
        channel_spectrograms = []
        
        for ch_idx in range(len(roi_channels)):
            channel_data = roi_data[ch_idx, :]
            
            # Compute spectrogram for this channel
            data_for_mne = channel_data[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_samples)
            power = mne.time_frequency.tfr_array_morlet(
                data_for_mne, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles_array,
                output='power', zero_mean=True
            )
            channel_power = power[0, 0, :, :]  # Shape: (n_freqs, n_times)
            
            # Z-score normalize this channel (using entire epoch statistics)
            ch_mean = np.mean(channel_power, axis=1)[:, np.newaxis]
            ch_std = np.std(channel_power, axis=1)[:, np.newaxis]
            ch_std = np.maximum(ch_std, 1e-12)  # Avoid division by zero
            
            normalized_power = (channel_power - ch_mean) / ch_std
            channel_spectrograms.append(normalized_power)
        
        # Average across ROI channels for this epoch
        epoch_spectrogram = np.mean(channel_spectrograms, axis=0)  # Shape: (n_freqs, n_times)
        epoch_spectrograms.append(epoch_spectrogram)
    
    epoch_spectrograms = np.array(epoch_spectrograms)  # Shape: (n_events, n_freqs, n_times)
    
    # Create time vector for spectrograms (same as epoch times)
    epoch_duration = n_samples_per_epoch / sfreq
    epoch_times = np.linspace(-epoch_duration/2, epoch_duration/2, n_samples_per_epoch)
    
    if verbose:
        print(f"   Epoch spectrograms shape: {epoch_spectrograms.shape}")
        print(f"   Frequency range: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz")
    
    return freqs, epoch_spectrograms, epoch_times


def clip_to_analysis_window(spectrograms: np.ndarray, times: np.ndarray,
                           analysis_window: Tuple[float, float] = (-1.0, 1.0),
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip spectrograms to analysis window (e.g., -1 to +1 seconds).
    
    Parameters:
    -----------
    spectrograms : np.ndarray
        Spectrograms (n_events, n_freqs, n_times)
    times : np.ndarray
        Time vector
    analysis_window : Tuple[float, float]
        Analysis time window (start, end) in seconds
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    clipped_spectrograms : np.ndarray
        Clipped spectrograms
    clipped_times : np.ndarray
        Clipped time vector
    """
    # Find indices for analysis window
    time_mask = (times >= analysis_window[0]) & (times <= analysis_window[1])
    time_indices = np.where(time_mask)[0]
    
    if len(time_indices) == 0:
        raise ValueError(f"No time points found in analysis window {analysis_window}")
    
    # Clip spectrograms
    clipped_spectrograms = spectrograms[:, :, time_indices]
    clipped_times = times[time_indices]
    
    if verbose:
        print(f"📊 Clipped to analysis window {analysis_window[0]} to {analysis_window[1]}s:")
        print(f"   Time points: {len(time_indices)}")
        print(f"   Clipped spectrograms shape: {clipped_spectrograms.shape}")
    
    return clipped_spectrograms, clipped_times


def process_single_rat_epoched(rat_id: str, roi_channels: Union[str, List[int]],
                               pkl_path: str, freq_range: Tuple[float, float] = (3, 8),
                               n_freqs: int = 60, n_cycles: float = 5.0,
                               epoch_duration: float = 4.0,
                               analysis_window: Tuple[float, float] = (-1.0, 1.0),
                               save_path: str = None, verbose: bool = True) -> Dict:
    """
    Process a single rat using the epoched approach.
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to process
    roi_channels : Union[str, List[int]]
        ROI channels specification
    pkl_path : str
        Path to main data file
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    n_cycles : float
        Number of cycles for Morlet wavelets
    epoch_duration : float
        Total epoch duration in seconds (e.g., 4.0 for -2 to +2 seconds)
    analysis_window : Tuple[float, float]
        Analysis time window (start, end) in seconds
    save_path : str
        Path to save results
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    results : Dict
        Results for the rat
    """
    if verbose:
        print(f"\n🐀 Processing rat {rat_id} - Epoched Approach")
        print(f"    Method: Paper's methodology (epoch-wise spectrograms)")
        print(f"    Epoch duration: ±{epoch_duration/2} seconds")
        print(f"    Analysis window: {analysis_window[0]} to {analysis_window[1]} seconds")
        print("=" * 60)
    
    try:
        # Load data
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # Filter sessions for this rat
        rat_sessions = [session for session in all_data if str(session.get('rat_id')) == str(rat_id)]
        
        if not rat_sessions:
            if verbose:
                print(f"❌ No sessions found for rat {rat_id}")
            return None
        
        if verbose:
            print(f"📊 Found {len(rat_sessions)} sessions for rat {rat_id}")
        
        # Parse ROI channels
        if isinstance(roi_channels, str):
            if roi_channels.isdigit():
                roi_channels_list = [int(roi_channels) - 1]  # Convert to 0-based indexing
            else:
                roi_channels_list = [int(ch.strip()) - 1 for ch in roi_channels.split(',')]
        else:
            roi_channels_list = [ch - 1 for ch in roi_channels]  # Convert to 0-based indexing
        
        if verbose:
            print(f"📊 Using ROI channels: {[ch + 1 for ch in roi_channels_list]} (1-based)")
        
        # Process each session
        all_nm_spectrograms = defaultdict(list)
        session_count = 0
        
        for session in rat_sessions:
            session_count += 1
            if verbose:
                print(f"\n📊 Processing session {session_count}/{len(rat_sessions)}")
            
            try:
                # Extract session data
                eeg_data = session['eeg']
                nm_peak_times = np.array(session['nm_peak_times'])
                nm_sizes = np.array(session['nm_sizes'])
                
                if verbose:
                    print(f"   EEG shape: {eeg_data.shape}")
                    print(f"   NM events: {len(nm_peak_times)}")
                    print(f"   NM sizes: {np.unique(nm_sizes)}")
                
                # Group events by NM size
                for nm_size in np.unique(nm_sizes):
                    size_mask = nm_sizes == nm_size
                    size_event_times = nm_peak_times[size_mask]
                    
                    if len(size_event_times) == 0:
                        continue
                    
                    if verbose:
                        print(f"   NM size {nm_size}: {len(size_event_times)} events")
                    
                    # Extract epochs for this NM size
                    try:
                        epochs, epoch_times = extract_event_epochs(
                            eeg_data, size_event_times, sfreq=200.0, 
                            epoch_duration=epoch_duration, verbose=False
                        )
                        
                        # Compute spectrograms for each epoch
                        freqs, epoch_spectrograms, spec_times = compute_epoch_spectrograms(
                            epochs, roi_channels_list, sfreq=200.0, 
                            freq_range=freq_range, n_freqs=n_freqs, 
                            n_cycles=n_cycles, verbose=False
                        )
                        
                        # Clip to analysis window
                        clipped_spectrograms, clipped_times = clip_to_analysis_window(
                            epoch_spectrograms, spec_times, analysis_window=analysis_window, 
                            verbose=False
                        )
                        
                        # Store spectrograms for this NM size
                        all_nm_spectrograms[nm_size].append(clipped_spectrograms)
                        
                        if verbose:
                            print(f"     ✓ Processed {len(epochs)} epochs for NM size {nm_size}")
                        
                    except Exception as e:
                        if verbose:
                            print(f"     ⚠️ Error processing NM size {nm_size}: {e}")
                        continue
                
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error processing session: {e}")
                continue
        
        # Aggregate results across sessions
        final_results = {}
        
        for nm_size, session_spectrograms in all_nm_spectrograms.items():
            if len(session_spectrograms) == 0:
                continue
            
            # Concatenate spectrograms from all sessions
            all_spectrograms = np.concatenate(session_spectrograms, axis=0)
            
            # Average across events (paper's final step)
            avg_spectrogram = np.mean(all_spectrograms, axis=0)
            
            # Compute single value by averaging across time and frequency
            time_averaged = np.mean(avg_spectrogram, axis=1)  # Average across time
            final_value = np.mean(time_averaged)  # Average across frequencies
            
            final_results[nm_size] = {
                'avg_spectrogram': avg_spectrogram,
                'final_value': final_value,
                'n_events': all_spectrograms.shape[0],
                'n_sessions': len(session_spectrograms),
                'window_times': clipped_times,
                'spectrograms_shape': all_spectrograms.shape
            }
            
            if verbose:
                print(f"✓ NM size {nm_size}: {all_spectrograms.shape[0]} events, final value = {final_value:.6f}")
        
        # Create comprehensive results
        rat_results = {
            'rat_id': rat_id,
            'method': 'epoched_approach',
            'frequencies': freqs,
            'roi_channels': [ch + 1 for ch in roi_channels_list],  # Convert back to 1-based
            'nm_results': final_results,
            'analysis_parameters': {
                'epoch_duration': epoch_duration,
                'analysis_window': analysis_window,
                'frequency_range': freq_range,
                'n_frequencies': n_freqs,
                'n_cycles': n_cycles
            },
            'n_sessions_processed': session_count
        }
        
        # Save results if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            results_file = os.path.join(save_path, f'rat_{rat_id}_epoched_results.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump(rat_results, f)
            
            if verbose:
                print(f"✓ Results saved to: {results_file}")
        
        return rat_results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing rat {rat_id}: {e}")
            import traceback
            traceback.print_exc()
        return None


def discover_rat_ids(pkl_path: str, verbose: bool = True) -> List[str]:
    """
    Discover all unique rat IDs from the dataset.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the main EEG data file
    verbose : bool
        Whether to print discovery progress
        
    Returns:
    --------
    rat_ids : List[str]
        List of unique rat IDs found in the dataset
    """
    if verbose:
        print(f"🔍 Discovering rat IDs from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    rat_ids = set()
    for session_data in all_data:
        rat_id = session_data.get('rat_id')
        if rat_id is not None:
            rat_ids.add(str(rat_id))
    
    rat_ids_list = sorted(list(rat_ids))
    if verbose:
        print(f"✓ Found {len(rat_ids_list)} unique rats: {rat_ids_list}")
    
    # Clean up memory
    del all_data
    gc.collect()
    
    return rat_ids_list


def run_cross_rats_epoched_analysis(roi: str, pkl_path: str = None,
                                   freq_range: Tuple[float, float] = (3, 8),
                                   n_freqs: int = 60, n_cycles: float = 5.0,
                                   epoch_duration: float = 4.0,
                                   analysis_window: Tuple[float, float] = (-1.0, 1.0),
                                   rat_ids: Optional[List[str]] = None,
                                   save_path: str = None, verbose: bool = True) -> Dict:
    """
    Run cross-rats analysis using the epoched approach.
    
    Parameters:
    -----------
    roi : str
        ROI channels specification
    pkl_path : str
        Path to main EEG data file
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    n_cycles : float
        Number of cycles for Morlet wavelets
    epoch_duration : float
        Total epoch duration in seconds (e.g., 4.0 for -2 to +2 seconds)
    analysis_window : Tuple[float, float]
        Analysis time window (start, end) in seconds
    rat_ids : Optional[List[str]]
        Specific rat IDs to process
    save_path : str
        Path to save results
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    results : Dict
        Cross-rats aggregated results
    """
    if pkl_path is None:
        pkl_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
    if save_path is None:
        save_path = "results/cross_rats_epoched"
    
    if verbose:
        print("🧠 Cross-Rats NM Theta Analysis - Epoched Approach")
        print("=" * 80)
        print(f"Method: Paper's methodology (epoch-wise spectrograms)")
        print(f"Data file: {pkl_path}")
        print(f"ROI: {roi}")
        print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz ({n_freqs} freqs)")
        print(f"Epoch duration: ±{epoch_duration/2} seconds")
        print(f"Analysis window: {analysis_window[0]} to {analysis_window[1]} seconds")
        print(f"n_cycles: {n_cycles}")
        print(f"Save path: {save_path}")
        print("=" * 80)
    
    # Discover rat IDs
    if rat_ids is None:
        rat_ids = discover_rat_ids(pkl_path, verbose=verbose)
    
    # Process each rat
    rat_results = {}
    
    for rat_id in rat_ids:
        rat_save_path = os.path.join(save_path, f'rat_{rat_id}')
        
        results = process_single_rat_epoched(
            rat_id=rat_id,
            roi_channels=roi,
            pkl_path=pkl_path,
            freq_range=freq_range,
            n_freqs=n_freqs,
            n_cycles=n_cycles,
            epoch_duration=epoch_duration,
            analysis_window=analysis_window,
            save_path=rat_save_path,
            verbose=verbose
        )
        
        rat_results[rat_id] = results
        
        # Force garbage collection after each rat
        gc.collect()
    
    # Aggregate results across rats
    valid_results = {rat_id: results for rat_id, results in rat_results.items() if results is not None}
    
    if len(valid_results) == 0:
        raise ValueError("No valid rat results to aggregate")
    
    # Collect final values for each NM size
    nm_size_results = defaultdict(lambda: {
        'final_values': [],
        'rat_ids': [],
        'total_events': [],
        'total_sessions': []
    })
    
    for rat_id, results in valid_results.items():
        for nm_size, nm_data in results['nm_results'].items():
            nm_size_results[nm_size]['final_values'].append(nm_data['final_value'])
            nm_size_results[nm_size]['rat_ids'].append(rat_id)
            nm_size_results[nm_size]['total_events'].append(nm_data['n_events'])
            nm_size_results[nm_size]['total_sessions'].append(nm_data['n_sessions'])
    
    # Compute cross-rats averages
    final_nm_results = {}
    for nm_size, data in nm_size_results.items():
        final_values = np.array(data['final_values'])
        cross_rats_average = np.mean(final_values)
        
        final_nm_results[nm_size] = {
            'cross_rats_average': cross_rats_average,
            'individual_values': final_values,
            'rat_ids': data['rat_ids'],
            'n_rats': len(final_values),
            'total_events_all_rats': sum(data['total_events']),
            'total_sessions_all_rats': sum(data['total_sessions'])
        }
    
    # Create aggregated results
    aggregated_results = {
        'analysis_type': 'cross_rats_epoched',
        'method': 'epoched_approach',
        'rat_ids': list(valid_results.keys()),
        'n_rats': len(valid_results),
        'nm_size_results': final_nm_results,
        'analysis_parameters': {
            'epoch_duration': epoch_duration,
            'analysis_window': analysis_window,
            'frequency_range': freq_range,
            'n_frequencies': n_freqs,
            'n_cycles': n_cycles
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save aggregated results
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, 'cross_rats_epoched_aggregated.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    # Print summary
    successful_rats = list(valid_results.keys())
    failed_rats = [rat_id for rat_id, results in rat_results.items() if results is None]
    
    print("\n" + "=" * 80)
    print("📊 EPOCHED ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total rats attempted: {len(rat_ids)}")
    print(f"Successfully processed: {len(successful_rats)}")
    print(f"Failed to process: {len(failed_rats)}")
    
    if successful_rats:
        print(f"\n✅ Successful rats: {successful_rats}")
    
    if failed_rats:
        print(f"\n❌ Failed rats: {failed_rats}")
    
    if len(successful_rats) > 0:
        print(f"\n📈 Final Cross-Rats Averages (Epoched Method):")
        for nm_size, data in final_nm_results.items():
            print(f"  NM size {nm_size}: {data['cross_rats_average']:.6f}")
        
        print(f"\n✅ Epoched analysis completed successfully!")
        print(f"Results saved to: {save_path}")
    
    return aggregated_results


def main():
    """
    Main function for epoched cross-rats analysis.
    """
    parser = argparse.ArgumentParser(
        description='Cross-Rats NM Theta Analysis - Epoched Approach (Paper\'s Method)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pkl_path', default=DataConfig.MAIN_EEG_DATA_FILE,
                       help='Path to main EEG data file')
    parser.add_argument('--roi', required=True,
                       help='ROI channels specification')
    parser.add_argument('--freq_min', type=float, default=3.0,
                       help='Minimum frequency (Hz)')
    parser.add_argument('--freq_max', type=float, default=8.0,
                       help='Maximum frequency (Hz)')
    parser.add_argument('--n_freqs', type=int, default=60,
                       help='Number of frequencies')
    parser.add_argument('--n_cycles', type=float, default=5.0,
                       help='Number of cycles for Morlet wavelets')
    parser.add_argument('--epoch_duration', type=float, default=4.0,
                       help='Total epoch duration in seconds (e.g., 4.0 for -2 to +2 seconds)')
    parser.add_argument('--analysis_start', type=float, default=-1.0,
                       help='Analysis window start time (seconds)')
    parser.add_argument('--analysis_end', type=float, default=1.0,
                       help='Analysis window end time (seconds)')
    parser.add_argument('--rat_ids', type=str, default=None,
                       help='Specific rat IDs to process (comma-separated)')
    parser.add_argument('--save_path', default="results/cross_rats_epoched",
                       help='Directory for saving results')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Parse rat_ids if provided
    rat_ids = None
    if args.rat_ids:
        rat_ids = [r.strip() for r in args.rat_ids.split(',')]
    
    # Run the analysis
    return run_cross_rats_epoched_analysis(
        roi=args.roi,
        pkl_path=args.pkl_path,
        freq_range=(args.freq_min, args.freq_max),
        n_freqs=args.n_freqs,
        n_cycles=args.n_cycles,
        epoch_duration=args.epoch_duration,
        analysis_window=(args.analysis_start, args.analysis_end),
        rat_ids=rat_ids,
        save_path=args.save_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) > 1:
        # Command line usage
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    else:
        # IDE usage - run analysis directly
        from config import DataConfig
        
        data_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
        
        start_time = time.time()
        results = run_cross_rats_epoched_analysis(
            roi="8",                        
            pkl_path=data_path,
            freq_range=(3.0, 7.0),           # 3-7Hz like your other analysis
            n_freqs=60,                      # Same as your other analysis
            n_cycles=5.0,                    # Same as your other analysis
            epoch_duration=4.0,              # -2 to +2 seconds (paper's method)
            analysis_window=(-1.0, 1.0),     # -1 to +1 seconds (paper's method)
            rat_ids=None,                
            save_path="results/cross_rats_epoched_test",
            verbose=True
        )
        end_time = time.time()
        print(f"Total elapsed time: {end_time - start_time:.2f} seconds")