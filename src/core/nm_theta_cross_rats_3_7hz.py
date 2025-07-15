#!/usr/bin/env python3
"""
Cross-Rats NM Theta Analysis (3-7Hz, -0.2 to 0 second windows)

This script performs NM theta analysis across multiple rats for 3-7Hz range,
extracting windows from -0.2 to 0 seconds and averaging across time, frequency,
windows, sessions, and rats.

Author: Generated for cross-rats EEG near-mistake analysis
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import AnalysisConfig, DataConfig, PlottingConfig

# Import the run_analysis function from nm_theta_analyzer
from nm_theta_analyzer import run_analysis

# Parula colormap data
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


def load_frequencies_from_file(freq_file_path: str) -> np.ndarray:
    """
    Load frequencies from a text file.
    
    Parameters:
    -----------
    freq_file_path : str
        Path to the frequencies file (one frequency per line)
    
    Returns:
    --------
    frequencies : np.ndarray
        Array of frequency values
    """
    if not os.path.exists(freq_file_path):
        raise FileNotFoundError(f"Frequencies file not found: {freq_file_path}")
    
    frequencies = []
    with open(freq_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    freq = float(line)
                    frequencies.append(freq)
                except ValueError:
                    print(f"Warning: Skipping invalid frequency value: {line}")
    
    if not frequencies:
        raise ValueError(f"No valid frequencies found in {freq_file_path}")
    
    return np.array(frequencies)


def get_3_7hz_frequencies(freq_file_path: str) -> np.ndarray:
    """
    Get frequencies in the 3-7Hz range from the frequencies file.
    
    Parameters:
    -----------
    freq_file_path : str
        Path to the frequencies file
    
    Returns:
    --------
    frequencies : np.ndarray
        Array of frequency values in 3-7Hz range
    """
    all_frequencies = load_frequencies_from_file(freq_file_path)
    
    # Filter frequencies to be within 3-7 Hz range
    mask = (all_frequencies >= 3.0) & (all_frequencies <= 7.0)
    frequencies = all_frequencies[mask]
    
    if len(frequencies) == 0:
        raise ValueError(f"No frequencies in range 3-7 Hz found in {freq_file_path}")
    
    # Sort frequencies (typically descending in the file)
    frequencies = np.sort(frequencies)
    
    print(f"Loaded {len(frequencies)} frequencies from {freq_file_path}")
    print(f"Frequency range: {frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz")
    
    return frequencies


def extract_short_windows(spectrograms: np.ndarray, window_times: np.ndarray,
                         window_start: float = -0.2, window_end: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract short windows from spectrograms (e.g., -0.2 to 0 seconds).
    
    Parameters:
    -----------
    spectrograms : np.ndarray
        Input spectrograms of shape (n_events, n_freqs, n_times)
    window_times : np.ndarray
        Time points corresponding to spectrogram time axis
    window_start : float
        Start time of the window (default: -0.2)
    window_end : float
        End time of the window (default: 0.0)
    
    Returns:
    --------
    short_spectrograms : np.ndarray
        Extracted spectrograms for the short window
    short_window_times : np.ndarray
        Time points for the short window
    """
    # Find time indices for the desired window
    time_mask = (window_times >= window_start) & (window_times <= window_end)
    time_indices = np.where(time_mask)[0]
    
    if len(time_indices) == 0:
        raise ValueError(f"No time points found in window [{window_start}, {window_end}]")
    
    # Extract spectrograms for the time window
    short_spectrograms = spectrograms[:, :, time_indices]
    short_window_times = window_times[time_indices]
    
    print(f"Extracted {len(time_indices)} time points for window [{window_start}, {window_end}]s")
    
    return short_spectrograms, short_window_times


def average_spectrogram_hierarchically(spectrograms: np.ndarray, frequencies: np.ndarray,
                                     window_times: np.ndarray) -> float:
    """
    Average spectrogram hierarchically: time -> frequency -> events.
    
    Parameters:
    -----------
    spectrograms : np.ndarray
        Input spectrograms of shape (n_events, n_freqs, n_times)
    frequencies : np.ndarray
        Frequency values
    window_times : np.ndarray
        Time points
    
    Returns:
    --------
    final_value : float
        Single averaged value
    """
    # Step 1: Average across time for each event and frequency
    # Shape: (n_events, n_freqs, n_times) -> (n_events, n_freqs)
    time_averaged = np.mean(spectrograms, axis=2)
    
    # Step 2: Average across frequencies for each event
    # Shape: (n_events, n_freqs) -> (n_events,)
    freq_averaged = np.mean(time_averaged, axis=1)
    
    # Step 3: Average across events
    # Shape: (n_events,) -> scalar
    final_value = np.mean(freq_averaged)
    
    return final_value


def process_single_rat_3_7hz(
    rat_id: str,
    roi_or_channels: Union[str, List[int]],
    pkl_path: str,
    frequencies: np.ndarray,
    window_duration: float = 2.0,
    n_cycles_factor: float = 3.0,
    base_save_path: str = 'results/cross_rats_3_7hz',
    show_plots: bool = False,
    verbose: bool = True
) -> Tuple[str, Optional[Dict]]:
    """
    Process a single rat for 3-7Hz analysis with -0.2 to 0 second windows.
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to process
    roi_or_channels : Union[str, List[int]]
        ROI specification
    pkl_path : str
        Path to main data file
    frequencies : np.ndarray
        Frequency values to use (3-7Hz range)
    window_duration : float
        Original event window duration (for spectrograms)
    n_cycles_factor : float
        Cycles factor for spectrograms
    base_save_path : str
        Base directory for saving results
    show_plots : bool
        Whether to show plots during processing
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    rat_id : str
        The processed rat ID
    results : Optional[Dict]
        Results for the rat, or None if failed
    """
    if verbose:
        print(f"\n🐀 Processing rat {rat_id} - 3-7Hz Analysis")
        print(f"    Frequencies: {len(frequencies)} freqs ({frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz)")
        print(f"    Window extraction: -0.2 to 0 seconds")
        print("=" * 60)
    
    try:
        # Create save path for this rat
        rat_save_path = os.path.join(base_save_path, f'rat_{rat_id}_3_7hz')
        
        # First, run the full analysis to get spectrograms
        results = run_analysis(
            mode='multi',
            method='basic',
            parallel_type=None,
            pkl_path=pkl_path,
            roi=roi_or_channels if isinstance(roi_or_channels, str) else ','.join(map(str, roi_or_channels)),
            session_index=None,
            rat_id=rat_id,
            freq_min=float(frequencies[0]),
            freq_max=float(frequencies[-1]),
            n_freqs=len(frequencies),
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            n_jobs=None,
            batch_size=8,
            save_path=rat_save_path,
            show_plots=show_plots,
            show_frequency_profiles=False
        )
        
        if results is None:
            if verbose:
                print(f"❌ Failed to get results for rat {rat_id}")
            return rat_id, None
        
        # Extract short windows and compute hierarchical averages
        processed_results = {}
        
        for nm_size, window_data in results['averaged_windows'].items():
            if verbose:
                print(f"  Processing NM size {nm_size}")
            
            # Get the full spectrograms
            full_spectrograms = window_data['avg_spectrogram']  # Shape: (n_freqs, n_times)
            full_window_times = window_data['window_times']
            
            # We need to reconstruct individual spectrograms for proper averaging
            # For now, we'll work with the averaged spectrogram as a proxy
            # In a real implementation, you'd need access to individual event spectrograms
            
            # Extract short window (-0.2 to 0 seconds)
            time_mask = (full_window_times >= -0.2) & (full_window_times <= 0.0)
            time_indices = np.where(time_mask)[0]
            
            if len(time_indices) == 0:
                if verbose:
                    print(f"    ⚠️  No time points in -0.2 to 0s window for NM size {nm_size}")
                continue
            
            # Extract short window spectrogram
            short_spectrogram = full_spectrograms[:, time_indices]
            short_window_times = full_window_times[time_indices]
            
            # Average hierarchically: time -> frequency
            # Step 1: Average across time
            time_averaged = np.mean(short_spectrogram, axis=1)  # Shape: (n_freqs,)
            
            # Step 2: Average across frequencies
            freq_averaged = np.mean(time_averaged)  # Shape: scalar
            
            # Store the single averaged value
            processed_results[nm_size] = {
                'averaged_value': freq_averaged,
                'n_events': window_data['n_events'],
                'n_sessions': window_data['n_sessions'],
                'n_time_points': len(time_indices),
                'n_frequencies': len(frequencies),
                'short_window_times': short_window_times,
                'frequencies_used': frequencies
            }
            
            if verbose:
                print(f"    ✓ NM size {nm_size}: averaged value = {freq_averaged:.4f}")
        
        # Create final results structure
        final_results = {
            'rat_id': rat_id,
            'analysis_type': '3_7hz_short_windows',
            'window_range': [-0.2, 0.0],
            'frequencies': frequencies,
            'roi_channels': results['roi_channels'],
            'processed_windows': processed_results,
            'n_sessions_analyzed': results.get('n_sessions_analyzed', 0),
            'analysis_parameters': {
                'frequency_range': [float(frequencies[0]), float(frequencies[-1])],
                'n_frequencies': len(frequencies),
                'window_duration': window_duration,
                'short_window_range': [-0.2, 0.0],
                'averaging_hierarchy': 'time -> frequency'
            }
        }
        
        # Save results
        os.makedirs(rat_save_path, exist_ok=True)
        results_file = os.path.join(rat_save_path, 'rat_3_7hz_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(final_results, f)
        
        if verbose:
            print(f"✓ Successfully processed rat {rat_id}")
            print(f"  Results saved to: {results_file}")
        
        return rat_id, final_results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing rat {rat_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        return rat_id, None


def aggregate_cross_rats_3_7hz_results(
    rat_results: Dict[str, Dict],
    frequencies: np.ndarray,
    save_path: str,
    verbose: bool = True
) -> Dict:
    """
    Aggregate 3-7Hz results across all rats.
    
    Parameters:
    -----------
    rat_results : Dict[str, Dict]
        Dictionary mapping rat_id -> results
    frequencies : np.ndarray
        Frequency values used
    save_path : str
        Path to save aggregated results
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    if verbose:
        print(f"\n📊 Aggregating 3-7Hz results across {len(rat_results)} rats")
        print("=" * 60)
    
    # Filter out failed results
    valid_results = {rat_id: results for rat_id, results in rat_results.items() if results is not None}
    n_valid_rats = len(valid_results)
    
    if n_valid_rats == 0:
        raise ValueError("No valid rat results to aggregate")
    
    if verbose:
        print(f"Valid results from {n_valid_rats} rats: {list(valid_results.keys())}")
    
    # Initialize aggregation containers
    aggregated_nm_sizes = defaultdict(lambda: {
        'averaged_values': [],
        'total_events': [],
        'n_sessions': [],
        'rat_ids': []
    })
    
    # Collect averaged values from all rats
    for rat_id, results in valid_results.items():
        if verbose:
            print(f"Processing results from rat {rat_id}")
        
        for nm_size, window_data in results['processed_windows'].items():
            aggregated_nm_sizes[nm_size]['averaged_values'].append(window_data['averaged_value'])
            aggregated_nm_sizes[nm_size]['total_events'].append(window_data['n_events'])
            aggregated_nm_sizes[nm_size]['n_sessions'].append(window_data['n_sessions'])
            aggregated_nm_sizes[nm_size]['rat_ids'].append(rat_id)
    
    # Compute cross-rats averages
    final_aggregated_results = {}
    
    for nm_size, data in aggregated_nm_sizes.items():
        averaged_values = np.array(data['averaged_values'])
        
        if verbose:
            print(f"NM size {nm_size}: {len(averaged_values)} rats")
            print(f"  Individual values: {averaged_values}")
        
        # Average across rats (final level of averaging)
        cross_rats_average = np.mean(averaged_values)
        
        final_aggregated_results[nm_size] = {
            'cross_rats_average': cross_rats_average,
            'individual_rat_values': averaged_values,
            'rat_ids': data['rat_ids'],
            'total_events_per_rat': data['total_events'],
            'n_sessions_per_rat': data['n_sessions'],
            'n_rats': len(averaged_values),
            'total_events_all_rats': sum(data['total_events']),
            'total_sessions_all_rats': sum(data['n_sessions'])
        }
        
        if verbose:
            print(f"  ✓ Cross-rats average: {cross_rats_average:.4f}")
    
    # Create comprehensive results dictionary
    aggregated_results = {
        'analysis_type': 'cross_rats_3_7hz',
        'rat_ids': list(valid_results.keys()),
        'n_rats': n_valid_rats,
        'frequencies': frequencies,
        'window_range': [-0.2, 0.0],
        'nm_size_results': final_aggregated_results,
        'analysis_parameters': {
            'frequency_range': [float(frequencies[0]), float(frequencies[-1])],
            'n_frequencies': len(frequencies),
            'window_range': [-0.2, 0.0],
            'averaging_hierarchy': 'time -> frequency -> windows -> sessions -> rats'
        },
        'processing_info': {
            'n_rats_attempted': len(rat_results),
            'n_rats_successful': n_valid_rats,
            'failed_rats': [rat_id for rat_id, results in rat_results.items() if results is None],
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save aggregated results
    os.makedirs(save_path, exist_ok=True)
    
    results_file = os.path.join(save_path, 'cross_rats_3_7hz_aggregated_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    if verbose:
        print(f"✓ Cross-rats 3-7Hz results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = os.path.join(save_path, 'cross_rats_3_7hz_summary.json')
    summary_data = {
        'n_rats': n_valid_rats,
        'rat_ids': list(valid_results.keys()),
        'nm_sizes_analyzed': [float(key) for key in final_aggregated_results.keys()],
        'cross_rats_averages': {nm_size: float(data['cross_rats_average']) 
                               for nm_size, data in final_aggregated_results.items()},
        'total_events_all_rats': {nm_size: data['total_events_all_rats']
                                 for nm_size, data in final_aggregated_results.items()},
        'total_sessions_all_rats': {nm_size: data['total_sessions_all_rats']
                                   for nm_size, data in final_aggregated_results.items()},
        'frequency_range': [float(frequencies[0]), float(frequencies[-1])],
        'n_frequencies': len(frequencies),
        'window_range': [-0.2, 0.0],
        'averaging_hierarchy': 'time -> frequency -> windows -> sessions -> rats'
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    if verbose:
        print(f"✓ Summary statistics saved to: {summary_file}")
    
    return aggregated_results


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


def run_cross_rats_3_7hz_analysis(
    roi: str,
    pkl_path: str = None,
    freq_file_path: str = None,
    window_duration: float = 2.0,
    n_cycles_factor: float = 3.0,
    rat_ids: Optional[List[str]] = None,
    save_path: str = None,
    show_plots: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run cross-rats 3-7Hz analysis with -0.2 to 0 second windows.
    
    Parameters:
    -----------
    roi : str
        ROI name or channels
    pkl_path : str, optional
        Path to main EEG data file
    freq_file_path : str, optional
        Path to frequencies file
    window_duration : float
        Original event window duration (for spectrograms)
    n_cycles_factor : float
        Cycles factor for spectrograms
    rat_ids : Optional[List[str]]
        Specific rat IDs to process
    save_path : str, optional
        Directory for saving results
    show_plots : bool
        Show plots during processing
    verbose : bool
        Print detailed progress information
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    # Apply defaults
    if pkl_path is None:
        pkl_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
    if freq_file_path is None:
        freq_file_path = "data/config/frequencies.txt"
    if save_path is None:
        save_path = "results/cross_rats_3_7hz"
    
    # Load 3-7Hz frequencies
    frequencies = get_3_7hz_frequencies(freq_file_path)
    
    if verbose:
        print("🧠 Cross-Rats 3-7Hz Analysis (Short Windows)")
        print("=" * 80)
        print(f"Data file: {pkl_path}")
        print(f"ROI: {roi}")
        print(f"Frequencies: {len(frequencies)} freqs ({frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz)")
        print(f"Window extraction: -0.2 to 0 seconds")
        print(f"Averaging hierarchy: time -> frequency -> windows -> sessions -> rats")
        print(f"Save path: {save_path}")
        print("=" * 80)
    
    # Discover rat IDs
    if rat_ids is None:
        rat_ids = discover_rat_ids(pkl_path, verbose=verbose)
    
    # Process each rat individually
    rat_results = {}
    
    for rat_id in rat_ids:
        rat_id_str, results = process_single_rat_3_7hz(
            rat_id=rat_id,
            roi_or_channels=roi,
            pkl_path=pkl_path,
            frequencies=frequencies,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            base_save_path=save_path,
            show_plots=show_plots,
            verbose=verbose
        )
        
        rat_results[rat_id_str] = results
        
        # Force garbage collection after each rat
        gc.collect()
    
    # Aggregate results across rats
    aggregated_results = aggregate_cross_rats_3_7hz_results(
        rat_results=rat_results,
        frequencies=frequencies,
        save_path=save_path,
        verbose=verbose
    )
    
    # Print final summary
    successful_rats = [rat_id for rat_id, results in rat_results.items() if results is not None]
    failed_rats = [rat_id for rat_id, results in rat_results.items() if results is None]
    
    print("\n" + "=" * 80)
    print("📊 3-7Hz ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total rats attempted: {len(rat_ids)}")
    print(f"Successfully processed: {len(successful_rats)}")
    print(f"Failed to process: {len(failed_rats)}")
    
    if successful_rats:
        print(f"\n✅ Successful rats: {successful_rats}")
    
    if failed_rats:
        print(f"\n❌ Failed rats: {failed_rats}")
    
    if len(successful_rats) > 0:
        print("\n✅ 3-7Hz analysis completed successfully!")
        print(f"Results saved to: {save_path}")
        
        # Show final averaged values
        print(f"\n📈 Final Cross-Rats Averages:")
        for nm_size, data in aggregated_results['nm_size_results'].items():
            print(f"  NM size {nm_size}: {data['cross_rats_average']:.4f}")
    else:
        print("\n❌ Analysis failed - no rats were successfully processed!")
    
    return aggregated_results


def main():
    """
    Main function for 3-7Hz cross-rats analysis.
    """
    parser = argparse.ArgumentParser(
        description='Cross-Rats 3-7Hz Analysis (Short Windows)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pkl_path', default=DataConfig.MAIN_EEG_DATA_FILE,
                       help='Path to main EEG data file')
    parser.add_argument('--roi', required=True,
                       help='ROI name or channels')
    parser.add_argument('--freq_file_path', default="data/config/frequencies.txt",
                       help='Path to frequencies file')
    parser.add_argument('--window_duration', type=float, default=2.0,
                       help='Original event window duration (s)')
    parser.add_argument('--n_cycles_factor', type=float, default=3.0,
                       help='Cycles factor for spectrograms')
    parser.add_argument('--rat_ids', type=str, default=None,
                       help='Specific rat IDs to process (comma-separated)')
    parser.add_argument('--save_path', default="results/cross_rats_3_7hz",
                       help='Directory for saving results')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots during processing')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Parse rat_ids if provided
    rat_ids = None
    if args.rat_ids:
        rat_ids = [r.strip() for r in args.rat_ids.split(',')]
    
    # Run the analysis
    return run_cross_rats_3_7hz_analysis(
        roi=args.roi,
        pkl_path=args.pkl_path,
        freq_file_path=args.freq_file_path,
        window_duration=args.window_duration,
        n_cycles_factor=args.n_cycles_factor,
        rat_ids=rat_ids,
        save_path=args.save_path,
        show_plots=args.show_plots,
        verbose=args.verbose
    )


if __name__ == "__main__":
    # Check if we're running in IDE mode or command line mode
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # IDE usage - run analysis directly
        from config import DataConfig
        
        data_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
        
        results = run_cross_rats_3_7hz_analysis(
            roi="1",
            pkl_path=data_path,
            freq_file_path="data/config/frequencies.txt",
            rat_ids=None,
            save_path="results/cross_rats_3_7hz",
            verbose=True
        )