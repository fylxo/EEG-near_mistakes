#!/usr/bin/env python3
"""
Cross-Rats NM Theta Analysis

This script performs NM theta analysis across multiple rats, aggregating results
from individual rat multi-session analyses.

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
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the run_analysis function from nm_theta_analyzer
from nm_theta_analyzer import run_analysis

# Constants for rat 9442 special handling
RAT_9442_32_CHANNEL_SESSIONS = ['070419', '080419', '090419', '190419']
RAT_9442_20_CHANNEL_ELECTRODES = [10, 11, 12, 13, 14, 15, 16, 19, 1, 24, 25, 29, 2, 3, 4, 5, 6, 7, 8, 9]


def load_electrode_mappings(mapping_file: str = 'data/config/consistent_electrode_mappings.csv') -> pd.DataFrame:
    """
    Load electrode mappings from CSV file.
    
    Parameters:
    -----------
    mapping_file : str
        Path to the electrode mappings CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rat_id as index and electrode mappings
    """
    df = pd.read_csv(mapping_file)
    return df.set_index('rat_id')


def get_electrode_numbers_from_roi(roi_or_channels: Union[str, List[int]], 
                                   mapping_df: pd.DataFrame,
                                   rat_id: str) -> Union[List[int], str]:
    """
    Convert ROI specification to electrode numbers using the mapping.
    
    Parameters:
    -----------
    roi_or_channels : Union[str, List[int]]
        ROI specification ("frontal", "hippocampus") or electrode numbers
    mapping_df : pd.DataFrame
        Electrode mappings DataFrame
    rat_id : str
        Rat ID for mapping lookup
        
    Returns:
    --------
    Union[List[int], str]
        List of electrode numbers, or ROI string if ROI mapping not implemented
    """
    if isinstance(roi_or_channels, list):
        return roi_or_channels
    
    if roi_or_channels.replace(',', '').replace(' ', '').isdigit():
        # It's a comma-separated list of channels
        return [int(ch.strip()) for ch in roi_or_channels.split(',')]
    
    # It's a ROI name - would need additional mapping logic
    # For now, return the raw specification
    return roi_or_channels


def check_rat_9442_compatibility(roi_or_channels: Union[str, List[int]], 
                                mapping_df: pd.DataFrame,
                                verbose: bool = True) -> bool:
    """
    Check if the requested ROI/channels are compatible with rat 9442's 20-channel sessions.
    
    Parameters:
    -----------
    roi_or_channels : Union[str, List[int]]
        ROI specification or electrode numbers
    mapping_df : pd.DataFrame
        Electrode mappings DataFrame
    verbose : bool
        Whether to print compatibility info
        
    Returns:
    --------
    bool
        True if compatible, False otherwise
    """
    try:
        # Get electrode numbers for the requested ROI/channels
        requested_electrodes = get_electrode_numbers_from_roi(roi_or_channels, mapping_df, '9442')
        
        # If it's a string ROI that couldn't be converted, assume it's compatible for now
        if isinstance(requested_electrodes, str):
            if verbose:
                print(f"  ROI '{requested_electrodes}' - assuming compatible (ROI mapping not implemented)")
            return True
        
        # Get available electrodes from the CSV mapping for rat 9442
        # Note: rat_id in CSV is stored as int, not string
        rat_id_int = 9442
        if rat_id_int not in mapping_df.index:
            if verbose:
                print(f"  ❌ Rat 9442 not found in electrode mappings")
            return False
        
        rat_9442_mapping = mapping_df.loc[rat_id_int]
        available_electrodes = []
        
        for col in rat_9442_mapping.index:
            if col.startswith('ch_'):
                electrode_value = rat_9442_mapping[col]
                if pd.notna(electrode_value) and electrode_value != 'None':
                    try:
                        available_electrodes.append(int(electrode_value))
                    except (ValueError, TypeError):
                        continue
        
        available_electrodes_set = set(available_electrodes)
        requested_electrodes_set = set(requested_electrodes)
        
        if verbose:
            print(f"  Available electrodes from CSV for rat 9442: {sorted(available_electrodes)}")
        
        missing_electrodes = requested_electrodes_set - available_electrodes_set
        
        if missing_electrodes:
            if verbose:
                print(f"  ❌ Rat 9442 incompatible - electrodes not in CSV mapping: {sorted(missing_electrodes)}")
            return False
        else:
            if verbose:
                print(f"  ✓ Rat 9442 compatible - all electrodes {sorted(requested_electrodes)} found in CSV mapping")
            return True
            
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error checking rat 9442 compatibility: {e}")
        return False


def discover_rat_ids(pkl_path: str, exclude_20_channel_rats: bool = False, verbose: bool = True, 
                     roi_or_channels: Optional[Union[str, List[int]]] = None) -> List[str]:
    """
    Discover all unique rat IDs from the dataset.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the main EEG data file
    exclude_20_channel_rats : bool
        Whether to exclude rats with 20 channels (default: False - removed validation)
    verbose : bool
        Whether to print discovery progress (default: True)
    roi_or_channels : Optional[Union[str, List[int]]]
        ROI specification to check compatibility with rat 9442 (default: None)
        
    Returns:
    --------
    rat_ids : List[str]
        List of unique rat IDs found in the dataset
    """
    if verbose:
        print(f"🔍 Discovering rat IDs from {pkl_path}")
        print("Loading data to scan for rat IDs...")
    
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
    
    # Handle rat 9442 special case
    excluded_rats = []
    if '9442' in rat_ids_list:
        if roi_or_channels is not None:
            # Load electrode mappings and check compatibility
            try:
                mapping_df = load_electrode_mappings()
                if verbose:
                    print(f"\n🔍 Checking rat 9442 compatibility with requested ROI/channels...")
                
                is_compatible = check_rat_9442_compatibility(roi_or_channels, mapping_df, verbose)
                
                if not is_compatible:
                    rat_ids_list.remove('9442')
                    excluded_rats.append('9442')
                    if verbose:
                        print(f"❌ Excluding rat 9442 (incompatible with requested ROI/channels)")
                else:
                    if verbose:
                        print(f"✓ Including rat 9442 (compatible with requested ROI/channels)")
                        
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error checking rat 9442 compatibility: {e}")
                    print(f"❌ Excluding rat 9442 (compatibility check failed)")
                rat_ids_list.remove('9442')
                excluded_rats.append('9442')
        else:
            # If no ROI specified, exclude rat 9442 by default
            rat_ids_list.remove('9442')
            excluded_rats.append('9442')
            if verbose:
                print(f"❌ Excluding rat 9442 (no ROI specified for compatibility check)")
    
    if verbose:
        print(f"\n📊 Final rat selection:")
        print(f"  Total rats found: {len(rat_ids_list) + len(excluded_rats)}")
        print(f"  Rats to process: {len(rat_ids_list)}")
        if excluded_rats:
            print(f"  Excluded rats: {excluded_rats}")
        else:
            print(f"  Excluded rats: None")
    
    # Clean up memory
    del all_data
    gc.collect()
    
    return rat_ids_list


def get_rat_9442_mapping_for_session(session_id: str, mapping_df: pd.DataFrame) -> str:
    """
    Get the appropriate rat mapping for rat 9442 based on session type.
    
    Parameters:
    -----------
    session_id : str
        Session ID (e.g., '070419')
    mapping_df : pd.DataFrame
        Electrode mappings DataFrame
        
    Returns:
    --------
    str
        Rat ID to use for mapping ('9151' for 32-channel sessions, '9442' for 20-channel sessions)
    """
    if session_id in RAT_9442_32_CHANNEL_SESSIONS:
        return '9151'  # Use rat 9151's mapping for 32-channel sessions
    else:
        return '9442'  # Use rat 9442's mapping for 20-channel sessions


def process_single_rat_multi_session(
    rat_id: str,
    roi_or_channels: Union[str, List[int]],
    pkl_path: str,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 30,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    base_save_path: str = 'results/cross_rats',
    show_plots: bool = False,
    method: str = 'mne',
    verbose: bool = True
) -> Tuple[str, Optional[Dict]]:
    """
    Process multi-session analysis for a single rat using nm_theta_analyzer.
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to process
    roi_or_channels : Union[str, List[int]]
        ROI specification
    pkl_path : str
        Path to main data file
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Cycles factor for spectrograms
    base_save_path : str
        Base directory for saving results
    show_plots : bool
        Whether to show plots during processing
    method : str
        Spectrogram calculation method: 'mne' (MNE-Python) or 'cwt' (SciPy CWT)
    verbose : bool
        Whether to print detailed progress information (default: True)
        
    Returns:
    --------
    rat_id : str
        The processed rat ID
    results : Optional[Dict]
        Multi-session results for the rat, or None if failed
    """
    if verbose:
        print(f"\n🐀 Processing rat {rat_id} - Multi-session analysis")
        print(f"    Method: {method.upper()} ({'MNE-Python' if method == 'mne' else 'SciPy CWT'})")
        if rat_id == '9442':
            print(f"    Special handling: Mixed 32/20 channel sessions")
        print("=" * 60)
    
    try:
        # Create save path for this rat
        rat_save_path = os.path.join(base_save_path, f'rat_{rat_id}_multi_session_{method}')
        
        # Special handling for rat 9442
        if rat_id == '9442':
            # Load electrode mappings for session-specific mapping
            try:
                mapping_df = load_electrode_mappings()
                if verbose:
                    print(f"    Loaded electrode mappings for rat 9442 special handling")
                    print(f"    32-channel sessions: {RAT_9442_32_CHANNEL_SESSIONS}")
                    print(f"    20-channel sessions: All others")
                    print(f"    ⚠️  NOTE: Session-specific mapping requires modification of underlying analysis functions")
            except Exception as e:
                if verbose:
                    print(f"    ⚠️  Failed to load electrode mappings: {e}")
                mapping_df = None
        else:
            mapping_df = None
        
        if method == 'mne':
            # Use nm_theta_analyzer for MNE-based analysis
            results = run_analysis(
                mode='multi',
                method='basic',  # method ignored for multi-session
                parallel_type=None,
                pkl_path=pkl_path,
                roi=roi_or_channels if isinstance(roi_or_channels, str) else ','.join(map(str, roi_or_channels)),
                session_index=None,  # ignored for multi-session
                rat_id=rat_id,
                freq_min=freq_range[0],
                freq_max=freq_range[1],
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                n_jobs=None,
                batch_size=8,
                save_path=rat_save_path,
                show_plots=show_plots,
                show_frequency_profiles=False
            )
        
        elif method == 'cwt':
            # Use vectorized CWT analysis
            from nm_theta_single_vectorized import analyze_rat_multi_session_vectorized
            
            results = analyze_rat_multi_session_vectorized(
                rat_id=rat_id,
                roi_or_channels=roi_or_channels,
                pkl_path=pkl_path,
                freq_range=freq_range,
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                save_path=rat_save_path,
                mapping_df=None,
                show_plots=show_plots,
                session_n_jobs=None,
                channel_batch_size=8,
                method='cwt'
            )
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mne' or 'cwt'.")
        
        if verbose:
            print(f"✓ Successfully processed rat {rat_id}")
            print(f"  Sessions analyzed: {results.get('n_sessions_analyzed', 'unknown')}")
            print(f"  Results saved to: {rat_save_path}")
        
        # Force garbage collection
        gc.collect()
        
        return rat_id, results
        
    except Exception as e:
        if verbose:
            print(f"❌ Error processing rat {rat_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        return rat_id, None


def aggregate_cross_rats_results(
    rat_results: Dict[str, Dict],
    roi_specification: Union[str, List[int]],
    freq_range: Tuple[float, float],
    save_path: str,
    verbose: bool = True
) -> Dict:
    """
    Aggregate multi-session results across all rats.
    
    Parameters:
    -----------
    rat_results : Dict[str, Dict]
        Dictionary mapping rat_id -> multi-session results
    roi_specification : Union[str, List[int]]
        ROI specification used for analysis
    freq_range : Tuple[float, float]
        Frequency range used
    save_path : str
        Path to save aggregated results
    verbose : bool
        Whether to print detailed progress information (default: True)
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    if verbose:
        print(f"\n📊 Aggregating results across {len(rat_results)} rats")
        print("=" * 60)
    
    # Filter out failed results
    valid_results = {rat_id: results for rat_id, results in rat_results.items() if results is not None}
    n_valid_rats = len(valid_results)
    
    if n_valid_rats == 0:
        raise ValueError("No valid rat results to aggregate")
    
    if verbose:
        print(f"Valid results from {n_valid_rats} rats: {list(valid_results.keys())}")
    
    # Get reference data from first valid result
    first_result = next(iter(valid_results.values()))
    frequencies = first_result['frequencies']
    n_freqs = len(frequencies)
    
    # Initialize aggregation containers
    aggregated_windows = defaultdict(lambda: {
        'spectrograms': [],
        'total_events': [],
        'n_sessions': [],
        'rat_ids': []
    })
    
    # Collect spectrograms from all rats
    for rat_id, results in valid_results.items():
        if verbose:
            print(f"Processing results from rat {rat_id}")
            
            # Debug: Print available keys
            print(f"  Available keys in results: {list(results.keys())}")
            if 'averaged_windows' in results:
                print(f"  NM sizes in averaged_windows: {list(results['averaged_windows'].keys())}")
            else:
                print(f"  ❌ ERROR: 'averaged_windows' key not found!")
                print(f"  Available keys: {list(results.keys())}")
                continue
        elif 'averaged_windows' not in results:
            continue
        
        for nm_size, window_data in results['averaged_windows'].items():
            spectrograms = aggregated_windows[nm_size]['spectrograms']
            spectrograms.append(window_data['avg_spectrogram'])
            
            aggregated_windows[nm_size]['total_events'].append(window_data['n_events'])
            aggregated_windows[nm_size]['n_sessions'].append(window_data['n_sessions'])
            aggregated_windows[nm_size]['rat_ids'].append(rat_id)
    
    # Compute cross-rats averages
    final_aggregated_windows = {}
    
    for nm_size, data in aggregated_windows.items():
        spectrograms = np.array(data['spectrograms'])  # Shape: (n_rats, n_freqs, n_times)
        
        if verbose:
            print(f"NM size {nm_size}: {spectrograms.shape[0]} rats, "
                  f"spectrogram shape: {spectrograms.shape[1:]}") 
        
        # Average across rats
        avg_spectrogram = np.mean(spectrograms, axis=0)
        
        final_aggregated_windows[nm_size] = {
            'avg_spectrogram': avg_spectrogram,
            'individual_spectrograms': spectrograms,
            'window_times': first_result['averaged_windows'][nm_size]['window_times'],
            'total_events_per_rat': data['total_events'],
            'n_sessions_per_rat': data['n_sessions'],
            'rat_ids': data['rat_ids'],
            'n_rats': spectrograms.shape[0],
            'total_events_all_rats': sum(data['total_events']),
            'total_sessions_all_rats': sum(data['n_sessions'])
        }
    
    # Create comprehensive results dictionary
    aggregated_results = {
        'analysis_type': 'cross_rats',
        'rat_ids': list(valid_results.keys()),
        'n_rats': n_valid_rats,
        'roi_specification': roi_specification,
        'roi_channels': first_result['roi_channels'],
        'frequencies': frequencies,
        'averaged_windows': final_aggregated_windows,
        'analysis_parameters': {
            'frequency_range': freq_range,
            'n_frequencies': n_freqs,
            'window_duration': first_result['analysis_parameters']['window_duration'],
            'normalization': first_result['analysis_parameters']['normalization']
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
    
    results_file = os.path.join(save_path, 'cross_rats_aggregated_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    if verbose:
        print(f"✓ Cross-rats results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = os.path.join(save_path, 'cross_rats_summary.json')
    summary_data = {
        'n_rats': n_valid_rats,
        'rat_ids': list(valid_results.keys()),
        'nm_sizes_analyzed': [float(key) for key in final_aggregated_windows.keys()],
        'total_events_all_rats': {nm_size: data['total_events_all_rats'] 
                                  for nm_size, data in final_aggregated_windows.items()},
        'total_sessions_all_rats': {nm_size: data['total_sessions_all_rats']
                                    for nm_size, data in final_aggregated_windows.items()},
        'frequency_range': freq_range,
        'roi_specification': str(roi_specification),
        'roi_channels': first_result['roi_channels'].tolist() if hasattr(first_result['roi_channels'], 'tolist') else first_result['roi_channels']
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    if verbose:
        print(f"✓ Summary statistics saved to: {summary_file}")
    
    return aggregated_results


def calculate_color_limits(spectrograms: List[np.ndarray], percentile: float = 95.0) -> Tuple[float, float]:
    """
    Calculate reasonable color map limits based on the actual data.
    
    Parameters:
    -----------
    spectrograms : List[np.ndarray]
        List of spectrogram arrays
    percentile : float
        Percentile to use for clipping outliers (default: 95.0)
        
    Returns:
    --------
    vmin, vmax : Tuple[float, float]
        Color map limits
    """
    # Concatenate all spectrograms to get overall data range
    all_data = np.concatenate([spec.flatten() for spec in spectrograms])
    
    # Calculate percentiles to clip outliers
    vmin = float(np.percentile(all_data, 100 - percentile))
    vmax = float(np.percentile(all_data, percentile))
    
    # Ensure symmetric limits around zero for better visualization
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
    
    # Round to reasonable decimal places
    vmin = round(vmin, 2)
    vmax = round(vmax, 2)
    
    return vmin, vmax


def create_cross_rats_visualizations(results: Dict, save_path: str, verbose: bool = True):
    """
    Create comprehensive visualizations for cross-rats results.
    
    Parameters:
    -----------
    results : Dict
        Cross-rats aggregated results
    save_path : str
        Directory to save visualizations
    verbose : bool
        Whether to print visualization progress (default: True)
    """
    if verbose:
        print(f"\n📈 Creating cross-rats visualizations")
        print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    
    frequencies = results['frequencies']
    n_rats = results['n_rats']
    roi_channels = results['roi_channels']
    
    # Create figure with subplots for each NM size
    nm_sizes = [float(key) for key in results['averaged_windows'].keys()]
    n_nm_sizes = len(nm_sizes)
    
    if n_nm_sizes == 0:
        if verbose:
            print("⚠️  No NM sizes to plot")
        return
    
    # Calculate color limits from all spectrograms
    all_spectrograms = []
    for nm_size in nm_sizes:
        window_data = results['averaged_windows'][nm_size]
        all_spectrograms.append(window_data['avg_spectrogram'])
        all_spectrograms.extend(window_data['individual_spectrograms'])
    
    vmin, vmax = calculate_color_limits(all_spectrograms)
    if verbose:
        print(f"📊 Color map limits: [{vmin}, {vmax}] (calculated from data)")
    
    fig, axes = plt.subplots(n_nm_sizes, 1, figsize=(10, 5 * n_nm_sizes))
    if n_nm_sizes == 1:
        axes = [axes]
    
    for i, nm_size in enumerate(nm_sizes):
        window_data = results['averaged_windows'][nm_size]
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        
        # Plot: Average spectrogram across rats using pcolormesh for better frequency axis
        ax = axes[i]
        
        # Use log-frequency spacing for y-axis
        log_frequencies = np.log10(frequencies)
        im = ax.pcolormesh(window_times, log_frequencies, avg_spectrogram,
                          shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'NM Size {nm_size} - Average Across {n_rats} Rats\n'
                    f'Total Events: {window_data["total_events_all_rats"]}, '
                    f'Sessions: {window_data["total_sessions_all_rats"]}')
        
        # Set y-axis ticks to show frequencies on log scale
        # Create tick positions at log-frequency values
        ax.set_yticks(log_frequencies)
        
        # Create labels array with empty strings for most frequencies
        freq_labels = [''] * len(frequencies)
        
        # Always show first and last frequency
        freq_labels[0] = f'{frequencies[0]:.1f}'
        freq_labels[-1] = f'{frequencies[-1]:.1f}'
        
        # Show 5 intermediate frequencies, excluding ones too close to first/last
        if len(frequencies) > 6:  # Need at least 7 frequencies for this approach
            n_intermediate = 5
            indices = np.linspace(1, len(frequencies)-2, n_intermediate, dtype=int)
            
            # Exclude indices too close to first (0) and last (-1)
            min_distance = max(1, len(frequencies) // 5)  # At least 20% away from edges
            
            for idx in indices:
                if idx >= min_distance and idx <= len(frequencies) - min_distance - 1:
                    freq_labels[idx] = f'{frequencies[idx]:.1f}'
        
        ax.set_yticklabels(freq_labels)
        
        plt.colorbar(im, ax=ax, label='Z-score')
    
    # Add overall title
    roi_str = f"ROI: {results['roi_specification']} (channels: {roi_channels})"
    freq_str = f"Freq: {results['analysis_parameters']['frequency_range'][0]}-{results['analysis_parameters']['frequency_range'][1]} Hz"
    
    plt.suptitle(f'Cross-Rats NM Theta Analysis\n{roi_str}, {freq_str}', fontsize=14)
    
    # Apply custom spacing parameters for better plot layout
    plt.subplots_adjust(left=0.052, bottom=0.07, right=0.55, top=0.924, wspace=0.206, hspace=0.656)
    
    # Save plot
    plot_file = os.path.join(save_path, 'cross_rats_spectrograms.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"✓ Spectrograms saved to: {plot_file}")
    
    plt.show()
    
    # Create individual rat comparison plot
    if n_nm_sizes == 1:  # Only create if single NM size to avoid complexity
        nm_size = nm_sizes[0]
        window_data = results['averaged_windows'][nm_size]
        individual_spectrograms = window_data['individual_spectrograms']
        rat_ids = window_data['rat_ids']
        
        n_rats_to_show = min(6, len(rat_ids))  # Show up to 6 rats
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for j in range(n_rats_to_show):
            ax = axes[j]
            spectrogram = individual_spectrograms[j]
            rat_id = rat_ids[j]
            
            # Use log-frequency spacing for y-axis
            log_frequencies = np.log10(frequencies)
            im = ax.pcolormesh(window_times, log_frequencies, spectrogram,
                              shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Rat {rat_id}\nEvents: {window_data["total_events_per_rat"][j]}, '
                        f'Sessions: {window_data["n_sessions_per_rat"][j]}')
            
            # Set y-axis ticks to show frequencies on log scale
            # Create tick positions at log-frequency values
            ax.set_yticks(log_frequencies)
            
            # Create labels array with empty strings for most frequencies
            freq_labels = [''] * len(frequencies)
            
            # Always show first and last frequency
            freq_labels[0] = f'{frequencies[0]:.1f}'
            freq_labels[-1] = f'{frequencies[-1]:.1f}'
            
            # Show 5 intermediate frequencies, excluding ones too close to first/last
            if len(frequencies) > 6:  # Need at least 7 frequencies for this approach
                n_intermediate = 5
                indices = np.linspace(1, len(frequencies)-2, n_intermediate, dtype=int)
                
                # Exclude indices too close to first (0) and last (-1)
                min_distance = max(1, len(frequencies) // 10)  # At least 10% away from edges
                
                for idx in indices:
                    if idx >= min_distance and idx <= len(frequencies) - min_distance - 1:
                        freq_labels[idx] = f'{frequencies[idx]:.1f}'
            
            ax.set_yticklabels(freq_labels)
            
            plt.colorbar(im, ax=ax, label='Z-score')
        
        # Hide unused subplots
        for j in range(n_rats_to_show, 6):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Individual Rat Spectrograms - NM Size {nm_size}\n{roi_str}, {freq_str}', fontsize=14)
        
        # Apply custom spacing parameters for individual rats plot
        plt.subplots_adjust(left=0.052, bottom=0.07, right=0.55, top=0.924, wspace=0.206, hspace=0.656)
        
        individual_plot_file = os.path.join(save_path, f'individual_rats_nm_{nm_size}.png')
        plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Individual rats plot saved to: {individual_plot_file}")
        
        plt.show()


def run_cross_rats_analysis(
    roi: str,
    pkl_path: str = 'data/processed/all_eeg_data.pkl',
    freq_min: float = 3.0,
    freq_max: float = 8.0,
    n_freqs: int = 30,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    rat_ids: Optional[List[str]] = None,
    save_path: str = 'results/cross_rats',
    show_plots: bool = False,
    method: str = 'mne',
    verbose: bool = True
) -> Dict:
    """
    Run cross-rats NM theta analysis with direct parameter specification.
    
    Parameters:
    -----------
    roi : str
        ROI name ("frontal", "hippocampus") or channels ("1,2,3")
    pkl_path : str
        Path to main EEG data file
    freq_min : float
        Minimum frequency (Hz)
    freq_max : float
        Maximum frequency (Hz)
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration (s)
    n_cycles_factor : float
        Cycles factor for spectrograms
    rat_ids : Optional[List[str]]
        Specific rat IDs to process, None for all rats
    save_path : str
        Base directory for saving results
    show_plots : bool
        Show plots during processing
    method : str
        Spectrogram calculation method: 'mne' (MNE-Python) or 'cwt' (SciPy CWT)
    verbose : bool
        Whether to print detailed progress information (default: True)
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    if verbose:
        print("🧠 Cross-Rats NM Theta Analysis")
        print("=" * 80)
        print(f"Data file: {pkl_path}")
        print(f"ROI: {roi}")
        print(f"Frequency range: {freq_min}-{freq_max} Hz ({n_freqs} freqs)")
        print(f"Window duration: {window_duration}s")
        print(f"Method: {method.upper()} ({'MNE-Python' if method == 'mne' else 'SciPy CWT'})")
        print(f"Save path: {save_path}")
        print("=" * 80)
    
    # Validate method parameter
    if method not in ['mne', 'cwt']:
        raise ValueError(f"Invalid method: {method}. Use 'mne' or 'cwt'.")
    
    # Discover rat IDs
    if rat_ids:
        if verbose:
            print(f"Using specified rat IDs: {rat_ids}")
        
        # Check compatibility for rat 9442 even when manually specified
        if '9442' in rat_ids:
            try:
                mapping_df = load_electrode_mappings()
                if verbose:
                    print(f"\n🔍 Checking rat 9442 compatibility with requested ROI/channels...")
                
                is_compatible = check_rat_9442_compatibility(roi, mapping_df, verbose)
                
                if not is_compatible:
                    rat_ids = [r for r in rat_ids if r != '9442']
                    if verbose:
                        print(f"❌ Removing rat 9442 from analysis (incompatible with requested ROI/channels)")
                    if not rat_ids:
                        raise ValueError("No compatible rats remaining for analysis")
                        
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error checking rat 9442 compatibility: {e}")
                    print(f"❌ Removing rat 9442 from analysis (compatibility check failed)")
                rat_ids = [r for r in rat_ids if r != '9442']
                if not rat_ids:
                    raise ValueError("No compatible rats remaining for analysis")
    else:
        rat_ids = discover_rat_ids(pkl_path, verbose=verbose, roi_or_channels=roi)
    
    # Process each rat individually
    rat_results = {}
    failed_rats = []
    error_details = {}
    successful_rats = []
    
    for rat_id in rat_ids:
        rat_id_str, results = process_single_rat_multi_session(
            rat_id=rat_id,
            roi_or_channels=roi,
            pkl_path=pkl_path,
            freq_range=(freq_min, freq_max),
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            base_save_path=save_path,
            show_plots=show_plots,
            method=method,
            verbose=verbose
        )
        
        rat_results[rat_id_str] = results
        
        # Track success/failure
        if results is None:
            failed_rats.append(rat_id_str)
            error_details[rat_id_str] = "Processing failed - see logs above for details"
        else:
            successful_rats.append(rat_id_str)
        
        # Force garbage collection after each rat
        gc.collect()
    
    # Aggregate results across rats
    aggregated_results = aggregate_cross_rats_results(
        rat_results=rat_results,
        roi_specification=roi,
        freq_range=(freq_min, freq_max),
        save_path=save_path,
        verbose=verbose
    )
    
    # Create visualizations
    create_cross_rats_visualizations(
        results=aggregated_results,
        save_path=save_path,
        verbose=verbose
    )
    
    # Print analysis completion and error summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total rats attempted: {len(rat_ids)}")
    print(f"Successfully processed: {len(successful_rats)}")
    print(f"Failed to process: {len(failed_rats)}")
    
    if successful_rats:
        print(f"\n✅ Successful rats: {successful_rats}")
    
    if failed_rats:
        print(f"\n❌ Failed rats: {failed_rats}")
        print("\n🔍 Error details:")
        for rat_id, error in error_details.items():
            print(f"  • Rat {rat_id}: {error}")
    
    if len(successful_rats) > 0:
        print("\n✅ Cross-rats analysis completed successfully!")
        if verbose:
            print(f"Results saved to: {save_path}")
            print(f"\nDetailed Summary:")
            print(f"  Method used: {method.upper()} ({'MNE-Python' if method == 'mne' else 'SciPy CWT'})")
            print(f"  Rats successfully processed: {aggregated_results['n_rats']}")
            print(f"  Successful rat IDs: {aggregated_results['rat_ids']}")
            print(f"  NM sizes analyzed: {[float(key) for key in aggregated_results['averaged_windows'].keys()]}")
            print(f"  ROI channels: {aggregated_results['roi_channels']}")
    else:
        print("\n❌ Analysis failed - no rats were successfully processed!")
    
    return aggregated_results


def main():
    """
    Main function for cross-rats NM theta analysis.
    Supports both command line arguments and direct parameter modification.
    """
    parser = argparse.ArgumentParser(
        description='Cross-Rats NM Theta Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze frontal ROI across all rats (MNE method)
  python nm_theta_cross_rats.py --roi frontal --freq_min 3 --freq_max 8
  
  # Analyze frontal ROI using SciPy CWT method
  python nm_theta_cross_rats.py --roi frontal --freq_min 3 --freq_max 8 --method cwt
  
  # Analyze specific channels across all rats (CWT method)
  python nm_theta_cross_rats.py --roi "1,2,3" --freq_min 1 --freq_max 12 --method cwt
  
  # Analyze hippocampus with custom parameters (MNE method)
  python nm_theta_cross_rats.py --roi hippocampus --freq_min 6 --freq_max 10 --n_freqs 20 --method mne
        """
    )
    
    # Data parameters
    parser.add_argument('--pkl_path', default='data/processed/all_eeg_data.pkl',
                       help='Path to main EEG data file')
    parser.add_argument('--roi', required=True,
                       help='ROI name ("frontal", "hippocampus") or channels ("1,2,3")')
    
    # Analysis parameters
    parser.add_argument('--freq_min', type=float, default=3.0,
                       help='Minimum frequency (Hz)')
    parser.add_argument('--freq_max', type=float, default=8.0,
                       help='Maximum frequency (Hz)')
    parser.add_argument('--n_freqs', type=int, default=30,
                       help='Number of frequencies')
    parser.add_argument('--window_duration', type=float, default=1.0,
                       help='Event window duration (s)')
    parser.add_argument('--n_cycles_factor', type=float, default=3.0,
                       help='Cycles factor for spectrograms')
    
    # Method parameter
    parser.add_argument('--method', choices=['mne', 'cwt'], default='mne',
                       help='Spectrogram calculation method: mne (MNE-Python) or cwt (SciPy CWT)')
    
    # Processing parameters
    parser.add_argument('--rat_ids', type=str, default=None,
                       help='Specific rat IDs to process (comma-separated), default: all rats')
    parser.add_argument('--save_path', default='results/cross_rats',
                       help='Base directory for saving results')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots during processing')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed progress output (opposite of verbose)')
    
    args = parser.parse_args()
    
    # Parse rat_ids if provided
    rat_ids = None
    if args.rat_ids:
        rat_ids = [r.strip() for r in args.rat_ids.split(',')]
    
    # Parse verbose flag (opposite of quiet)
    verbose = not args.quiet
    
    # Run the analysis
    return run_cross_rats_analysis(
        roi=args.roi,
        pkl_path=args.pkl_path,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        n_freqs=args.n_freqs,
        window_duration=args.window_duration,
        n_cycles_factor=args.n_cycles_factor,
        rat_ids=rat_ids,
        save_path=args.save_path,
        show_plots=args.show_plots,
        method=args.method,
        verbose=verbose
    )


def run_all_channels_analysis(
    pkl_path: str = 'data/processed/all_eeg_data.pkl',
    freq_min: float = 1.0,
    freq_max: float = 45.0,
    n_freqs: int = 30,
    window_duration: float = 2.0,
    n_cycles_factor: float = 3.0,
    rat_ids: Optional[List[str]] = None,
    base_save_path: str = 'results',
    show_plots: bool = False,
    channels: Optional[List[int]] = None
) -> Dict[str, Dict]:
    """
    Run cross-rats NM theta analysis for all channels sequentially with memory cleanup.
    
    Parameters:
    -----------
    pkl_path : str
        Path to main EEG data file
    freq_min : float
        Minimum frequency (Hz)
    freq_max : float
        Maximum frequency (Hz)
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration (s)
    n_cycles_factor : float
        Cycles factor for spectrograms
    rat_ids : Optional[List[str]]
        Specific rat IDs to process, None for all rats
    base_save_path : str
        Base directory for saving results
    show_plots : bool
        Show plots during processing
    channels : Optional[List[int]]
        Specific channels to process, None for all channels (1-32)
        
    Returns:
    --------
    all_results : Dict[str, Dict]
        Dictionary mapping channel -> results for each channel
    """
    print("🧠 Cross-Rats NM Theta Analysis - All Channels")
    print("=" * 80)
    print(f"Data file: {pkl_path}")
    print(f"Frequency range: {freq_min}-{freq_max} Hz ({n_freqs} freqs)")
    print(f"Window duration: {window_duration}s")
    print(f"Base save path: {base_save_path}")
    print("=" * 80)
    
    # Define channels to process
    if channels is None:
        channels = list(range(1, 33))  # Channels 1-32
    else:
        channels = [int(ch) for ch in channels]
    
    print(f"📊 Processing {len(channels)} channels: {channels}")
    
    # Create main results directory
    all_channels_dir = os.path.join(base_save_path, 'all_channels_analysis')
    os.makedirs(all_channels_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(all_channels_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Store results for all channels
    all_results = {}
    
    # Discover rat IDs once
    if rat_ids:
        print(f"Using specified rat IDs: {rat_ids}")
    else:
        rat_ids = discover_rat_ids(pkl_path)
    
    for i, channel in enumerate(channels):
        print(f"\n{'='*60}")
        print(f"📡 Processing Channel {channel} ({i+1}/{len(channels)})")
        print(f"{'='*60}")
        
        try:
            # Create temporary directory for this channel
            temp_save_path = os.path.join(base_save_path, 'cross_rats')
            
            # Run analysis for this channel
            results = run_cross_rats_analysis(
                roi=str(channel),
                pkl_path=pkl_path,
                freq_min=freq_min,
                freq_max=freq_max,
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                rat_ids=rat_ids,
                save_path=temp_save_path,
                show_plots=show_plots
            )
            
            # Store results
            all_results[f"channel_{channel}"] = results
            
            # Move plots to organized directory
            channel_plots_dir = os.path.join(plots_dir, f'channel_{channel}')
            os.makedirs(channel_plots_dir, exist_ok=True)
            
            # Move plot files
            plot_files = [
                'cross_rats_spectrograms.png',
                'cross_rats_aggregated_results.pkl',
                'cross_rats_summary.json'
            ]
            
            for plot_file in plot_files:
                src_path = os.path.join(temp_save_path, plot_file)
                dst_path = os.path.join(channel_plots_dir, plot_file)
                if os.path.exists(src_path):
                    import shutil
                    shutil.move(src_path, dst_path)
            
            # Move individual rat plots if they exist
            for nm_size in results['averaged_windows'].keys():
                individual_plot = f'individual_rats_nm_{nm_size}.png'
                src_path = os.path.join(temp_save_path, individual_plot)
                dst_path = os.path.join(channel_plots_dir, individual_plot)
                if os.path.exists(src_path):
                    import shutil
                    shutil.move(src_path, dst_path)
            
            print(f"✓ Channel {channel} completed successfully")
            print(f"  Results saved to: {channel_plots_dir}")
            
        except Exception as e:
            print(f"❌ Error processing channel {channel}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[f"channel_{channel}"] = None
            
        finally:
            # Clean up temporary files and memory
            print(f"🧹 Cleaning up memory and temporary files for channel {channel}")
            
            # Remove temporary cross_rats directory
            if os.path.exists(temp_save_path):
                import shutil
                try:
                    shutil.rmtree(temp_save_path)
                    print(f"  ✓ Removed temporary directory: {temp_save_path}")
                except Exception as e:
                    print(f"  ⚠️  Could not remove temporary directory: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to ensure cleanup
            import time
            time.sleep(1)
    
    # Save comprehensive results
    all_results_file = os.path.join(all_channels_dir, 'all_channels_results.pkl')
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create summary
    successful_channels = [ch for ch, results in all_results.items() if results is not None]
    failed_channels = [ch for ch, results in all_results.items() if results is None]
    
    summary = {
        'total_channels': len(channels),
        'successful_channels': len(successful_channels),
        'failed_channels': len(failed_channels),
        'successful_channel_list': [ch.split('_')[1] for ch in successful_channels],
        'failed_channel_list': [ch.split('_')[1] for ch in failed_channels],
        'analysis_parameters': {
            'frequency_range': [freq_min, freq_max],
            'n_frequencies': n_freqs,
            'window_duration': window_duration,
            'n_cycles_factor': n_cycles_factor
        },
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(all_channels_dir, 'all_channels_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ All Channels Analysis Completed!")
    print(f"{'='*80}")
    print(f"📊 Summary:")
    print(f"  Total channels processed: {len(channels)}")
    print(f"  Successful: {len(successful_channels)}")
    print(f"  Failed: {len(failed_channels)}")
    print(f"  Results directory: {all_channels_dir}")
    print(f"  Plots directory: {plots_dir}")
    
    if failed_channels:
        print(f"\n❌ Failed channels: {[ch.split('_')[1] for ch in failed_channels]}")
    
    return all_results


# Example usage for IDE/notebook environment
if __name__ == "__main__":
    # Check if we're running in IDE mode (with direct function calls)
    # or command line mode
    import sys
    
    # If there are command line arguments, use command line mode
    if len(sys.argv) > 1:
        # For command line usage, call main()
        main()
    else:
        # For IDE usage, run the analysis directly
        # Uncomment the line below to run all channels analysis
        #all_results = run_all_channels_analysis()
        
        # Or run single channel analysis as before 
        results = run_cross_rats_analysis(
            roi="1,2,3",                    # ROI specification
            pkl_path="data/processed/all_eeg_data.pkl",  # Data file path
            freq_min=1.0,                     # Minimum frequency
            freq_max=45.0,                     # Maximum frequency
            n_freqs=40,                       # Number of frequencies
            window_duration=2.0,              # Event window duration
            n_cycles_factor=3.0,              # Cycles factor
            rat_ids=None,                     # None for all rats, or ["10501", "1055"] for specific rats
            save_path="results/cross_rats",   # Save directory
            show_plots=False,                 # Show plots during processing
            method="mne",                     # Spectrogram method: "mne" (MNE-Python) or "cwt" (SciPy CWT)
            verbose=False                      # Print detailed progress information
        )
