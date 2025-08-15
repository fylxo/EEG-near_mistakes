#!/usr/bin/env python3
"""
Cross-Rats NM Theta Analysis - Multiple Electrodes Hierarchical Averaging

This script performs NM theta analysis with hierarchical averaging:
1. Trial averaging per electrode
2. Electrode averaging per session  
3. Session averaging per rat
4. SEM calculation across rats

Key Features:
- Hierarchical averaging approach: trials → electrodes → sessions → rats
- Flexible electrode selection (any combination of electrodes)
- Computes single theta power value per rat with SEM across rats
- Supports custom time windows and frequency ranges

Author: Generated for cross-rats EEG near-mistake analysis with hierarchical averaging
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
from scipy import stats

# Configure UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        import io
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import AnalysisConfig, DataConfig, PlottingConfig


def load_electrode_mappings(mapping_file: str = None) -> pd.DataFrame:
    """Load electrode mappings from CSV file."""
    if mapping_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_file = os.path.join(script_dir, '..', 'electrode_mapping.csv')
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Electrode mapping file not found: {mapping_file}")
    
    return pd.read_csv(mapping_file)


def get_electrode_numbers_from_selection(electrode_selection: Union[str, List[int]], 
                                       electrode_mapping: pd.DataFrame = None) -> List[int]:
    """
    Convert electrode selection to list of electrode numbers.
    
    Parameters:
    -----------
    electrode_selection : Union[str, List[int]]
        Either ROI name (e.g., 'frontal') or list of electrode numbers
    electrode_mapping : pd.DataFrame
        Electrode mapping dataframe
        
    Returns:
    --------
    List[int]
        List of electrode numbers
    """
    if isinstance(electrode_selection, list):
        return electrode_selection
    
    if electrode_mapping is None:
        electrode_mapping = load_electrode_mappings()
    
    # ROI-based selection
    roi_name = electrode_selection.lower()
    roi_column = f"{roi_name}_roi"
    
    if roi_column not in electrode_mapping.columns:
        raise ValueError(f"ROI '{roi_name}' not found in electrode mapping. Available: {electrode_mapping.columns}")
    
    electrodes = electrode_mapping[electrode_mapping[roi_column] == 1]['electrode_number'].tolist()
    
    if not electrodes:
        raise ValueError(f"No electrodes found for ROI '{roi_name}'")
    
    return electrodes


def discover_rat_ids(pkl_path: str, exclude_20_channel_rats: bool = False, verbose: bool = True, 
                     roi_or_channels: Optional[Union[str, List[int]]] = None) -> List[str]:
    """
    Discover all unique rat IDs from the dataset (copied from original nm_theta_cross_rats.py).
    """
    if verbose:
        print(f"🔍 Discovering rat IDs from {pkl_path}")
        print("Loading data to scan for rat IDs...")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    rat_ids = set()
    for session_data in all_data:
        if isinstance(session_data, dict) and 'session_id' in session_data:
            session_id = session_data['session_id']
            if '_' in session_id:
                rat_id = session_id.split('_')[0]
                rat_ids.add(rat_id)
    
    rat_ids = sorted(list(rat_ids))
    
    if exclude_20_channel_rats:
        # Remove rat 9442 if requested
        rat_ids = [r for r in rat_ids if r != '9442']
        if verbose:
            print("Excluding 20-channel rats (rat 9442)")
    
    if verbose:
        print(f"Found {len(rat_ids)} rats: {rat_ids}")
    
    return rat_ids


def process_single_rat_hierarchical_averaging(rat_id: str,
                                             electrode_selection: Union[str, List[int]],
                                             all_data: List[Dict],
                                             time_window: Tuple[float, float] = (-0.2, 0.0),
                                             freq_range: Tuple[float, float] = (3, 8),
                                             verbose: bool = True) -> Optional[Dict[str, float]]:
    """
    Process a single rat with hierarchical averaging: trials → electrodes → sessions → rat.
    
    This implements the exact procedure you described:
    1. For each trial and electrode, compute windowed power in time/freq window
    2. Average across trials for each electrode (trial-averaged power per electrode)
    3. Average across electrodes for each session (session-averaged power)
    4. Average across sessions for the rat (rat-averaged power)
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to process
    electrode_selection : Union[str, List[int]]
        Either ROI name or list of electrode numbers
    all_data : List[Dict]
        All session data from the pickle file
    time_window : Tuple[float, float]
        Time window in seconds (e.g., (-0.2, 0.0))
    freq_range : Tuple[float, float]
        Frequency range in Hz (e.g., (3, 8))
    verbose : bool
        Print debug information
        
    Returns:
    --------
    Optional[Dict[str, float]]
        Dictionary mapping NM size to rat-level averaged power, or None if failed
        Keys: 'nm_1', 'nm_2', 'nm_3'
    """
    try:
        # Get electrode numbers
        electrode_numbers = get_electrode_numbers_from_selection(electrode_selection)
        
        if verbose:
            print(f"Processing rat {rat_id} with electrodes: {electrode_numbers}")
        
        # Find sessions for this rat
        rat_sessions = []
        for session_data in all_data:
            if isinstance(session_data, dict) and 'session_id' in session_data:
                session_id = session_data['session_id']
                if session_id.startswith(f"{rat_id}_"):
                    rat_sessions.append(session_data)
        
        if not rat_sessions:
            if verbose:
                print(f"No sessions found for rat {rat_id}")
            return None
        
        if verbose:
            print(f"Found {len(rat_sessions)} sessions for rat {rat_id}")
        
        # Process each session with hierarchical averaging, separated by NM size
        session_powers_by_nm_size = {'nm_1': [], 'nm_2': [], 'nm_3': []}
        
        for session_data in rat_sessions:
            session_id = session_data['session_id']
            
            # Check if we have the required data
            if 'nm_events' not in session_data or 'times' not in session_data or 'frequencies' not in session_data:
                if verbose:
                    print(f"Warning: Missing required data in session {session_id}")
                continue
            
            times = session_data['times']
            frequencies = session_data['frequencies']
            nm_events = session_data['nm_events']
            
            # Get time and frequency masks
            time_mask = (times >= time_window[0]) & (times <= time_window[1])
            freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            
            if np.sum(time_mask) == 0 or np.sum(freq_mask) == 0:
                if verbose:
                    print(f"Warning: No data in time/freq window for session {session_id}")
                continue
            
            # Process each NM size separately
            for nm_size in [1, 2, 3]:
                nm_size_key = f'nm_{nm_size}'
                
                # STEP 1 & 2: For each electrode, compute trial-averaged power for this NM size
                electrode_powers = []
                
                for electrode_num in electrode_numbers:
                    electrode_key = f'electrode_{electrode_num}'
                    
                    if electrode_key not in session_data:
                        if verbose:
                            print(f"Warning: {electrode_key} not found in session {session_id}")
                        continue
                    
                    electrode_data = session_data[electrode_key]
                    
                    if 'power_windows' not in electrode_data or 'nm_sizes' not in electrode_data:
                        if verbose:
                            print(f"Warning: power_windows or nm_sizes not found for {electrode_key} in session {session_id}")
                        continue
                    
                    # Get power windows and corresponding NM sizes
                    power_windows = electrode_data['power_windows']
                    nm_sizes = electrode_data['nm_sizes']
                    
                    # Extract power for trials of this specific NM size
                    trial_powers = []
                    for trial_idx, (power_window, trial_nm_size) in enumerate(zip(power_windows, nm_sizes)):
                        if trial_nm_size == nm_size:
                            # power_window shape: (n_freqs, n_times)
                            windowed_power = power_window[freq_mask, :][:, time_mask]
                            
                            # Average across time and frequency for this trial
                            trial_power = np.mean(windowed_power)
                            trial_powers.append(trial_power)
                    
                    if trial_powers:
                        # STEP 2: Average across trials for this electrode and NM size
                        electrode_avg_power = np.mean(trial_powers)
                        electrode_powers.append(electrode_avg_power)
                        
                        if verbose:
                            print(f"  Electrode {electrode_num}, NM size {nm_size}: {len(trial_powers)} trials, avg power = {electrode_avg_power:.6f}")
                
                if electrode_powers:
                    # STEP 3: Average across electrodes for this session and NM size
                    session_avg_power = np.mean(electrode_powers)
                    session_powers_by_nm_size[nm_size_key].append(session_avg_power)
                    
                    if verbose:
                        print(f"Session {session_id}, NM size {nm_size}: {len(electrode_powers)} electrodes, avg power = {session_avg_power:.6f}")
        
        # STEP 4: Average across sessions for this rat, for each NM size
        rat_powers_by_nm_size = {}
        
        for nm_size_key in ['nm_1', 'nm_2', 'nm_3']:
            session_powers = session_powers_by_nm_size[nm_size_key]
            
            if session_powers:
                rat_avg_power = np.mean(session_powers)
                rat_powers_by_nm_size[nm_size_key] = rat_avg_power
                
                if verbose:
                    print(f"Rat {rat_id}, {nm_size_key}: {len(session_powers)} sessions, final power = {rat_avg_power:.6f}")
            else:
                if verbose:
                    print(f"No valid session powers for rat {rat_id}, {nm_size_key}")
                rat_powers_by_nm_size[nm_size_key] = np.nan
        
        # Return results only if we have at least one valid NM size
        if any(not np.isnan(power) for power in rat_powers_by_nm_size.values()):
            return rat_powers_by_nm_size
        else:
            if verbose:
                print(f"No valid powers for any NM size for rat {rat_id}")
            return None
        
    except Exception as e:
        if verbose:
            print(f"Error processing rat {rat_id}: {e}")
            import traceback
            traceback.print_exc()
        return None


def compute_trial_averaged_power_per_electrode(session_data: Dict,
                                              electrode_numbers: List[int],
                                              nm_size: int,
                                              time_window: Tuple[float, float],
                                              freq_range: Tuple[float, float],
                                              custom_frequencies: Optional[np.ndarray] = None,
                                              verbose: bool = False) -> Dict[int, float]:
    """
    STEP 1: For each electrode, compute windowed power for each trial, then average across trials.
    
    Returns:
    --------
    Dict[int, float]: electrode_number -> trial-averaged power
    """
    electrode_trial_averages = {}
    
    times = session_data['times']
    frequencies = session_data['frequencies']
    
    # Get time mask
    time_mask = (times >= time_window[0]) & (times <= time_window[1])
    
    # Get frequency mask - use custom frequencies if provided
    if custom_frequencies is not None:
        # Use custom frequency array - find closest matches in session frequencies
        freq_indices = []
        for custom_freq in custom_frequencies:
            closest_idx = np.argmin(np.abs(frequencies - custom_freq))
            freq_indices.append(closest_idx)
        freq_mask = np.zeros(len(frequencies), dtype=bool)
        freq_mask[freq_indices] = True
        
        if verbose:
            print(f"Using custom frequencies: {custom_frequencies}")
    else:
        # Use frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    
    if np.sum(time_mask) == 0 or np.sum(freq_mask) == 0:
        if verbose:
            print(f"Warning: No data in time/freq window")
        return {}
    
    for electrode_num in electrode_numbers:
        electrode_key = f'electrode_{electrode_num}'
        
        if electrode_key not in session_data:
            if verbose:
                print(f"Warning: {electrode_key} not found")
            continue
        
        electrode_data = session_data[electrode_key]
        
        if 'power_windows' not in electrode_data or 'nm_sizes' not in electrode_data:
            if verbose:
                print(f"Warning: missing power_windows or nm_sizes for {electrode_key}")
            continue
        
        power_windows = electrode_data['power_windows']
        nm_sizes = electrode_data['nm_sizes']
        
        # Extract trials for this specific NM size
        trial_powers = []
        for power_window, trial_nm_size in zip(power_windows, nm_sizes):
            if trial_nm_size == nm_size:
                # power_window shape: (n_freqs, n_times)
                windowed_power = power_window[freq_mask, :][:, time_mask]
                trial_power = np.mean(windowed_power)  # Average across time and frequency
                trial_powers.append(trial_power)
        
        if trial_powers:
            # STEP 1: Average across trials for this electrode
            electrode_trial_average = np.mean(trial_powers)
            electrode_trial_averages[electrode_num] = electrode_trial_average
            
            if verbose:
                print(f"  Electrode {electrode_num}, NM {nm_size}: {len(trial_powers)} trials -> {electrode_trial_average:.6f}")
    
    return electrode_trial_averages


def compute_session_averaged_power(electrode_trial_averages: Dict[int, float],
                                   verbose: bool = False) -> float:
    """
    STEP 2: Average trial-averaged powers across electrodes for this session.
    
    Returns:
    --------
    float: session-averaged power
    """
    if not electrode_trial_averages:
        return np.nan
    
    electrode_powers = list(electrode_trial_averages.values())
    session_average = np.mean(electrode_powers)
    
    if verbose:
        print(f"    Session average: {len(electrode_powers)} electrodes -> {session_average:.6f}")
    
    return session_average


def compute_rat_averaged_power(session_powers: List[float],
                               verbose: bool = False) -> float:
    """
    STEP 3: Average session-averaged powers across sessions for this rat.
    
    Returns:
    --------
    float: rat-averaged power
    """
    if not session_powers:
        return np.nan
    
    valid_powers = [p for p in session_powers if not np.isnan(p)]
    
    if not valid_powers:
        return np.nan
    
    rat_average = np.mean(valid_powers)
    
    if verbose:
        print(f"      Rat average: {len(valid_powers)} sessions -> {rat_average:.6f}")
    
    return rat_average


def process_single_rat_true_hierarchical(rat_id: str,
                                        electrode_selection: Union[str, List[int]],
                                        all_data: List[Dict],
                                        time_window: Tuple[float, float],
                                        freq_range: Tuple[float, float],
                                        custom_frequencies: Optional[np.ndarray] = None,
                                        verbose: bool = True) -> Optional[Dict[str, float]]:
    """
    Process a single rat using TRUE hierarchical averaging:
    1. Trial averaging per electrode
    2. Electrode averaging per session  
    3. Session averaging per rat
    
    Returns one power value per NM size for this rat.
    """
    try:
        electrode_numbers = get_electrode_numbers_from_selection(electrode_selection)
        
        if verbose:
            print(f"Processing rat {rat_id} with electrodes: {electrode_numbers}")
        
        # Find sessions for this rat
        rat_sessions = []
        for session_data in all_data:
            if isinstance(session_data, dict) and 'session_id' in session_data:
                session_id = session_data['session_id']
                if session_id.startswith(f"{rat_id}_"):
                    rat_sessions.append(session_data)
        
        if not rat_sessions:
            if verbose:
                print(f"No sessions found for rat {rat_id}")
            return None
        
        if verbose:
            print(f"Found {len(rat_sessions)} sessions for rat {rat_id}")
        
        # Process each NM size separately
        rat_powers = {}
        
        for nm_size in [1, 2, 3]:
            nm_size_key = f'nm_{nm_size}'
            
            if verbose:
                print(f"  Processing NM size {nm_size}")
            
            # STEP 3: Collect session-averaged powers for this NM size
            session_powers = []
            
            for session_data in rat_sessions:
                session_id = session_data['session_id']
                
                # Check required data
                if 'times' not in session_data or 'frequencies' not in session_data:
                    if verbose:
                        print(f"Warning: Missing times/frequencies in {session_id}")
                    continue
                
                # STEP 1: Compute trial-averaged power per electrode for this NM size
                electrode_trial_averages = compute_trial_averaged_power_per_electrode(
                    session_data, electrode_numbers, nm_size, time_window, freq_range, custom_frequencies, verbose
                )
                
                # STEP 2: Compute session-averaged power across electrodes
                session_power = compute_session_averaged_power(electrode_trial_averages, verbose)
                
                if not np.isnan(session_power):
                    session_powers.append(session_power)
                    if verbose:
                        print(f"    Session {session_id}: {session_power:.6f}")
            
            # STEP 3: Compute rat-averaged power across sessions
            rat_power = compute_rat_averaged_power(session_powers, verbose)
            rat_powers[nm_size_key] = rat_power
            
            if verbose:
                print(f"  Rat {rat_id}, NM {nm_size}: {rat_power:.6f}")
        
        return rat_powers
        
    except Exception as e:
        if verbose:
            print(f"Error processing rat {rat_id}: {e}")
            import traceback
            traceback.print_exc()
        return None


def load_frequencies_from_file(freq_file_path: str) -> np.ndarray:
    """
    Load frequencies from a text file (copied from nm_theta_cross_rats.py).
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
                    print(f"Warning: Skipping invalid frequency line: '{line}'")
    
    if not frequencies:
        raise ValueError(f"No valid frequencies found in {freq_file_path}")
    
    return np.array(frequencies)


def get_frequencies(freq_min: float, freq_max: float, freq_file_path: str = None) -> np.ndarray:
    """
    Get frequencies from file and filter by freq_min/freq_max (based on nm_theta_cross_rats.py).
    """
    if freq_file_path:
        # Load from file
        all_frequencies = load_frequencies_from_file(freq_file_path)
        
        # Filter frequencies in range
        main_frequencies = all_frequencies[(all_frequencies >= freq_min) & (all_frequencies <= freq_max)]
        
        if len(main_frequencies) == 0:
            raise ValueError(f"No frequencies in range {freq_min}-{freq_max} Hz found in {freq_file_path}")
        
        return main_frequencies
    else:
        raise ValueError("freq_file_path must be provided")


def run_cross_rats_hierarchical_analysis(electrode_selection: Union[str, List[int]],
                                        pkl_path: str = None,
                                        time_window: Tuple[float, float] = (-0.2, 0.0),
                                        freq_range: Tuple[float, float] = (3, 8),
                                        freq_file_path: str = None,
                                        exclude_20_channel_rats: bool = False,
                                        verbose: bool = True) -> Dict:
    """
    Run cross-rats analysis with TRUE hierarchical averaging:
    1. For each trial and electrode: compute windowed power
    2. For each electrode: average across trials  
    3. For each session: average across electrodes
    4. For each rat: average across sessions
    5. Across rats: compute SEM
    
    Parameters:
    -----------
    freq_file_path : str, optional
        Path to frequencies file (one frequency per line). If provided, loads frequencies 
        from file and filters by freq_range. If None, uses freq_range on session frequencies.
    """
    # Use original configuration if pkl_path not provided
    if pkl_path is None:
        pkl_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
    
    print("Starting TRUE hierarchical cross-rats analysis...")
    print("Procedure: Trials → Electrodes → Sessions → Rats → SEM")
    print(f"Electrode selection: {electrode_selection}")
    print(f"Time window: {time_window}")
    print(f"Frequency range: {freq_range}")
    
    # Load custom frequencies if file provided
    custom_frequencies = None
    if freq_file_path:
        custom_frequencies = get_frequencies(freq_range[0], freq_range[1], freq_file_path)
        print(f"📁 Using frequencies from file: {freq_file_path}")
        print(f"   Loaded {len(custom_frequencies)} frequencies in range {freq_range[0]}-{freq_range[1]} Hz")
        print(f"   Frequency range: {custom_frequencies[0]:.3f}-{custom_frequencies[-1]:.3f} Hz")
    
    # Load all data
    if verbose:
        print("Loading data...")
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Discover rat IDs
    rat_ids = discover_rat_ids(pkl_path, exclude_20_channel_rats, verbose, electrode_selection)
    
    if not rat_ids:
        raise ValueError("No rat IDs found")
    
    # Process each rat with true hierarchical averaging
    rat_results = {}
    failed_rats = []
    successful_rats = []
    
    for rat_id in rat_ids:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing rat {rat_id}")
            print(f"{'='*50}")
        
        rat_powers = process_single_rat_true_hierarchical(
            rat_id, electrode_selection, all_data, time_window, freq_range, custom_frequencies, verbose
        )
        
        if rat_powers is not None:
            rat_results[rat_id] = rat_powers
            successful_rats.append(rat_id)
            if verbose:
                print(f"✓ Rat {rat_id}: Success")
        else:
            failed_rats.append(rat_id)
            if verbose:
                print(f"✗ Rat {rat_id}: Failed")
    
    if not rat_results:
        raise ValueError("No rats processed successfully")
    
    # STEP 4: Aggregate results by NM size and compute SEM across rats
    return aggregate_hierarchical_results_by_nm_size(rat_results, electrode_selection, freq_range, time_window, verbose)


def apply_hierarchical_averaging_to_results(results: Dict, 
                                           time_window: Tuple[float, float],
                                           freq_range: Tuple[float, float],
                                           verbose: bool = True) -> Optional[Dict]:
    """
    Apply hierarchical averaging to results from process_single_rat_multi_session.
    
    Takes the spectrograms from the original results and applies:
    1. Trial averaging per electrode
    2. Electrode averaging per session (for each NM size)
    3. Session averaging per rat (for each NM size)
    
    Returns:
    --------
    Dict with keys 'nm_1', 'nm_2', 'nm_3' mapping to hierarchically averaged powers
    """
    try:
        if 'averaged_windows' not in results:
            if verbose:
                print("Warning: No averaged_windows in results")
            return None
        
        averaged_windows = results['averaged_windows']
        times = results['times']
        frequencies = results['frequencies']
        
        # Get time and frequency masks
        time_mask = (times >= time_window[0]) & (times <= time_window[1])
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        
        if np.sum(time_mask) == 0 or np.sum(freq_mask) == 0:
            if verbose:
                print("Warning: No data in specified time/freq window")
            return None
        
        hierarchical_powers = {}
        
        # Process each NM size
        for nm_size_key in ['nm_1', 'nm_2', 'nm_3']:
            if nm_size_key not in averaged_windows:
                if verbose:
                    print(f"Warning: {nm_size_key} not found in averaged_windows")
                hierarchical_powers[nm_size_key] = np.nan
                continue
            
            nm_data = averaged_windows[nm_size_key]
            
            if 'averaged_spectrogram' not in nm_data:
                if verbose:
                    print(f"Warning: averaged_spectrogram not found for {nm_size_key}")
                hierarchical_powers[nm_size_key] = np.nan
                continue
            
            # The averaged_spectrogram is already session-averaged across trials and electrodes
            # We just need to extract the power in our specific time/frequency window
            spectrogram = nm_data['averaged_spectrogram']  # Shape: (n_freqs, n_times)
            
            # Extract power in specified window
            windowed_power = spectrogram[freq_mask, :][:, time_mask]
            
            # Average across time and frequency
            hierarchical_power = np.mean(windowed_power)
            hierarchical_powers[nm_size_key] = hierarchical_power
            
            if verbose:
                print(f"  {nm_size_key}: hierarchical power = {hierarchical_power:.6f}")
        
        return hierarchical_powers
        
    except Exception as e:
        if verbose:
            print(f"Error in hierarchical averaging: {e}")
        return None


def aggregate_hierarchical_results_by_nm_size(rat_results: Dict,
                                             electrode_selection: Union[str, List[int]],
                                             freq_range: Tuple[float, float],
                                             time_window: Tuple[float, float],
                                             verbose: bool = True) -> Dict:
    """
    Aggregate hierarchical results across rats, organized by NM size.
    
    Returns results suitable for bar plots and ANOVA analysis.
    """
    if verbose:
        print(f"\n📊 Aggregating hierarchical results across {len(rat_results)} rats")
        print("=" * 60)
    
    # Organize data by NM size
    nm_size_data = {
        'nm_1': {'rat_powers': {}, 'values': []},
        'nm_2': {'rat_powers': {}, 'values': []},
        'nm_3': {'rat_powers': {}, 'values': []}
    }
    
    # Collect individual rat values for each NM size
    for rat_id, rat_result in rat_results.items():
        for nm_size_key in ['nm_1', 'nm_2', 'nm_3']:
            if nm_size_key in rat_result and not np.isnan(rat_result[nm_size_key]):
                power_value = rat_result[nm_size_key]
                nm_size_data[nm_size_key]['rat_powers'][rat_id] = power_value
                nm_size_data[nm_size_key]['values'].append(power_value)
    
    # Compute statistics for each NM size
    final_results = {
        'by_nm_size': {},
        'all_rat_values': {},  # For ANOVA
        'analysis_params': {
            'electrode_selection': electrode_selection,
            'time_window': time_window,
            'freq_range': freq_range,
            'total_rats': len(rat_results)
        }
    }
    
    for nm_size_key in ['nm_1', 'nm_2', 'nm_3']:
        values = nm_size_data[nm_size_key]['values']
        rat_powers = nm_size_data[nm_size_key]['rat_powers']
        
        if values:
            values_array = np.array(values)
            mean_power = np.mean(values_array)
            sem_power = stats.sem(values_array)
            std_power = np.std(values_array, ddof=1)
            n_rats = len(values)
            
            final_results['by_nm_size'][nm_size_key] = {
                'mean': mean_power,
                'sem': sem_power,
                'std': std_power,
                'n_rats': n_rats,
                'rat_powers': rat_powers
            }
            
            # Store individual values for ANOVA
            final_results['all_rat_values'][nm_size_key] = values
            
            if verbose:
                print(f"{nm_size_key.upper()}: Mean = {mean_power:.6f}, SEM = {sem_power:.6f}, N = {n_rats}")
        else:
            if verbose:
                print(f"{nm_size_key.upper()}: No valid data")
            final_results['by_nm_size'][nm_size_key] = {
                'mean': np.nan, 'sem': np.nan, 'std': np.nan, 'n_rats': 0, 'rat_powers': {}
            }
            final_results['all_rat_values'][nm_size_key] = []
    
    return final_results


def create_nm_size_bar_plot(results: Dict, save_path: str = None, show_plot: bool = True):
    """
    Create bar plot showing mean ± SEM for each NM size.
    """
    nm_sizes = ['nm_1', 'nm_2', 'nm_3'] 
    means = []
    sems = []
    labels = ['NM Size 1', 'NM Size 2', 'NM Size 3']
    
    for nm_size in nm_sizes:
        if nm_size in results['by_nm_size']:
            means.append(results['by_nm_size'][nm_size]['mean'])
            sems.append(results['by_nm_size'][nm_size]['sem'])
        else:
            means.append(np.nan)
            sems.append(np.nan)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                  color=['skyblue', 'lightcoral', 'lightgreen'],
                  edgecolor='black', linewidth=1)
    
    ax.set_xlabel('NM Size')
    ax.set_ylabel('Theta Power')
    ax.set_title('Mean Theta Power by NM Size\n(Hierarchical Averaging: Trials → Electrodes → Sessions → Rats)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    
    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(means, sems)):
        if not np.isnan(mean):
            ax.text(i, mean + sem + 0.01*max(means), f'{mean:.4f}', 
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def export_anova_data(results: Dict, save_path: str, verbose: bool = True):
    """
    Export data in format suitable for ANOVA analysis.
    
    Creates a CSV with columns: rat_id, nm_size, theta_power
    This gives you the 14 rats × 3 NM sizes = up to 42 rows for ANOVA.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    anova_data = []
    
    for nm_size_key in ['nm_1', 'nm_2', 'nm_3']:
        nm_size_num = nm_size_key.split('_')[1]  # Extract '1', '2', '3'
        
        if nm_size_key in results['by_nm_size']:
            rat_powers = results['by_nm_size'][nm_size_key]['rat_powers']
            
            for rat_id, power in rat_powers.items():
                anova_data.append({
                    'rat_id': rat_id,
                    'nm_size': nm_size_num,
                    'theta_power': power
                })
    
    # Create DataFrame and save
    anova_df = pd.DataFrame(anova_data)
    anova_df = anova_df.sort_values(['rat_id', 'nm_size'])
    anova_df.to_csv(save_path, index=False)
    
    if verbose:
        print(f"ANOVA data exported to: {save_path}")
        print(f"Data shape: {anova_df.shape[0]} rows × {anova_df.shape[1]} columns")
        print(f"Unique rats: {anova_df['rat_id'].nunique()}")
        print(f"NM sizes: {sorted(anova_df['nm_size'].unique())}")
    
    return anova_df


def save_results(results: Dict, save_path: str, verbose: bool = True):
    """Save analysis results to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save full results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary CSV
    summary_path = save_path.replace('.json', '_summary.csv')
    
    summary_data = []
    for rat_id, power in results['rat_powers'].items():
        summary_data.append({
            'rat_id': rat_id,
            'theta_power': power
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False)
    
    if verbose:
        print(f"Results saved to: {save_path}")
        print(f"Summary saved to: {summary_path}")


def main():
    """Main function for cross-rats hierarchical theta analysis."""
    parser = argparse.ArgumentParser(
        description='Cross-Rats NM Theta Analysis - Hierarchical Averaging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze frontal ROI with default time/frequency windows
  python nm_theta_cross_rats_multiple_electrodes.py data.pkl frontal
  
  # Analyze specific electrodes with custom time window
  python nm_theta_cross_rats_multiple_electrodes.py data.pkl "1,2,3,4" --time_window -0.5 0.0
  
  # Analyze with custom frequency range
  python nm_theta_cross_rats_multiple_electrodes.py data.pkl frontal --freq_range 4 12
"""
    )
    
    parser.add_argument('--pkl_path', default=DataConfig.MAIN_EEG_DATA_FILE,
                       help=f'Path to main EEG data file (default: {DataConfig.MAIN_EEG_DATA_FILE})')
    parser.add_argument('electrode_selection', 
                       help='Either ROI name (e.g., "frontal") or comma-separated electrode numbers (e.g., "1,2,3,4")')
    parser.add_argument('--time_window', nargs=2, type=float, default=[-0.2, 0.0],
                       metavar=('START', 'END'),
                       help='Time window in seconds (default: -0.2 0.0)')
    parser.add_argument('--freq_range', nargs=2, type=float, default=[3, 8],
                       metavar=('MIN', 'MAX'),
                       help='Frequency range in Hz (default: 3 8)')
    parser.add_argument('--exclude_20_channel_rats', action='store_true',
                       help='Exclude rats with 20-channel setup')
    parser.add_argument('--save_path', default=None,
                       help='Path to save results (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Parse electrode selection
    if args.electrode_selection.replace(',', '').replace(' ', '').isdigit():
        # Comma-separated electrode numbers
        electrode_selection = [int(x.strip()) for x in args.electrode_selection.split(',')]
    else:
        # ROI name
        electrode_selection = args.electrode_selection
    
    # Generate save path if not provided
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        electrode_str = str(electrode_selection).replace(' ', '').replace('[', '').replace(']', '').replace(',', '_')
        save_dir = os.path.join('results', 'hierarchical_analysis')
        args.save_path = os.path.join(save_dir, f'hierarchical_analysis_{electrode_str}_{timestamp}.json')
    
    try:
        # Run analysis
        results = run_cross_rats_hierarchical_analysis(
            pkl_path=args.pkl_path,
            electrode_selection=electrode_selection,
            time_window=tuple(args.time_window),
            freq_range=tuple(args.freq_range),
            exclude_20_channel_rats=args.exclude_20_channel_rats,
            verbose=args.verbose
        )
        
        # Save results
        save_results(results, args.save_path, args.verbose)
        
        print(f"\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Check if we're running in IDE mode (with direct function calls)
    # or command line mode
    import sys
    import time
    
    # If there are command line arguments, use command line mode
    if len(sys.argv) > 1:
        # For command line usage, call main()
        sys.exit(main())
    else:
        # ============================================
        # IDE MODE: MODIFY PARAMETERS HERE
        # ============================================
        
        # For IDE usage, run the analysis directly with explicit parameters
        from config import DataConfig
        data_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
        print(f"🔍 Debug - Data file path: {data_path}")
        print(f"🔍 Debug - File exists: {os.path.exists(data_path)}")
        
        # Run the hierarchical analysis with explicit parameters
        results = run_cross_rats_hierarchical_analysis(
            electrode_selection=[8,9,6,11],        # Change electrode selection here: ROI name or [1,2,3,4]
            pkl_path=data_path,                    # Keep explicit path from config
            time_window=(-0.2, 0.0),              # Time window in seconds
            freq_range=(3, 8),                    # Frequency range in Hz (theta band)
            freq_file_path="data/config/frequencies.txt",  # Path to frequencies file (optional)
            exclude_20_channel_rats=False,        # Set to True to exclude rat 9442
            verbose=True                          # Detailed output
        )
        
        # Save results and create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        electrode_str = str([8,9,6,11]).replace(' ', '').replace('[', '').replace(']', '').replace(',', '_')
        save_dir = os.path.join('results', 'hierarchical_analysis')
        save_path = os.path.join(save_dir, f'hierarchical_analysis_{electrode_str}_{timestamp}.json')
        
        # Save full results
        save_results(results, save_path, verbose=True)
        
        # Create and save bar plot
        plot_path = os.path.join(save_dir, f'nm_size_barplot_{electrode_str}_{timestamp}.png')
        create_nm_size_bar_plot(results, save_path=plot_path, show_plot=True)
        
        # Print individual rat values for ANOVA
        print(f"\n📋 INDIVIDUAL RAT VALUES FOR ANOVA:")
        print("=" * 50)
        for nm_size in ['nm_1', 'nm_2', 'nm_3']:
            if nm_size in results['all_rat_values']:
                values = results['all_rat_values'][nm_size]
                if values:
                    print(f"{nm_size.upper()} ({len(values)} rats): {[f'{v:.6f}' for v in values]}")
                else:
                    print(f"{nm_size.upper()}: No data")
        
        # Export ANOVA-ready CSV
        anova_path = os.path.join(save_dir, f'anova_data_{electrode_str}_{timestamp}.csv')
        export_anova_data(results, anova_path)
        
        print(f"\n✅ Analysis completed!")
        print(f"📊 Results saved to: {save_path}")
        print(f"📈 Bar plot saved to: {plot_path}")
        print(f"📋 ANOVA data saved to: {anova_path}")