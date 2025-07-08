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

# Import memory monitoring utilities
from memory_monitor import monitor_memory_usage, force_garbage_collection, check_memory_requirements, log_memory_warning

# Import session resilience system
from session_resilience import SessionProcessor
from resilient_analysis_wrapper import run_analysis_with_resilience

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

# Constants for rat 9442 special handling
RAT_9442_32_CHANNEL_SESSIONS = ['070419', '080419', '090419', '190419']
RAT_9442_20_CHANNEL_ELECTRODES = [10, 11, 12, 13, 14, 15, 16, 19, 1, 24, 25, 29, 2, 3, 4, 5, 6, 7, 8, 9]


def load_electrode_mappings(mapping_file: str = None) -> pd.DataFrame:
    """
    Load electrode mappings from CSV file.
    
    Parameters:
    -----------
    mapping_file : str, optional
        Path to the electrode mappings CSV file (default: from DataConfig)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rat_id as index and electrode mappings
    """
    if mapping_file is None:
        mapping_file = DataConfig.get_data_file_path(DataConfig.ELECTRODE_MAPPINGS_FILE)
    
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


def cleanup_session_folders(rat_save_path: str, verbose: bool = True):
    """
    Clean up intermediate session folders after multi-session averaging.
    
    Deletes individual session folders (session_XX) but keeps the main 
    multi_session_results.pkl file to save disk space.
    
    Parameters:
    -----------
    rat_save_path : str
        Path to the rat's results directory
    verbose : bool
        Whether to print cleanup progress (default: True)
    """
    if not os.path.exists(rat_save_path):
        if verbose:
            print(f"⚠️  Cleanup skipped - directory does not exist: {rat_save_path}")
        return
    
    session_folders = []
    total_size_deleted = 0
    
    # Find all session folders (session_XX pattern)
    for item in os.listdir(rat_save_path):
        item_path = os.path.join(rat_save_path, item)
        if os.path.isdir(item_path) and item.startswith('session_'):
            session_folders.append(item_path)
    
    if not session_folders:
        if verbose:
            print(f"  🧹 No session folders found to cleanup in {rat_save_path}")
        return
    
    if verbose:
        print(f"  🧹 Cleaning up {len(session_folders)} session folders...")
    
    # Delete session folders
    for folder_path in session_folders:
        try:
            # Calculate folder size before deletion
            folder_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        folder_size += os.path.getsize(file_path)
                    except (OSError, IOError):
                        pass
            
            # Delete the folder
            import shutil
            shutil.rmtree(folder_path)
            total_size_deleted += folder_size
            
            if verbose:
                folder_size_mb = folder_size / (1024 * 1024)
                print(f"    ✓ Deleted {os.path.basename(folder_path)} ({folder_size_mb:.1f} MB)")
                
        except Exception as e:
            if verbose:
                print(f"    ❌ Error deleting {folder_path}: {e}")
    
    if verbose:
        total_size_mb = total_size_deleted / (1024 * 1024)
        print(f"  ✓ Cleanup complete - freed {total_size_mb:.1f} MB of disk space")


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
    cleanup_intermediate_files: bool = True,
    session_resilience: bool = True,
    max_session_retries: int = 3,
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
    cleanup_intermediate_files : bool
        Whether to delete individual session folders after averaging (default: True)
    session_resilience : bool
        Whether to enable session-level retry logic for memory failures (default: True)
    max_session_retries : int
        Maximum number of retry attempts per failed session (default: 3)
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
        
        # Memory requirements estimation
        n_channels = len(roi_or_channels) if isinstance(roi_or_channels, list) else 3  # Estimate
        estimates = check_memory_requirements(
            n_freqs=n_freqs,
            n_time_points=int(window_duration * 1000 * 60),  # Rough estimate for session length
            n_channels=n_channels
        )
        log_memory_warning(f"rat {rat_id} analysis", estimates, verbose=verbose)
    
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
        
        # Setup memory monitoring
        memory_log_file = os.path.join(rat_save_path, f'memory_usage_rat_{rat_id}.log') if verbose else None
        
        with monitor_memory_usage(f"rat {rat_id} multi-session analysis", 
                                 log_file=memory_log_file, 
                                 verbose=verbose) as monitor:
            
            if method == 'mne':
                # Use resilient analysis wrapper for MNE-based analysis
                if session_resilience:
                    results = run_analysis_with_resilience(
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
                        show_frequency_profiles=False,
                        session_resilience=session_resilience,
                        max_session_retries=max_session_retries,
                        verbose=verbose
                    )
                else:
                    # Use original analysis without resilience
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
        
        # Cleanup intermediate session files if requested
        if cleanup_intermediate_files:
            cleanup_session_folders(rat_save_path, verbose=verbose)
        
        # Force garbage collection with monitoring
        if verbose:
            force_garbage_collection(verbose=True)
        else:
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
            # 'individual_spectrograms': spectrograms,  # Commented out to save memory/storage
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
        # Note: individual_spectrograms removed to save memory/storage
    
    vmin, vmax = calculate_color_limits(all_spectrograms)
    # Hardcoded color limits for now
    vmin = -0.5
    vmax = 0.3
    
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
                          shading='auto', cmap=PARULA_COLORMAP, vmin=vmin, vmax=vmax)
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
        
        # Create individual rat summary plot (since individual spectrograms not stored)
        rat_ids = window_data['rat_ids']
        n_rats_to_show = min(6, len(rat_ids))  # Show up to 6 rats
        
        if n_rats_to_show > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for j in range(n_rats_to_show):
                ax = axes[j]
                rat_id = rat_ids[j]
                
                # Show the average spectrogram for each rat (same for all, but with individual stats)
                avg_spectrogram = window_data['avg_spectrogram']
                window_times = window_data['window_times']
                
                # Use log-frequency spacing for y-axis
                log_frequencies = np.log10(frequencies)
                im = ax.pcolormesh(window_times, log_frequencies, avg_spectrogram,
                                  shading='auto', cmap=PARULA_COLORMAP, vmin=vmin, vmax=vmax)
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_title(f'Rat {rat_id} (Average Pattern)\nEvents: {window_data["total_events_per_rat"][j]}, '
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
            
            plt.suptitle(f'Individual Rat Contributions - NM Size {nm_size}\n{roi_str}, {freq_str}\n(Note: Showing average pattern with individual rat statistics)', fontsize=14)
            
            # Apply custom spacing parameters for individual rats plot
            plt.subplots_adjust(left=0.052, bottom=0.07, right=0.55, top=0.924, wspace=0.206, hspace=0.656)
            
            individual_plot_file = os.path.join(save_path, f'individual_rats_nm_{nm_size}.png')
            plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
            if verbose:
                print(f"✓ Individual rats summary plot saved to: {individual_plot_file}")
            
            plt.show()
        else:
            if verbose:
                print("⚠️  Individual rat plots skipped (no rats to display)")


def save_checkpoint(checkpoint_data: Dict, checkpoint_file: str):
    """Save checkpoint data to file."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint data from file."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None

def run_cross_rats_analysis(
    roi: str,
    pkl_path: str = None,
    freq_min: float = None,
    freq_max: float = None,
    n_freqs: int = None,
    window_duration: float = None,
    n_cycles_factor: float = None,
    rat_ids: Optional[List[str]] = None,
    save_path: str = None,
    show_plots: bool = None,
    method: str = None,
    cleanup_intermediate_files: bool = None,
    resume_from_checkpoint: bool = None,
    session_resilience: bool = None,
    max_session_retries: int = None,
    verbose: bool = None
) -> Dict:
    """
    Run cross-rats NM theta analysis with direct parameter specification.
    
    Parameters:
    -----------
    roi : str
        ROI name ("frontal", "hippocampus") or channels ("1,2,3")
    pkl_path : str, optional
        Path to main EEG data file (default: from DataConfig)
    freq_min : float, optional
        Minimum frequency (Hz) (default: from AnalysisConfig)
    freq_max : float, optional
        Maximum frequency (Hz) (default: from AnalysisConfig)
    n_freqs : int, optional
        Number of frequencies (default: from AnalysisConfig)
    window_duration : float, optional
        Event window duration (s) (default: from AnalysisConfig)
    n_cycles_factor : float, optional
        Cycles factor for spectrograms (default: from AnalysisConfig)
    rat_ids : Optional[List[str]]
        Specific rat IDs to process, None for all rats
    save_path : str, optional
        Base directory for saving results (default: from DataConfig)
    show_plots : bool, optional
        Show plots during processing (default: from AnalysisConfig)
    method : str, optional
        Spectrogram calculation method (default: from AnalysisConfig)
    cleanup_intermediate_files : bool, optional
        Whether to delete individual session folders after averaging (default: True)
    resume_from_checkpoint : bool, optional
        Whether to resume from existing checkpoint if available (default: True)
    session_resilience : bool, optional
        Whether to enable session-level retry logic for memory failures (default: True)
    max_session_retries : int, optional
        Maximum number of retry attempts per failed session (default: 3)
    verbose : bool, optional
        Whether to print detailed progress information (default: from AnalysisConfig)
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    # Apply configuration defaults for None values
    if pkl_path is None:
        pkl_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
    if freq_min is None:
        freq_min = AnalysisConfig.THETA_MIN_FREQ
    if freq_max is None:
        freq_max = AnalysisConfig.THETA_MAX_FREQ
    if n_freqs is None:
        n_freqs = AnalysisConfig.N_FREQS_DEFAULT
    if window_duration is None:
        window_duration = AnalysisConfig.WINDOW_DURATION_DEFAULT
    if n_cycles_factor is None:
        n_cycles_factor = AnalysisConfig.N_CYCLES_FACTOR_DEFAULT
    if save_path is None:
        save_path = DataConfig.get_default_save_path('cross_rats')
    if show_plots is None:
        show_plots = AnalysisConfig.CROSS_RATS_SHOW_PLOTS
    if method is None:
        method = AnalysisConfig.SPECTROGRAM_METHOD_DEFAULT
    if cleanup_intermediate_files is None:
        cleanup_intermediate_files = True  # Default to cleanup enabled
    if resume_from_checkpoint is None:
        resume_from_checkpoint = True  # Default to resume enabled
    if session_resilience is None:
        session_resilience = True  # Default to resilience enabled
    if max_session_retries is None:
        max_session_retries = 3  # Default to 3 retries per session
    if verbose is None:
        verbose = AnalysisConfig.CROSS_RATS_VERBOSE
    
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
    
    # Setup checkpoint system
    checkpoint_file = os.path.join(save_path, 'analysis_checkpoint.pkl')
    
    # Try to load existing checkpoint
    checkpoint_data = None
    if resume_from_checkpoint:
        checkpoint_data = load_checkpoint(checkpoint_file)
    
    if checkpoint_data and verbose:
        completed_rats = checkpoint_data.get('completed_rats', [])
        print(f"📁 Resuming from checkpoint - {len(completed_rats)} rats already completed: {completed_rats}")
    
    # Process each rat individually
    rat_results = checkpoint_data.get('rat_results', {}) if checkpoint_data else {}
    failed_rats = checkpoint_data.get('failed_rats', []) if checkpoint_data else []
    error_details = checkpoint_data.get('error_details', {}) if checkpoint_data else {}
    successful_rats = checkpoint_data.get('successful_rats', []) if checkpoint_data else []
    completed_rats = checkpoint_data.get('completed_rats', []) if checkpoint_data else []
    
    for rat_id in rat_ids:
        # Skip if already completed
        if rat_id in completed_rats:
            if verbose:
                print(f"⏭️  Skipping rat {rat_id} (already completed)")
            continue
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
            cleanup_intermediate_files=cleanup_intermediate_files,
            session_resilience=session_resilience,
            max_session_retries=max_session_retries,
            verbose=verbose
        )
        
        rat_results[rat_id_str] = results
        
        # Track success/failure
        if results is None:
            failed_rats.append(rat_id_str)
            error_details[rat_id_str] = "Processing failed - see logs above for details"
        else:
            successful_rats.append(rat_id_str)
        
        # Add to completed list
        completed_rats.append(rat_id_str)
        
        # Save checkpoint after each rat
        if resume_from_checkpoint:
            checkpoint_data = {
                'rat_results': rat_results,
                'failed_rats': failed_rats,
                'error_details': error_details,
                'successful_rats': successful_rats,
                'completed_rats': completed_rats,
                'timestamp': datetime.now().isoformat(),
                'analysis_params': {
                    'roi': roi,
                    'freq_range': (freq_min, freq_max),
                    'n_freqs': n_freqs,
                    'method': method
                }
            }
            save_checkpoint(checkpoint_data, checkpoint_file)
            if verbose:
                print(f"💾 Checkpoint saved ({len(completed_rats)}/{len(rat_ids)} rats completed)")
        
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
        
        # Clean up checkpoint file on successful completion
        if resume_from_checkpoint and os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                if verbose:
                    print("🧹 Checkpoint file cleaned up")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not remove checkpoint file: {e}")
        
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
        if verbose and os.path.exists(checkpoint_file):
            print(f"💾 Checkpoint file preserved for debugging: {checkpoint_file}")
    
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
    parser.add_argument('--pkl_path', default=DataConfig.MAIN_EEG_DATA_FILE,
                       help=f'Path to main EEG data file (default: {DataConfig.MAIN_EEG_DATA_FILE})')
    parser.add_argument('--roi', required=True,
                       help='ROI name ("frontal", "hippocampus") or channels ("1,2,3")')
    
    # Analysis parameters
    parser.add_argument('--freq_min', type=float, default=AnalysisConfig.THETA_MIN_FREQ,
                       help=f'Minimum frequency (Hz) (default: {AnalysisConfig.THETA_MIN_FREQ})')
    parser.add_argument('--freq_max', type=float, default=AnalysisConfig.THETA_MAX_FREQ,
                       help=f'Maximum frequency (Hz) (default: {AnalysisConfig.THETA_MAX_FREQ})')
    parser.add_argument('--n_freqs', type=int, default=AnalysisConfig.N_FREQS_DEFAULT,
                       help=f'Number of frequencies (default: {AnalysisConfig.N_FREQS_DEFAULT})')
    parser.add_argument('--window_duration', type=float, default=AnalysisConfig.WINDOW_DURATION_DEFAULT,
                       help=f'Event window duration (s) (default: {AnalysisConfig.WINDOW_DURATION_DEFAULT})')
    parser.add_argument('--n_cycles_factor', type=float, default=AnalysisConfig.N_CYCLES_FACTOR_DEFAULT,
                       help=f'Cycles factor for spectrograms (default: {AnalysisConfig.N_CYCLES_FACTOR_DEFAULT})')
    
    # Method parameter
    parser.add_argument('--method', choices=['mne', 'cwt'], default=AnalysisConfig.SPECTROGRAM_METHOD_DEFAULT,
                       help=f'Spectrogram calculation method (default: {AnalysisConfig.SPECTROGRAM_METHOD_DEFAULT})')
    
    # Processing parameters
    parser.add_argument('--rat_ids', type=str, default=None,
                       help='Specific rat IDs to process (comma-separated), default: all rats')
    parser.add_argument('--save_path', default=DataConfig.CROSS_RATS_RESULTS_DIR,
                       help=f'Base directory for saving results (default: {DataConfig.CROSS_RATS_RESULTS_DIR})')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots during processing')
    parser.add_argument('--no_cleanup', action='store_true',
                       help='Keep individual session folders (do not delete after averaging)')
    parser.add_argument('--no_checkpoint', action='store_true',
                       help='Disable checkpoint/resume functionality')
    parser.add_argument('--no_resilience', action='store_true',
                       help='Disable session-level retry logic for memory failures')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry attempts per failed session (default: 3)')
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
        cleanup_intermediate_files=not args.no_cleanup,  # Cleanup enabled by default
        resume_from_checkpoint=not args.no_checkpoint,  # Checkpoint enabled by default
        session_resilience=not args.no_resilience,  # Resilience enabled by default
        max_session_retries=args.max_retries,
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
        
        # Or run single channel analysis using configuration defaults
        
        # Debug: Check data file path
        from config import DataConfig
        data_path = DataConfig.get_data_file_path(DataConfig.MAIN_EEG_DATA_FILE)
        print(f"🔍 Debug - Data file path: {data_path}")
        print(f"🔍 Debug - File exists: {os.path.exists(data_path)}")

    
        results = run_cross_rats_analysis(
            roi="1,2,3",                   # Change ROI here
            pkl_path=data_path,               # Keep explicit path
            freq_min=1.0,                     # Override config - test narrow theta
            freq_max=40.0,                     # Override config
            n_freqs=240,                       # Override config - faster analysis
            #rat_ids=["9442"],
            window_duration=2.0,              # Override config - longer window
            save_path="D:/nm_theta_results",  # Save to D: 
            cleanup_intermediate_files=True,  # Cleanup session folders (saves space)
            resume_from_checkpoint=False,     # Disable checkpoint/resume for testing
            session_resilience=True,          # Enable session-level retry logic
            max_session_retries=3,            # Try up to 3 times per failed session
            verbose=True                      # Override config
        )
        

