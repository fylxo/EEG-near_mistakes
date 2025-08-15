#!/usr/bin/env python3
"""
Fixed Electrode Mapping Functions for Rat 9442 Compatibility

This module provides corrected implementations of the buggy functions identified
in nm_theta_cross_rats.py. These fixes address:

1. Missing function definition (get_electrode_numbers_from_roi)
2. Type inconsistency in rat ID handling  
3. Incomplete ROI resolution
4. Incorrect DataFrame column access

Author: Generated for EEG analysis pipeline bug fixes
"""

from typing import Union, List, Dict, Optional
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from electrode_utils import get_channels, ROI_MAP, load_electrode_mappings
except ImportError as e:
    print(f"Warning: Could not import electrode_utils: {e}")
    print("Some functionality may be limited")
    # Updated ROI definitions as of latest specification
    ROI_MAP = {
        'mPFC': [8, 6, 9, 11],  # medial Prefrontal Cortex
        'motor': [7, 10, 5, 12, 3, 14],  # Motor cortex
        'somatomotor': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],  # Somatomotor cortex
        'visual': [20, 21, 22, 27, 28, 29, 17, 18, 19, 30, 31, 32],  # Visual cortex
        
        # Keep legacy names for backward compatibility
        'frontal': [8, 6, 9, 11],  # Alias for mPFC
        'ss': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],  # Alias for somatomotor (somatosensory)
    }


def get_electrode_numbers_from_roi_fixed(roi_or_channels: Union[str, List[int]], 
                                        mapping_df: pd.DataFrame,
                                        rat_id: str) -> List[int]:
    """
    Convert ROI specification to electrode numbers using existing electrode_utils.
    
    This is the FIXED version that properly resolves ROI specifications to electrode numbers.
    
    Parameters:
    -----------
    roi_or_channels : Union[str, List[int]]
        ROI specification - can be:
        - Comma-separated string like "8,16,24"
        - ROI name like "frontal" 
        - Single electrode as string like "8"
        - List of electrode numbers like [8, 16, 24]
    mapping_df : pd.DataFrame
        Electrode mappings DataFrame
    rat_id : str  
        Rat identifier (for compatibility, not used in current implementation)
    
    Returns:
    --------
    electrode_numbers : List[int]
        List of electrode numbers (1-32 range)
    """
    try:
        if isinstance(roi_or_channels, str):
            # Handle comma-separated electrode numbers like "8,16,24"
            if ',' in roi_or_channels:
                electrode_numbers = [int(ch.strip()) for ch in roi_or_channels.split(',')]
                
            # Handle ROI names like "frontal", "hippocampus"
            elif roi_or_channels in ROI_MAP:
                electrode_numbers = ROI_MAP[roi_or_channels].copy()  # Copy to avoid modifying original
                
            # Handle single electrode number as string like "8"
            elif roi_or_channels.isdigit():
                electrode_numbers = [int(roi_or_channels)]
                
            else:
                raise ValueError(f"Unknown ROI specification: '{roi_or_channels}'. "
                               f"Valid ROI names: {list(ROI_MAP.keys())}")
                
        elif isinstance(roi_or_channels, (list, tuple)):
            # Handle direct electrode number lists like [8, 16, 24]
            electrode_numbers = [int(x) for x in roi_or_channels]
            
        else:
            raise ValueError(f"ROI specification must be string or list, got {type(roi_or_channels)}")
        
        # Validate electrode numbers are in valid range
        for elec_num in electrode_numbers:
            if not (1 <= elec_num <= 32):
                raise ValueError(f"Electrode number {elec_num} out of valid range (1-32)")
        
        return electrode_numbers
        
    except Exception as e:
        print(f"Error resolving ROI specification '{roi_or_channels}': {e}")
        print(f"Valid ROI names: {list(ROI_MAP.keys())}")
        print(f"Valid formats: '8,16,24', 'frontal', [8,16,24], '8'")
        raise


def check_rat_9442_compatibility_fixed(roi_or_channels: Union[str, List[int]], 
                                      mapping_df: pd.DataFrame,
                                      verbose: bool = True) -> bool:
    """
    Check if the requested ROI/channels are compatible with rat 9442's electrode mapping.
    
    This is the FIXED version that properly handles:
    - Type consistency (always uses string rat IDs)
    - Proper DataFrame column access
    - Complete ROI resolution
    - Robust error handling
    
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
    compatible : bool
        True if compatible, False otherwise
    """
    try:
        # FIXED: Get electrode numbers using the corrected function
        electrode_numbers = get_electrode_numbers_from_roi_fixed(roi_or_channels, mapping_df, '9442')
        
        if verbose:
            print(f"  Checking compatibility for electrodes: {electrode_numbers}")
        
        # FIXED: Use consistent string representation for rat ID
        rat_id_str = '9442'
        rat_id_int = 9442
        
        # Get rat 9442 mapping - handle both string and int keys robustly
        rat_9442_mapping = None
        if rat_id_str in mapping_df.index:
            rat_9442_mapping = mapping_df.loc[rat_id_str]
        elif rat_id_int in mapping_df.index:
            rat_9442_mapping = mapping_df.loc[rat_id_int]
        else:
            if verbose:
                print(f"  ❌ Rat 9442 not found in electrode mappings")
                print(f"     Available rat IDs: {list(mapping_df.index)}")
            return False
        
        # FIXED: Properly extract available electrodes from DataFrame row
        available_electrodes = []
        
        # The rat_9442_mapping is a pandas Series representing one row
        # We need to iterate over its values, not its index
        for col_name, electrode_value in rat_9442_mapping.items():
            # Only process columns that look like channel mappings
            if pd.notna(electrode_value) and electrode_value != 'None':
                try:
                    electrode_num = int(float(electrode_value))  # Handle both int and float
                    if 1 <= electrode_num <= 32:  # Validate range
                        available_electrodes.append(electrode_num)
                except (ValueError, TypeError):
                    continue  # Skip invalid values
        
        # Remove duplicates and sort for clean display
        available_electrodes = sorted(list(set(available_electrodes)))
        
        # Check compatibility
        available_set = set(available_electrodes)
        requested_set = set(electrode_numbers)
        missing_electrodes = requested_set - available_set
        
        if verbose:
            print(f"  Available electrodes for rat 9442: {available_electrodes}")
            print(f"  Requested electrodes: {sorted(electrode_numbers)}")
            
            if missing_electrodes:
                print(f"  ❌ Missing electrodes: {sorted(missing_electrodes)}")
                print(f"     Rat 9442 is INCOMPATIBLE with requested ROI")
            else:
                print(f"  ✅ All requested electrodes available")
                print(f"     Rat 9442 is COMPATIBLE with requested ROI")
        
        return len(missing_electrodes) == 0
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Error checking rat 9442 compatibility: {e}")
            print(f"     Assuming INCOMPATIBLE for safety")
        return False


def validate_electrode_mappings(mapping_df: pd.DataFrame, verbose: bool = True) -> Dict[str, bool]:
    """
    Validate electrode mappings for all rats with common test cases.
    
    Parameters:
    -----------
    mapping_df : pd.DataFrame
        Electrode mappings DataFrame
    verbose : bool
        Whether to print detailed validation info
        
    Returns:
    --------
    validation_results : Dict[str, bool]
        Dictionary mapping test case names to success/failure
    """
    if verbose:
        print("🔍 VALIDATING ELECTRODE MAPPING FIXES")
        print("=" * 60)
    
    test_cases = [
        ("Comma-separated", "8,16,24"),
        ("ROI name frontal", "frontal"),
        ("ROI name hippocampus", "hippocampus"), 
        ("List format", [8, 16, 24]),
        ("Single electrode", "8"),
        ("Single electrode int", 8)
    ]
    
    results = {}
    
    for test_name, test_roi in test_cases:
        if verbose:
            print(f"\n📊 Testing: {test_name} = {test_roi}")
        
        try:
            # Test ROI resolution
            electrode_numbers = get_electrode_numbers_from_roi_fixed(test_roi, mapping_df, '9442')
            if verbose:
                print(f"  ✅ ROI resolution: {test_roi} → {electrode_numbers}")
            
            # Test rat 9442 compatibility
            compatible = check_rat_9442_compatibility_fixed(test_roi, mapping_df, verbose=False)
            if verbose:
                print(f"  {'✅' if compatible else '❌'} Rat 9442 compatibility: {compatible}")
            
            results[test_name] = True
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            results[test_name] = False
    
    # Summary
    if verbose:
        print(f"\n📈 VALIDATION SUMMARY:")
        print("=" * 30)
        passed = sum(results.values())
        total = len(results)
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("✅ All validation tests PASSED - fixes are working correctly!")
        else:
            print("❌ Some validation tests FAILED - review errors above")
            for test_name, success in results.items():
                if not success:
                    print(f"  Failed: {test_name}")
    
    return results


def get_electrode_numbers_from_channels_fixed(rat_id: Union[str, int], 
                                             channel_indices: List[int], 
                                             mapping_df: pd.DataFrame) -> List[int]:
    """
    Convert channel indices back to electrode numbers for display purposes.
    
    This is the FIXED version that ensures consistent return types.
    
    Parameters:
    -----------
    rat_id : Union[str, int]
        Rat identifier
    channel_indices : List[int]
        0-based channel indices
    mapping_df : pd.DataFrame
        Electrode mapping dataframe
    
    Returns:
    --------
    electrode_numbers : List[int]  
        Electrode numbers (1-32) corresponding to the channel indices
    """
    # Handle rat_id type conversion (same logic as in electrode_utils.py)
    str_id = str(rat_id)
    int_id = int(rat_id) if isinstance(rat_id, (str, int)) and str(rat_id).isdigit() else None

    # Find the rat in the mapping DataFrame
    row = None
    if rat_id in mapping_df.index:
        row = mapping_df.loc[rat_id].values
    elif str_id in mapping_df.index:
        row = mapping_df.loc[str_id].values
    elif int_id is not None and int_id in mapping_df.index:
        row = mapping_df.loc[int_id].values
    else:
        raise ValueError(f"Rat ID {rat_id} not found in mapping DataFrame")

    # Remove NaN values and convert to integers
    row = row[~pd.isna(row)].astype(int).tolist()
    
    # FIXED: Get electrode numbers for the given channel indices, ensuring consistent types
    electrode_numbers = []
    for ch_idx in channel_indices:
        if ch_idx < len(row):
            electrode_numbers.append(int(row[ch_idx]))  # Ensure int type
        else:
            # FIXED: Don't mix types - skip invalid channels and warn
            print(f"Warning: Channel index {ch_idx} out of range for rat {rat_id} (max: {len(row)-1}), skipping")
            continue  # Skip invalid channels rather than adding strings
    
    return electrode_numbers  # Now guaranteed to be List[int]


def add_minimum_events_validation(baseline_stats: Dict, min_events: int = 3, verbose: bool = True) -> Dict:
    """
    Add validation for minimum number of events for stable statistics.
    
    Parameters:
    -----------
    baseline_stats : Dict
        Baseline statistics dictionary from compute_baseline_statistics
    min_events : int
        Minimum number of events required for stable statistics
    verbose : bool
        Whether to print warnings
        
    Returns:
    --------
    validated_stats : Dict
        Same as input but with validation warnings added
    """
    if verbose:
        print(f"\n⚠️  VALIDATING MINIMUM EVENTS (threshold: {min_events})")
        print("-" * 50)
    
    for nm_size, stats in baseline_stats.items():
        n_events = stats.get('n_events', 0)
        
        if n_events < min_events:
            warning_msg = (f"NM size {nm_size} has only {n_events} events (< {min_events}). "
                          f"Statistics may be unstable.")
            
            if verbose:
                print(f"⚠️  WARNING: {warning_msg}")
                print(f"   Consider combining with other NM sizes or excluding from analysis.")
            
            # Add warning to the stats dictionary
            if 'warnings' not in stats:
                stats['warnings'] = []
            stats['warnings'].append(warning_msg)
        
        elif verbose:
            print(f"✅ NM size {nm_size}: {n_events} events (sufficient)")
    
    return baseline_stats


# Test function for development and validation
def test_all_fixes():
    """
    Comprehensive test of all bug fixes.
    Run this to validate that all fixes are working correctly.
    """
    print("🧪 TESTING ALL ELECTRODE MAPPING FIXES")
    print("=" * 80)
    
    try:
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        print(f"✅ Loaded electrode mappings for {len(mapping_df)} rats")
        print(f"Available rats: {list(mapping_df.index)}")
        
        # Run comprehensive validation
        results = validate_electrode_mappings(mapping_df, verbose=True)
        
        # Test specific rat 9442 cases
        print(f"\n🐀 SPECIFIC RAT 9442 TESTS")
        print("=" * 40)
        
        rat_9442_test_cases = [
            "8,16,24",      # Common ROI
            "frontal",      # Should work if frontal electrodes available
            "6",            # Single electrode  
            [7, 15, 23],    # List format
        ]
        
        for test_roi in rat_9442_test_cases:
            print(f"\nTesting rat 9442 with ROI: {test_roi}")
            try:
                compatible = check_rat_9442_compatibility_fixed(test_roi, mapping_df, verbose=True)
                print(f"Result: {'✅ COMPATIBLE' if compatible else '❌ INCOMPATIBLE'}")
            except Exception as e:
                print(f"❌ ERROR: {e}")
        
        return results
        
    except FileNotFoundError:
        print("❌ ERROR: Electrode mappings CSV file not found")
        print("Please ensure data/config/consistent_electrode_mappings.csv exists in project root")
        return {}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {}


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_all_fixes()