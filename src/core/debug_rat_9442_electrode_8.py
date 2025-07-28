#!/usr/bin/env python3
"""
Debug script to investigate why ROI 8 (electrode 8) might not be found for rat 9442.

This script tests the electrode mapping logic and identifies potential bugs in 
the get_channel_indices_from_electrodes function for rat 9442 specifically.
"""

import pandas as pd
import numpy as np
from electrode_utils import load_electrode_mappings, get_channel_indices_from_electrodes, get_channels

def debug_rat_9442_electrode_8():
    """Debug electrode 8 mapping for rat 9442."""
    
    print("=" * 80)
    print("DEBUGGING RAT 9442 ELECTRODE 8 MAPPING")
    print("=" * 80)
    
    # Load electrode mappings
    try:
        mapping_df = load_electrode_mappings()
        print(f"✓ Successfully loaded electrode mappings")
        print(f"Available rat IDs: {list(mapping_df.index)}")
    except Exception as e:
        print(f"❌ Error loading electrode mappings: {e}")
        return
    
    # Check if rat 9442 exists in the mapping
    rat_id = 9442
    if rat_id not in mapping_df.index:
        print(f"❌ Rat {rat_id} not found in mapping DataFrame")
        return
    
    print(f"\n🔍 ANALYSIS FOR RAT {rat_id}")
    print("-" * 50)
    
    # Get the raw mapping for rat 9442
    rat_9442_row = mapping_df.loc[rat_id]
    print(f"Raw mapping row for rat {rat_id}:")
    print(rat_9442_row.to_dict())
    
    # Get non-NaN values (actual electrodes)
    non_nan_values = rat_9442_row.values[~pd.isna(rat_9442_row.values)]
    non_nan_values_int = non_nan_values.astype(int)
    print(f"\nNon-NaN electrode values: {non_nan_values_int.tolist()}")
    print(f"Number of active channels: {len(non_nan_values_int)}")
    
    # Check if electrode 8 is in the mapping
    electrode_8_present = 8 in non_nan_values_int
    print(f"\nElectrode 8 present in mapping: {electrode_8_present}")
    
    if electrode_8_present:
        # Find the position of electrode 8
        electrode_8_positions = [i for i, val in enumerate(non_nan_values_int) if val == 8]
        print(f"Electrode 8 found at position(s): {electrode_8_positions}")
        
        # Check the CSV column where electrode 8 is stored
        full_row_with_nans = rat_9442_row.values
        electrode_8_csv_positions = [i for i, val in enumerate(full_row_with_nans) if not pd.isna(val) and int(val) == 8]
        print(f"Electrode 8 found at CSV column(s): {electrode_8_csv_positions}")
        
        if electrode_8_csv_positions:
            csv_pos = electrode_8_csv_positions[0]
            print(f"Electrode 8 is at ch_{csv_pos} (0-based index {csv_pos})")
    
    print(f"\n🧪 TESTING get_channel_indices_from_electrodes FUNCTION")
    print("-" * 60)
    
    # Test the get_channel_indices_from_electrodes function
    try:
        # Test with electrode 8 specifically
        electrode_list = [8]
        result_indices = get_channel_indices_from_electrodes(rat_id, electrode_list, mapping_df)
        print(f"✓ get_channel_indices_from_electrodes({rat_id}, {electrode_list}) = {result_indices}")
        
        # Verify the result
        if result_indices:
            for idx in result_indices:
                actual_electrode = non_nan_values_int[idx] if idx < len(non_nan_values_int) else "OUT_OF_RANGE"
                print(f"  Channel index {idx} maps to electrode {actual_electrode}")
        else:
            print(f"  ❌ No channel indices returned for electrode 8!")
            
    except Exception as e:
        print(f"❌ Error in get_channel_indices_from_electrodes: {e}")
    
    print(f"\n🧪 TESTING get_channels FUNCTION")
    print("-" * 40)
    
    # Test the get_channels function with electrode 8
    try:
        electrode_list = [8]
        result_channels = get_channels(rat_id, electrode_list, mapping_df)
        print(f"✓ get_channels({rat_id}, {electrode_list}) = {result_channels}")
        
    except Exception as e:
        print(f"❌ Error in get_channels: {e}")
    
    print(f"\n🔍 COMPARISON WITH OTHER RATS")
    print("-" * 40)
    
    # Compare with another rat (9151) to see the difference
    other_rat = 9151
    if other_rat in mapping_df.index:
        print(f"Testing rat {other_rat} for comparison:")
        try:
            result_other = get_channel_indices_from_electrodes(other_rat, [8], mapping_df)
            print(f"✓ get_channel_indices_from_electrodes({other_rat}, [8]) = {result_other}")
            
            other_row = mapping_df.loc[other_rat].values[~pd.isna(mapping_df.loc[other_rat].values)].astype(int)
            print(f"  Rat {other_rat} non-NaN electrodes: {other_row.tolist()}")
            print(f"  Number of channels: {len(other_row)}")
            
        except Exception as e:
            print(f"❌ Error testing rat {other_rat}: {e}")
    
    print(f"\n🐛 POTENTIAL BUG ANALYSIS")
    print("-" * 30)
    
    # Analyze potential issues
    print("Potential issues identified:")
    
    # Issue 1: Channel count mismatch
    if len(non_nan_values_int) == 20:
        print("1. ✓ Rat 9442 has 20 channels as expected")
    else:
        print(f"1. ❌ Rat 9442 has {len(non_nan_values_int)} channels, expected 20")
    
    # Issue 2: Electrode 8 position
    if electrode_8_present:
        electrode_8_pos = electrode_8_positions[0]
        if electrode_8_pos == 18:  # Should be at position 18 (ch_18)
            print("2. ✓ Electrode 8 is at the expected position (18)")
        else:
            print(f"2. ⚠️  Electrode 8 is at position {electrode_8_pos}, expected 18")
    else:
        print("2. ❌ Electrode 8 not found in rat 9442 mapping!")
    
    # Issue 3: Check against the hardcoded list
    RAT_9442_20_CHANNEL_ELECTRODES = [10, 11, 12, 13, 14, 15, 16, 19, 1, 24, 25, 29, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"\nHardcoded RAT_9442_20_CHANNEL_ELECTRODES: {RAT_9442_20_CHANNEL_ELECTRODES}")
    csv_electrodes = non_nan_values_int.tolist()
    print(f"CSV electrode mapping: {csv_electrodes}")
    
    if csv_electrodes == RAT_9442_20_CHANNEL_ELECTRODES:
        print("3. ✓ CSV mapping matches hardcoded list")
    else:
        print("3. ⚠️  CSV mapping differs from hardcoded list")
        print(f"   Differences: CSV has {set(csv_electrodes) - set(RAT_9442_20_CHANNEL_ELECTRODES)}")
        print(f"   Missing: {set(RAT_9442_20_CHANNEL_ELECTRODES) - set(csv_electrodes)}")

if __name__ == "__main__":
    debug_rat_9442_electrode_8()