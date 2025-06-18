#!/usr/bin/env python3
"""
Debug script to check electrode mapping issues
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.electrode_utils import get_channels, load_electrode_mappings, ROI_MAP

def debug_electrode_mapping():
    """Debug the electrode mapping to find the indexing issue"""
    print("Debugging electrode mapping...")
    
    # Load the CSV directly
    mapping_df = load_electrode_mappings()
    print(f"Loaded DataFrame shape: {mapping_df.shape}")
    print(f"Columns: {list(mapping_df.columns)}")
    
    # Look at first rat
    first_rat = mapping_df.index[0]
    print(f"\nExamining rat {first_rat}:")
    
    # Get the raw values
    row_values = mapping_df.loc[first_rat].values
    row_values_clean = row_values[~pd.isna(row_values)].astype(int)
    print(f"All electrode values: {row_values_clean}")
    print(f"Length: {len(row_values_clean)}")
    
    # Print column by column mapping
    print(f"\nColumn-by-column mapping for rat {first_rat}:")
    for i, (col_name, value) in enumerate(zip(mapping_df.columns, row_values)):
        if not pd.isna(value):
            print(f"  {col_name} (pandas index {i}) → electrode {int(value)}")
    
    # Test frontal ROI specifically
    print(f"\nTesting frontal ROI: {ROI_MAP['frontal']}")
    
    # Find where each frontal electrode is located
    for electrode_num in ROI_MAP['frontal']:
        for i, value in enumerate(row_values):
            if not pd.isna(value) and int(value) == electrode_num:
                col_name = mapping_df.columns[i]
                print(f"  Electrode {electrode_num} found in {col_name} (pandas index {i})")
    
    # Test get_channels function
    print(f"\nTesting get_channels function:")
    try:
        frontal_indices = get_channels(first_rat, 'frontal', mapping_df)
        print(f"get_channels result for 'frontal': {frontal_indices}")
        
        # Check what electrodes these indices actually correspond to
        print(f"These indices correspond to electrodes:")
        for idx in frontal_indices:
            if idx < len(row_values_clean):
                electrode = row_values_clean[idx] 
                col_name = mapping_df.columns[idx]
                print(f"  Index {idx} ({col_name}) → electrode {electrode}")
            else:
                print(f"  Index {idx} is out of range!")
                
    except Exception as e:
        print(f"Error in get_channels: {e}")
        import traceback
        traceback.print_exc()
    
    # Test custom electrode list
    print(f"\nTesting custom electrode list [2, 29, 31, 32]:")
    try:
        custom_indices = get_channels(first_rat, [2, 29, 31, 32], mapping_df)
        print(f"get_channels result for [2, 29, 31, 32]: {custom_indices}")
        
        print(f"These indices correspond to electrodes:")
        for idx in custom_indices:
            if idx < len(row_values_clean):
                electrode = row_values_clean[idx]
                col_name = mapping_df.columns[idx]
                print(f"  Index {idx} ({col_name}) → electrode {electrode}")
                
    except Exception as e:
        print(f"Error in custom electrode test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_electrode_mapping()