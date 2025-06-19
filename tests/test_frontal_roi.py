#!/usr/bin/env python3
"""
Test the frontal ROI analysis to see exactly what channels are being used
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))

from electrode_utils import get_channels, load_electrode_mappings, ROI_MAP

def test_frontal_roi():
    """Test exactly what happens with frontal ROI"""
    
    print("Testing frontal ROI analysis...")
    
    # Load mappings
    mapping_df = load_electrode_mappings()
    rat_id = mapping_df.index[0]
    
    print(f"Rat ID: {rat_id}")
    print(f"Frontal ROI definition: {ROI_MAP['frontal']}")
    
    # Get channels using the function
    frontal_indices = get_channels(rat_id, 'frontal', mapping_df)
    print(f"get_channels result: {frontal_indices}")
    
    # Show what electrodes these indices correspond to
    row = mapping_df.loc[rat_id]
    actual_electrodes = []
    print(f"\nDetailed mapping:")
    for i, idx in enumerate(frontal_indices):
        electrode = row.iloc[idx]
        col_name = mapping_df.columns[idx]
        actual_electrodes.append(electrode)
        print(f"  Index {idx} ({col_name}) → electrode {electrode}")
    
    print(f"\nSummary:")
    print(f"  Requested: {ROI_MAP['frontal']}")
    print(f"  Got indices: {frontal_indices}")
    print(f"  Got electrodes: {actual_electrodes}")
    print(f"  Match: {set(ROI_MAP['frontal']) == set(actual_electrodes)}")
    
    # Test what the user might be seeing
    print(f"\nIf indices were 1-based instead of 0-based:")
    user_interpretation = []
    for idx in frontal_indices:
        if idx + 1 < len(row):
            electrode = row.iloc[idx + 1]
            user_interpretation.append(electrode)
            print(f"  Index {idx} interpreted as {idx+1} → electrode {electrode}")
    print(f"  This would give electrodes: {user_interpretation}")
    
    # Check if there's a different electrode mapping file
    print(f"\nColumn names in CSV: {list(mapping_df.columns)}")
    print(f"Are these 0-indexed? Yes, ch_0 to ch_31")
    
    return True

if __name__ == "__main__":
    test_frontal_roi()