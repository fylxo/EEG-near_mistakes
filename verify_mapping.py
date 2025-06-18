#!/usr/bin/env python3
"""
Verify the exact electrode mapping that the user is seeing
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_user_mapping():
    """Check exactly what electrode mapping the user has"""
    
    # Load CSV directly
    df = pd.read_csv('consistent_electrode_mappings.csv', index_col='rat_id')
    
    # Check a specific rat
    rat_id = df.index[0]  # First rat
    print(f"Checking rat {rat_id}:")
    
    # Show the mapping for frontal channels
    frontal_electrodes = [2, 29, 31, 32]
    
    print("\nFrontal ROI electrode locations:")
    for electrode in frontal_electrodes:
        for col_idx, col_name in enumerate(df.columns):
            if df.loc[rat_id, col_name] == electrode:
                print(f"  Electrode {electrode} is in column '{col_name}' (index {col_idx})")
                break
    
    # Check what electrodes are at indices 20, 21, 23, 24
    print(f"\nWhat's actually at the indices the algorithm returns:")
    indices = [20, 21, 23, 24]
    row = df.loc[rat_id]
    for idx in indices:
        col_name = df.columns[idx]
        electrode = row.iloc[idx]
        print(f"  Index {idx} (column '{col_name}') contains electrode {electrode}")
    
    # What the user expected
    print(f"\nWhat would be at indices 21, 22, 24, 25 (user's expectation):")
    expected_indices = [21, 22, 24, 25]
    for idx in expected_indices:
        if idx < len(df.columns):
            col_name = df.columns[idx]
            electrode = row.iloc[idx]
            print(f"  Index {idx} (column '{col_name}') contains electrode {electrode}")
    
    # Check for off-by-one error pattern
    print(f"\nChecking if there's a pattern suggesting an off-by-one error:")
    print("Current mapping gives electrodes:", [row.iloc[i] for i in [20, 21, 23, 24]])
    print("Expected electrodes for frontal:", frontal_electrodes)
    print("User expected indices would give:", [row.iloc[i] for i in [21, 22, 24, 25] if i < len(row)])

if __name__ == "__main__":
    verify_user_mapping()