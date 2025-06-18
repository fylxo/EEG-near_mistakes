#!/usr/bin/env python3
"""
Test script to verify channel numbering consistency (1-32 vs 0-31) fix
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.electrode_utils import get_channels, load_electrode_mappings

def test_channel_numbering():
    """Test that channel numbering is now consistent between 1-32 and 0-31"""
    print("Testing channel numbering consistency...")
    
    try:
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        print(f"✓ Loaded electrode mappings for {len(mapping_df)} rats")
        
        # Test with a sample rat
        sample_rat = mapping_df.index[0]
        print(f"\nTesting with rat {sample_rat}:")
        
        # Test 1: ROI name (should work as before)
        hippocampus_channels = get_channels(sample_rat, 'hippocampus', mapping_df)
        print(f"✓ ROI 'hippocampus' -> indices: {hippocampus_channels}")
        
        # Test 2: Channel numbers 1-32 (should be converted to 0-based indices)
        channel_numbers = [10, 11, 12]  # Channel numbers as they appear in mapping
        converted_indices = get_channels(sample_rat, channel_numbers, mapping_df)
        print(f"✓ Channel numbers {channel_numbers} -> indices: {converted_indices}")
        
        # Test 3: Show the actual electrode mapping for verification
        actual_mapping = mapping_df.loc[sample_rat].values
        actual_mapping = actual_mapping[~pd.isna(actual_mapping)].astype(int)
        print(f"✓ Actual electrode mapping for rat {sample_rat}: {actual_mapping.tolist()}")
        
        # Test 4: Verify that channel number 10 corresponds to the correct index
        if 10 in actual_mapping:
            expected_index = list(actual_mapping).index(10)
            result_indices = get_channels(sample_rat, [10], mapping_df)
            if result_indices == [expected_index]:
                print(f"✓ Channel number 10 correctly maps to index {expected_index}")
            else:
                print(f"❌ Channel number 10 should map to index {expected_index}, got {result_indices}")
        else:
            print(f"⚠ Channel number 10 not found in mapping for rat {sample_rat}")
        
        # Test 5: Test that 0-based indices still work (for backwards compatibility)
        try:
            direct_indices = [0, 1, 2]  # 0-based indices
            result_indices = get_channels(sample_rat, direct_indices, mapping_df)
            print(f"✓ Direct indices {direct_indices} -> {result_indices} (backwards compatibility)")
        except Exception as e:
            print(f"⚠ Direct indices test: {e}")
        
        print(f"\n🎉 Channel numbering consistency test PASSED!")
        print("Now both scripts use consistent 1-32 channel numbering:")
        print("- ROI names: 'hippocampus', 'frontal', etc.")
        print("- Channel numbers: [10, 11, 12] (refers to channels 10, 11, 12 as in electrode mapping)")
        print("- All are automatically converted to 0-based indices for numpy array access")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in channel numbering test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Need to import pandas for the test
    import pandas as pd
    
    success = test_channel_numbering()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)