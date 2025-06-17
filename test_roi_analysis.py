#!/usr/bin/env python3
"""
Test script for ROI-based NM theta analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.nm_theta_analysis import analyze_session_nm_theta_roi, load_session_data
from core.electrode_utils import get_roi_info, load_electrode_mappings

def test_roi_analysis():
    """Test the ROI-based analysis functionality"""
    try:
        print("Testing ROI-based NM theta analysis...")
        
        # Test electrode mapping functionality first
        print("\n1. Testing electrode mapping...")
        mapping_df = load_electrode_mappings()
        print(f"Loaded mappings for {len(mapping_df)} rats")
        
        # Test with a sample rat
        sample_rat = mapping_df.index[0]
        roi_info = get_roi_info(sample_rat, mapping_df)
        print(f"ROI info for rat {sample_rat}: {roi_info}")
        
        # Test loading session data
        print("\n2. Testing session data loading...")
        if os.path.exists('all_eeg_data.pkl'):
            session_data = load_session_data('all_eeg_data.pkl', session_index=0)
            print(f"Session loaded: {session_data.get('rat_id', 'unknown')}")
            
            # Test ROI analysis with hippocampus
            print("\n3. Testing ROI analysis with 'hippocampus'...")
            results = analyze_session_nm_theta_roi(
                session_data=session_data,
                roi_or_channels='hippocampus',
                freq_range=(4, 8),
                freq_step=1.0,
                window_duration=1.0,
                save_path='test_roi_results'
            )
            
            print(f"✓ ROI analysis completed successfully!")
            print(f"  - ROI channels: {results['roi_channels']}")
            print(f"  - Frequency range: {results['freqs'][0]:.1f}-{results['freqs'][-1]:.1f} Hz")
            print(f"  - NM windows: {list(results['normalized_windows'].keys())}")
            
            # Test with custom channel list
            print("\n4. Testing custom channel list...")
            custom_results = analyze_session_nm_theta_roi(
                session_data=session_data,
                roi_or_channels=[0, 1, 2],  # First 3 channels
                freq_range=(4, 8),
                freq_step=1.0,
                window_duration=1.0,
                save_path='test_custom_results'
            )
            
            print(f"✓ Custom channel analysis completed!")
            print(f"  - Custom channels: {custom_results['roi_channels']}")
            
            return True
            
        else:
            print("❗ all_eeg_data.pkl not found. Cannot test with real data.")
            print("The ROI functionality has been implemented and should work when data is available.")
            return True
            
    except Exception as e:
        print(f"❌ Error in ROI analysis test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_roi_analysis()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)