#!/usr/bin/env python3
"""
Test Script for Updated ROI Definitions

This script tests the new ROI definitions:
- mPFC (8,6,9,11)
- motor (7,10,5,12,3,14) 
- somatomotor (4,13,2,15,23,24,1,16,25,26)
- visual (20,21,22,27,28,29,17,18,19,30,31,32)

Author: Generated for updated ROI definition testing
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_new_roi_definitions():
    """Test all new ROI definitions."""
    print("🗺️  TESTING NEW ROI DEFINITIONS")
    print("=" * 60)
    
    try:
        from electrode_mapping_fixes import get_electrode_numbers_from_roi_fixed
        from electrode_utils import load_electrode_mappings, ROI_MAP
        
        # Display updated ROI definitions
        print("📍 Updated ROI Definitions:")
        for roi_name, electrodes in ROI_MAP.items():
            print(f"  {roi_name:12}: {electrodes}")
        
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        print(f"\n✅ Loaded electrode mappings for {len(mapping_df)} rats")
        
        # Test each new ROI definition
        new_rois = ['mPFC', 'motor', 'somatomotor', 'visual']
        expected_electrodes = {
            'mPFC': [8, 6, 9, 11],
            'motor': [7, 10, 5, 12, 3, 14],
            'somatomotor': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],
            'visual': [20, 21, 22, 27, 28, 29, 17, 18, 19, 30, 31, 32]
        }
        
        print(f"\n🧪 Testing ROI Resolution:")
        test_results = {}
        
        for roi_name in new_rois:
            try:
                result = get_electrode_numbers_from_roi_fixed(roi_name, mapping_df, '9442')
                expected = expected_electrodes[roi_name]
                
                # Check if result matches expected (order doesn't matter)
                matches = set(result) == set(expected)
                
                print(f"  {roi_name:12}: {result}")
                print(f"  {'✅' if matches else '❌'} Expected: {expected}")
                print(f"  {'✅' if matches else '❌'} Match: {matches}")
                
                test_results[roi_name] = matches
                
            except Exception as e:
                print(f"  ❌ {roi_name}: Error - {e}")
                test_results[roi_name] = False
        
        # Test backward compatibility aliases
        print(f"\n🔄 Testing Backward Compatibility:")
        legacy_rois = ['frontal', 'ss']
        legacy_expected = {
            'frontal': [8, 6, 9, 11],  # Should match mPFC
            'ss': [4, 13, 2, 15, 23, 24, 1, 16, 25, 26]  # Should match somatomotor
        }
        
        for roi_name in legacy_rois:
            try:
                result = get_electrode_numbers_from_roi_fixed(roi_name, mapping_df, '9442')
                expected = legacy_expected[roi_name]
                
                matches = set(result) == set(expected)
                
                print(f"  {roi_name:12}: {result}")
                print(f"  {'✅' if matches else '❌'} Expected: {expected}")
                print(f"  {'✅' if matches else '❌'} Match: {matches}")
                
                test_results[f"{roi_name}_legacy"] = matches
                
            except Exception as e:
                print(f"  ❌ {roi_name}: Error - {e}")
                test_results[f"{roi_name}_legacy"] = False
        
        # Test rat 9442 compatibility with new ROIs
        print(f"\n🐀 Testing Rat 9442 Compatibility with New ROIs:")
        from electrode_mapping_fixes import check_rat_9442_compatibility_fixed
        
        for roi_name in new_rois:
            try:
                compatible = check_rat_9442_compatibility_fixed(roi_name, mapping_df, verbose=False)
                print(f"  {roi_name:12}: {'✅ Compatible' if compatible else '⚠️  Incompatible'}")
                
                # For completeness, let's also check what electrodes are missing if incompatible
                if not compatible:
                    try:
                        electrodes = get_electrode_numbers_from_roi_fixed(roi_name, mapping_df, '9442')
                        # This will show detailed compatibility info
                        check_rat_9442_compatibility_fixed(roi_name, mapping_df, verbose=True)
                    except Exception:
                        pass
                        
            except Exception as e:
                print(f"  ❌ {roi_name}: Error - {e}")
        
        # Summary
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"\n📊 ROI DEFINITION TEST RESULTS:")
        print(f"  Tests passed: {passed}/{total}")
        for test_name, success in test_results.items():
            print(f"    {test_name}: {'✅ PASSED' if success else '❌ FAILED'}")
        
        success = passed == total
        print(f"\n  Overall: {'✅ PASSED' if success else '❌ FAILED'}")
        
        if success:
            print("\n🎉 All ROI definitions are working correctly!")
            print("You can now use the new ROI names:")
            print("  - 'mPFC' for medial Prefrontal Cortex")
            print("  - 'motor' for Motor cortex") 
            print("  - 'somatomotor' for Somatomotor cortex")
            print("  - 'visual' for Visual cortex")
            print("  - 'frontal' and 'ss' still work for backward compatibility")
        else:
            print(f"\n⚠️  Some ROI definitions failed. Please check the errors above.")
        
        return success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False


def show_roi_electrode_counts():
    """Show electrode counts for each ROI for easy reference."""
    print(f"\n📈 ROI ELECTRODE COUNTS:")
    print("-" * 40)
    
    try:
        from electrode_utils import ROI_MAP
        
        for roi_name, electrodes in ROI_MAP.items():
            if roi_name not in ['frontal', 'ss']:  # Skip legacy aliases for count
                print(f"  {roi_name:12}: {len(electrodes):2d} electrodes - {electrodes}")
        
        print(f"\n  Legacy aliases:")
        for roi_name in ['frontal', 'ss']:
            if roi_name in ROI_MAP:
                electrodes = ROI_MAP[roi_name]
                print(f"  {roi_name:12}: {len(electrodes):2d} electrodes - {electrodes}")
                
    except Exception as e:
        print(f"Error displaying electrode counts: {e}")


def main():
    """Run ROI definition tests."""
    print("🗺️  NEW ROI DEFINITIONS TEST")
    print("=" * 50)
    print("Testing updated ROI definitions for EEG analysis")
    print("=" * 50)
    
    show_roi_electrode_counts()
    success = test_new_roi_definitions()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)