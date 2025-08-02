#!/usr/bin/env python3
"""
Test Script for Cross-Rats Pipeline Bug Fixes

This script comprehensively tests all the bug fixes implemented in:
- electrode_mapping_fixes.py
- baseline_normalization.py 
- nm_theta_single_basic.py
- nm_theta_cross_rats.py

Run this script to validate that all fixes are working correctly before using
the main analysis pipeline.

Author: Generated for EEG analysis pipeline bug fix validation
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_electrode_mapping_fixes():
    """Test all electrode mapping bug fixes."""
    print("🔧 TESTING ELECTRODE MAPPING FIXES")
    print("=" * 60)
    
    try:
        from electrode_mapping_fixes import (
            get_electrode_numbers_from_roi_fixed,
            check_rat_9442_compatibility_fixed,
            validate_electrode_mappings,
            test_all_fixes
        )
        from electrode_utils import load_electrode_mappings
        
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        print(f"✅ Loaded electrode mappings for {len(mapping_df)} rats")
        
        # Test ROI resolution
        print(f"\n📍 Testing ROI Resolution:")
        test_cases = [
            ("Comma-separated", "8,16,24"),
            ("ROI name frontal", "frontal"),
            ("Single electrode", "8"),
            ("List format", [8, 16, 24])
        ]
        
        roi_results = {}
        for test_name, test_roi in test_cases:
            try:
                result = get_electrode_numbers_from_roi_fixed(test_roi, mapping_df, '9442')
                print(f"  ✅ {test_name}: {test_roi} → {result}")
                roi_results[test_name] = True
            except Exception as e:
                print(f"  ❌ {test_name}: {test_roi} → Error: {e}")
                roi_results[test_name] = False
        
        # Test rat 9442 compatibility
        print(f"\n🐀 Testing Rat 9442 Compatibility:")
        rat_9442_results = {}
        for test_name, test_roi in test_cases:
            try:
                compatible = check_rat_9442_compatibility_fixed(test_roi, mapping_df, verbose=False)
                print(f"  {'✅' if compatible else '⚠️ '} {test_name}: {test_roi} → {'Compatible' if compatible else 'Incompatible'}")
                rat_9442_results[test_name] = True
            except Exception as e:
                print(f"  ❌ {test_name}: {test_roi} → Error: {e}")
                rat_9442_results[test_name] = False
        
        # Summary
        roi_passed = sum(roi_results.values())
        rat_passed = sum(rat_9442_results.values())
        total_tests = len(test_cases)
        
        print(f"\n📊 ELECTRODE MAPPING TEST RESULTS:")
        print(f"  ROI Resolution: {roi_passed}/{total_tests} passed")
        print(f"  Rat 9442 Compatibility: {rat_passed}/{total_tests} passed")
        
        success = (roi_passed == total_tests) and (rat_passed == total_tests)
        print(f"  Overall: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in electrode mapping tests: {e}")
        return False


def test_baseline_normalization_fixes():
    """Test baseline normalization validation fixes."""
    print(f"\n🧠 TESTING BASELINE NORMALIZATION FIXES")
    print("=" * 60)
    
    try:
        from baseline_normalization import compute_baseline_statistics
        
        # Create mock data to test validation
        print("Creating mock data for validation testing...")
        
        # Case 1: Normal case with sufficient events
        mock_windows_normal = {
            1.0: {
                'windows': np.random.randn(5, 10, 100),  # 5 events, 10 freqs, 100 time points
                'window_times': np.linspace(-1, 1, 100),
                'valid_events': np.array([0, 1, 2, 3, 4]),
                'n_events': 5
            }
        }
        
        # Case 2: Few events case (should trigger warning)
        mock_windows_few = {
            2.0: {
                'windows': np.random.randn(2, 10, 100),  # Only 2 events (< 3)
                'window_times': np.linspace(-1, 1, 100),
                'valid_events': np.array([0, 1]),
                'n_events': 2
            }
        }
        
        # Case 3: Very low variance case (should trigger warning)
        low_var_data = np.random.randn(4, 10, 100) * 1e-8  # Very small variance
        mock_windows_lowvar = {
            3.0: {
                'windows': low_var_data,
                'window_times': np.linspace(-1, 1, 100),
                'valid_events': np.array([0, 1, 2, 3]),
                'n_events': 4
            }
        }
        
        window_times = np.linspace(-1, 1, 100)
        
        test_results = {}
        
        # Test normal case
        print(f"\n📊 Testing normal case (5 events):")
        try:
            stats_normal = compute_baseline_statistics(mock_windows_normal, window_times)
            has_warnings = 'warnings' in stats_normal.get(1.0, {})
            print(f"  {'✅' if not has_warnings else '⚠️ '} Normal case: {'No warnings' if not has_warnings else 'Unexpected warnings'}")
            test_results['normal'] = not has_warnings
        except Exception as e:
            print(f"  ❌ Normal case failed: {e}")
            test_results['normal'] = False
        
        # Test few events case
        print(f"\n📊 Testing few events case (2 events):")
        try:
            stats_few = compute_baseline_statistics(mock_windows_few, window_times)
            has_warnings = 'warnings' in stats_few.get(2.0, {})
            print(f"  {'✅' if has_warnings else '❌'} Few events case: {'Warning triggered' if has_warnings else 'Warning missing'}")
            test_results['few_events'] = has_warnings
        except Exception as e:
            print(f"  ❌ Few events case failed: {e}")
            test_results['few_events'] = False
        
        # Test low variance case
        print(f"\n📊 Testing low variance case:")
        try:
            stats_lowvar = compute_baseline_statistics(mock_windows_lowvar, window_times)
            has_warnings = 'warnings' in stats_lowvar.get(3.0, {})
            print(f"  {'✅' if has_warnings else '⚠️ '} Low variance case: {'Warning triggered' if has_warnings else 'Warning missing'}")
            test_results['low_variance'] = has_warnings
        except Exception as e:
            print(f"  ❌ Low variance case failed: {e}")
            test_results['low_variance'] = False
        
        # Summary
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"\n📊 BASELINE NORMALIZATION TEST RESULTS:")
        print(f"  Tests passed: {passed}/{total}")
        for test_name, success in test_results.items():
            print(f"    {test_name}: {'✅ PASSED' if success else '❌ FAILED'}")
        
        success = passed == total
        print(f"  Overall: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in baseline normalization tests: {e}")
        return False


def test_type_consistency_fixes():
    """Test type consistency fixes in electrode utilities."""
    print(f"\n🔢 TESTING TYPE CONSISTENCY FIXES")
    print("=" * 60)
    
    try:
        from nm_theta_single_basic import get_electrode_numbers_from_channels
        from electrode_utils import load_electrode_mappings
        
        # Load electrode mappings
        mapping_df = load_electrode_mappings()
        
        # Find a rat that exists in the mapping
        available_rats = list(mapping_df.index)
        if not available_rats:
            print("❌ No rats found in electrode mappings")
            return False
        
        test_rat = available_rats[0]
        print(f"Testing with rat: {test_rat}")
        
        # Get the actual electrode mapping for this rat to test with valid indices
        rat_mapping = mapping_df.loc[test_rat].values
        rat_mapping = rat_mapping[~pd.isna(rat_mapping)].astype(int).tolist()
        max_channels = len(rat_mapping)
        
        print(f"Rat {test_rat} has {max_channels} channels with electrodes: {rat_mapping}")
        
        test_results = {}
        
        # Test 1: Valid channel indices
        print(f"\n📍 Testing valid channel indices:")
        valid_indices = [0, 1, 2] if max_channels >= 3 else list(range(min(max_channels, 3)))
        try:
            result = get_electrode_numbers_from_channels(test_rat, valid_indices, mapping_df)
            all_ints = all(isinstance(x, int) for x in result)
            print(f"  Input: {valid_indices}")
            print(f"  Output: {result}")
            print(f"  {'✅' if all_ints else '❌'} All outputs are integers: {all_ints}")
            test_results['valid_indices'] = all_ints
        except Exception as e:
            print(f"  ❌ Valid indices test failed: {e}")
            test_results['valid_indices'] = False
        
        # Test 2: Out-of-range channel indices (should skip them, not add strings)
        print(f"\n📍 Testing out-of-range channel indices:")
        mixed_indices = [0, max_channels + 5, max_channels + 10]  # Include invalid indices
        try:
            result = get_electrode_numbers_from_channels(test_rat, mixed_indices, mapping_df)
            all_ints = all(isinstance(x, int) for x in result)
            no_strings = not any(isinstance(x, str) for x in result)
            print(f"  Input: {mixed_indices} (includes invalid indices)")
            print(f"  Output: {result}")
            print(f"  {'✅' if all_ints and no_strings else '❌'} No strings in output: {all_ints and no_strings}")
            test_results['invalid_indices'] = all_ints and no_strings
        except Exception as e:
            print(f"  ❌ Invalid indices test failed: {e}")
            test_results['invalid_indices'] = False
        
        # Test 3: Empty result case
        print(f"\n📍 Testing all-invalid indices:")
        all_invalid = [max_channels + 1, max_channels + 2]
        try:
            result = get_electrode_numbers_from_channels(test_rat, all_invalid, mapping_df)
            is_empty_list = isinstance(result, list) and len(result) == 0
            print(f"  Input: {all_invalid} (all invalid)")
            print(f"  Output: {result}")
            print(f"  {'✅' if is_empty_list else '❌'} Returns empty list: {is_empty_list}")
            test_results['all_invalid'] = is_empty_list
        except Exception as e:
            print(f"  ❌ All invalid indices test failed: {e}")
            test_results['all_invalid'] = False
        
        # Summary
        passed = sum(test_results.values())
        total = len(test_results)
        
        print(f"\n📊 TYPE CONSISTENCY TEST RESULTS:")
        print(f"  Tests passed: {passed}/{total}")
        for test_name, success in test_results.items():
            print(f"    {test_name}: {'✅ PASSED' if success else '❌ FAILED'}")
        
        success = passed == total
        print(f"  Overall: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in type consistency tests: {e}")
        return False


def test_integration():
    """Test integration of all fixes together."""
    print(f"\n🔗 TESTING INTEGRATION OF ALL FIXES")
    print("=" * 60)
    
    try:
        # Test that imports work correctly
        print("Testing imports...")
        
        from electrode_mapping_fixes import (
            get_electrode_numbers_from_roi_fixed,
            check_rat_9442_compatibility_fixed
        )
        from baseline_normalization import compute_baseline_statistics
        from nm_theta_single_basic import get_electrode_numbers_from_channels
        from electrode_utils import load_electrode_mappings
        
        print("✅ All imports successful")
        
        # Test that the main script can import fixes
        print("Testing main script integration...")
        
        # Import the main script (this will test the import statements we added)
        try:
            import nm_theta_cross_rats
            print("✅ Main script imports successful")
        except Exception as e:
            print(f"❌ Main script import failed: {e}")
            return False
        
        # Test a complete workflow
        print("Testing complete workflow...")
        
        mapping_df = load_electrode_mappings()
        test_roi = "8,16,24"
        
        # Step 1: ROI resolution
        electrode_numbers = get_electrode_numbers_from_roi_fixed(test_roi, mapping_df, '9442')
        print(f"  ✅ ROI resolution: {test_roi} → {electrode_numbers}")
        
        # Step 2: Compatibility check
        compatible = check_rat_9442_compatibility_fixed(test_roi, mapping_df, verbose=False)
        print(f"  {'✅' if compatible else '⚠️ '} Rat 9442 compatibility: {compatible}")
        
        # Step 3: Test deprecated function wrapper
        try:
            # This should call the fixed version and print a warning
            compatible_deprecated = nm_theta_cross_rats.check_rat_9442_compatibility(test_roi, mapping_df, verbose=False)
            print(f"  ✅ Deprecated function wrapper works: {compatible_deprecated}")
        except Exception as e:
            print(f"  ❌ Deprecated function wrapper failed: {e}")
            return False
        
        print("✅ Integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in integration test: {e}")
        return False


def main():
    """Run all bug fix tests."""
    print("🧪 COMPREHENSIVE BUG FIX TESTING")
    print("=" * 80)
    print("Testing all implemented bug fixes for the cross-rats EEG analysis pipeline")
    print("=" * 80)
    
    test_results = {}
    
    # Run all test suites
    test_results['electrode_mapping'] = test_electrode_mapping_fixes()
    test_results['baseline_normalization'] = test_baseline_normalization_fixes()
    test_results['type_consistency'] = test_type_consistency_fixes()
    test_results['integration'] = test_integration()
    
    # Overall summary
    print(f"\n🎯 OVERALL TEST RESULTS")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_suite, success in test_results.items():
        status = "✅ PASSED  " if success else "❌ FAILED  "
        print(f"  {status} {test_suite.replace('_', ' ').title()}")
    
    print(f"\nSummary: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("All bug fixes are working correctly.")
        print("The cross-rats analysis pipeline is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} TEST SUITE(S) FAILED")
        print("Please review the failed tests above and fix any issues.")
        print("Do not run the main analysis until all tests pass.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)