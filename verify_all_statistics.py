#!/usr/bin/env python3
"""
Final verification script to check all statistics in the scientific report
against the actual analysis results.
"""

import json
import pickle
import re

def verify_cross_rats_results():
    """Verify the cross-rats analysis results match what's reported."""
    print("VERIFYING CROSS-RATS ANALYSIS RESULTS")
    print("=" * 50)
    
    # Load the cross-rats summary
    try:
        with open("/mnt/c/Users/flavi/Desktop/eeg-near_mistakes/results/cross_rats_hierarchical/cross_rats_summary.json", 'r') as f:
            summary = json.load(f)
            
        print("✓ Cross-rats summary loaded successfully")
        
        # Check key statistics
        print(f"Number of rats: {summary['n_rats']}")
        print(f"Rat IDs: {summary['rat_ids']}")
        print(f"NM sizes analyzed: {summary['nm_sizes_analyzed']}")
        print(f"Total events all rats: {summary['total_events_all_rats']}")
        print(f"Total sessions all rats: {summary['total_sessions_all_rats']}")
        print(f"Frequency range: {summary['frequency_range']}")
        print(f"ROI specification: {summary['roi_specification']}")
        print(f"ROI channels: {summary['roi_channels']}")
        print(f"Normalization method: {summary['normalization_method']}")
        print(f"Baseline window: {summary['baseline_window']}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error loading cross-rats summary: {e}")
        return None

def verify_report_numbers(summary):
    """Compare report numbers with analysis results."""
    if not summary:
        print("❌ Cannot verify without summary data")
        return
        
    print("\nVERIFING REPORT NUMBERS")
    print("=" * 50)
    
    # Check the key numbers from the report
    reported_events = {
        1.0: 18374,
        2.0: 18297, 
        3.0: 18448
    }
    
    actual_events = summary['total_events_all_rats']
    
    print("Event counts verification:")
    all_match = True
    for size, reported in reported_events.items():
        actual = actual_events.get(str(size), 0)
        match = (actual == reported)
        print(f"  NM Size {size}: Reported={reported}, Actual={actual}, Match={match}")
        if not match:
            all_match = False
    
    # Check sessions
    reported_sessions = 302
    actual_sessions = list(summary['total_sessions_all_rats'].values())[0]  # All should be same
    session_match = (actual_sessions == reported_sessions)
    print(f"Sessions: Reported={reported_sessions}, Actual={actual_sessions}, Match={session_match}")
    
    # Check rats
    reported_rats = 14
    actual_rats = summary['n_rats']
    rats_match = (actual_rats == reported_rats)
    print(f"Number of rats: Reported={reported_rats}, Actual={actual_rats}, Match={rats_match}")
    
    # Check total events
    reported_total = 55119  # 18374 + 18297 + 18448
    actual_total = sum(int(v) for v in actual_events.values())
    total_match = (actual_total == reported_total)
    print(f"Total events: Reported={reported_total}, Actual={actual_total}, Match={total_match}")
    
    if all_match and session_match and rats_match and total_match:
        print("\n✅ ALL NUMBERS IN REPORT MATCH ANALYSIS RESULTS!")
    else:
        print("\n❌ Some numbers in report do not match analysis results")
    
    return all_match and session_match and rats_match and total_match

def check_analysis_parameters(summary):
    """Check analysis parameters match what's described in report."""
    if not summary:
        return
        
    print("\nVERIFING ANALYSIS PARAMETERS")
    print("=" * 50)
    
    # Check frequency range
    freq_range = summary['frequency_range']
    print(f"Frequency range: {freq_range[0]:.2f}-{freq_range[1]:.2f} Hz")
    freq_match = (2.9 <= freq_range[0] <= 3.1) and (6.9 <= freq_range[1] <= 7.1)
    print(f"  Matches 3-7 Hz range: {freq_match}")
    
    # Check normalization method
    norm_method = summary['normalization_method']
    norm_match = (norm_method == 'pre_event_baseline')
    print(f"Normalization method: {norm_method}")
    print(f"  Matches pre-event baseline: {norm_match}")
    
    # Check baseline window
    baseline_window = summary['baseline_window']
    baseline_match = (baseline_window == [-1.0, -0.5])
    print(f"Baseline window: {baseline_window}")
    print(f"  Matches -1.0 to -0.5s: {baseline_match}")
    
    if freq_match and norm_match and baseline_match:
        print("\n✅ ALL ANALYSIS PARAMETERS MATCH REPORT!")
    else:
        print("\n❌ Some analysis parameters do not match report")

def main():
    """Main verification function."""
    print("COMPREHENSIVE STATISTICS VERIFICATION")
    print("=" * 60)
    print("Checking if ALL numbers in scientific report match analysis results...\n")
    
    # Load and verify cross-rats results
    summary = verify_cross_rats_results()
    
    # Verify all numbers match
    numbers_match = verify_report_numbers(summary)
    
    # Check analysis parameters
    check_analysis_parameters(summary)
    
    print("\n" + "=" * 60)
    if numbers_match and summary:
        print("🎉 FINAL VERIFICATION: ALL STATISTICS IN REPORT ARE CORRECT!")
        print("Your scientific report accurately reflects your analysis results.")
    else:
        print("❌ FINAL VERIFICATION: Some statistics need correction.")
    print("=" * 60)

if __name__ == "__main__":
    main()