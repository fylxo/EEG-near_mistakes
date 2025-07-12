#!/usr/bin/env python3
"""
Aggregation Script for Cross-Rats Analysis

This script aggregates individual rat results from array jobs into cross-rat averages.
It reads the multi_session_results.pkl file from each rat and combines them.
"""

import os
import sys
import pickle
import json
import glob
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the aggregation function from nm_theta_cross_rats
from core.nm_theta_cross_rats import aggregate_cross_rats_results, create_cross_rats_visualizations

def find_rat_results(results_base_path: str, verbose: bool = True) -> Dict[str, str]:
    """
    Find all individual rat result files.
    
    Parameters:
    -----------
    results_base_path : str
        Base directory containing rat_* subdirectories
    verbose : bool
        Whether to print discovery progress
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping rat_id -> path to results file
    """
    rat_result_files = {}
    
    # Look for rat_* directories
    rat_dirs = glob.glob(os.path.join(results_base_path, "rat_*"))
    
    if verbose:
        print(f"🔍 Searching for rat results in: {results_base_path}")
        print(f"Found {len(rat_dirs)} rat directories")
    
    for rat_dir in rat_dirs:
        rat_id = os.path.basename(rat_dir).replace("rat_", "")
        
        # Look for multi_session_results.pkl
        result_file = os.path.join(rat_dir, "multi_session_results.pkl")
        
        if os.path.exists(result_file):
            rat_result_files[rat_id] = result_file
            if verbose:
                print(f"  ✓ Found results for rat {rat_id}: {result_file}")
        else:
            if verbose:
                print(f"  ❌ No results file found for rat {rat_id} in {rat_dir}")
    
    return rat_result_files

def load_rat_results(rat_result_files: Dict[str, str], verbose: bool = True) -> Dict[str, Dict]:
    """
    Load individual rat results from pickle files.
    
    Parameters:
    -----------
    rat_result_files : Dict[str, str]
        Dictionary mapping rat_id -> path to results file
    verbose : bool
        Whether to print loading progress
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary mapping rat_id -> loaded results
    """
    rat_results = {}
    
    if verbose:
        print(f"\n📂 Loading {len(rat_result_files)} rat result files...")
    
    for rat_id, result_file in rat_result_files.items():
        try:
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            # Validate that this is a multi-session result
            if 'averaged_windows' not in results:
                if verbose:
                    print(f"  ⚠️  Rat {rat_id}: No 'averaged_windows' found, skipping")
                continue
            
            rat_results[rat_id] = results
            
            if verbose:
                n_nm_sizes = len(results['averaged_windows'])
                n_sessions = results.get('n_sessions_analyzed', 'unknown')
                print(f"  ✓ Rat {rat_id}: {n_nm_sizes} NM sizes, {n_sessions} sessions")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Error loading rat {rat_id}: {e}")
            continue
    
    return rat_results

def check_compatibility(rat_results: Dict[str, Dict], verbose: bool = True) -> bool:
    """
    Check if rat results are compatible for aggregation.
    
    Parameters:
    -----------
    rat_results : Dict[str, Dict]
        Dictionary of rat results
    verbose : bool
        Whether to print compatibility info
        
    Returns:
    --------
    bool
        True if compatible, False otherwise
    """
    if not rat_results:
        if verbose:
            print("❌ No rat results to check compatibility")
        return False
    
    if verbose:
        print(f"\n🔍 Checking compatibility of {len(rat_results)} rats...")
    
    # Get reference from first rat
    first_rat_id = next(iter(rat_results.keys()))
    first_results = rat_results[first_rat_id]
    
    reference_frequencies = first_results['frequencies']
    reference_roi_channels = first_results['roi_channels']
    reference_nm_sizes = set(first_results['averaged_windows'].keys())
    
    if verbose:
        print(f"  Reference rat: {first_rat_id}")
        print(f"  Frequencies: {len(reference_frequencies)} ({reference_frequencies[0]:.2f}-{reference_frequencies[-1]:.2f} Hz)")
        print(f"  ROI channels: {reference_roi_channels}")
        print(f"  NM sizes: {sorted([float(x) for x in reference_nm_sizes])}")
    
    # Check all other rats
    compatible = True
    for rat_id, results in rat_results.items():
        if rat_id == first_rat_id:
            continue
        
        # Check frequencies
        if not len(results['frequencies']) == len(reference_frequencies):
            if verbose:
                print(f"  ❌ Rat {rat_id}: Different number of frequencies ({len(results['frequencies'])} vs {len(reference_frequencies)})")
            compatible = False
            continue
        
        # Check ROI channels
        if not (results['roi_channels'] == reference_roi_channels).all():
            if verbose:
                print(f"  ❌ Rat {rat_id}: Different ROI channels")
            compatible = False
            continue
        
        # Check NM sizes
        rat_nm_sizes = set(results['averaged_windows'].keys())
        if rat_nm_sizes != reference_nm_sizes:
            if verbose:
                print(f"  ❌ Rat {rat_id}: Different NM sizes ({sorted([float(x) for x in rat_nm_sizes])} vs {sorted([float(x) for x in reference_nm_sizes])})")
            compatible = False
            continue
        
        if verbose:
            print(f"  ✓ Rat {rat_id}: Compatible")
    
    return compatible

def main():
    import argparse
    
    print("🚀 Starting aggregate_results.py script...")
    print(f"📍 Script location: {__file__}")
    print(f"📁 Current working directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Aggregate individual rat results into cross-rat averages')
    parser.add_argument('--results_path', required=True, help='Base path containing rat_* result directories')
    parser.add_argument('--output_path', help='Output directory for aggregated results (default: results_path/cross_rats_aggregated)')
    parser.add_argument('--roi', default='1,2,3', help='ROI specification (for metadata)')
    parser.add_argument('--freq_min', type=float, default=1.0, help='Minimum frequency (for metadata)')
    parser.add_argument('--freq_max', type=float, default=45.0, help='Maximum frequency (for metadata)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    print("🔧 Parsing command line arguments...")
    args = parser.parse_args()
    print(f"✅ Arguments parsed successfully")
    
    # Set default output path
    if args.output_path is None:
        args.output_path = os.path.join(args.results_path, 'cross_rats_aggregated')
    
    print("🧠 Cross-Rats Results Aggregation")
    print("=" * 60)
    print(f"Results path: {args.results_path}")
    print(f"Output path: {args.output_path}")
    print(f"ROI: {args.roi}")
    print(f"Frequency range: {args.freq_min}-{args.freq_max} Hz")
    print("=" * 60)
    
    # Step 1: Find rat result files
    print(f"\n🔍 Step 1: Looking for rat result files...")
    print(f"📂 Checking directory: {os.path.abspath(args.results_path)}")
    print(f"📊 Directory exists: {os.path.exists(args.results_path)}")
    if os.path.exists(args.results_path):
        print(f"📋 Directory contents: {os.listdir(args.results_path)[:10]}...")  # Show first 10 items
    
    rat_result_files = find_rat_results(args.results_path, verbose=args.verbose)
    
    if not rat_result_files:
        print("❌ No rat result files found!")
        print("Make sure you have rat_* directories with multi_session_results.pkl files")
        sys.exit(1)
    
    # Step 2: Load rat results
    rat_results = load_rat_results(rat_result_files, verbose=args.verbose)
    
    if not rat_results:
        print("❌ No valid rat results loaded!")
        sys.exit(1)
    
    # Step 3: Check compatibility
    if not check_compatibility(rat_results, verbose=args.verbose):
        print("❌ Rat results are not compatible for aggregation!")
        print("Make sure all rats were processed with the same parameters")
        sys.exit(1)
    
    # Step 4: Aggregate results
    print(f"\n📊 Aggregating results from {len(rat_results)} rats...")
    
    try:
        aggregated_results = aggregate_cross_rats_results(
            rat_results=rat_results,
            roi_specification=args.roi,
            freq_range=(args.freq_min, args.freq_max),
            save_path=args.output_path,
            verbose=args.verbose
        )
        
        print(f"✅ Aggregation completed successfully!")
        print(f"Results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Create visualizations
    print(f"\n📈 Creating visualizations...")
    
    try:
        create_cross_rats_visualizations(
            results=aggregated_results,
            save_path=args.output_path,
            verbose=args.verbose
        )
        
        print(f"✅ Visualizations created successfully!")
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't exit here - aggregation was successful
    
    # Step 6: Summary
    print(f"\n" + "=" * 60)
    print("📊 AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Rats processed: {len(rat_results)}")
    print(f"Successful rats: {aggregated_results['n_rats']}")
    print(f"Rat IDs: {aggregated_results['rat_ids']}")
    print(f"NM sizes: {[float(key) for key in aggregated_results['averaged_windows'].keys()]}")
    print(f"ROI channels: {aggregated_results['roi_channels']}")
    print(f"Results directory: {args.output_path}")
    
    print(f"\n✅ Cross-rats analysis completed successfully!")

if __name__ == "__main__":
    main()