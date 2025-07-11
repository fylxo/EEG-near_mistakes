#!/usr/bin/env python3
"""
Session Aggregation Script for Partial SLURM Job Results

This script aggregates individual session results (nm_roi_theta_analysis_results.pkl)
into multi_session_results.pkl when SLURM jobs fail partially but some sessions
were processed successfully.

It essentially recreates what nm_theta_multi_session.py does but from pre-existing
individual session result files rather than processing sessions from scratch.

Usage:
    python aggregate_sessions_to_multi.py --rat_dir /path/to/rat_531_multi_session_mne
    python aggregate_sessions_to_multi.py --rat_dir /path/to/rat_531_multi_session_mne --verbose
"""

import os
import sys
import pickle
import json
import glob
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def find_session_results(rat_dir: str, verbose: bool = True) -> Dict[str, str]:
    """
    Find all individual session result files in a rat directory.
    
    Parameters:
    -----------
    rat_dir : str
        Path to rat directory containing session_* subdirectories
    verbose : bool
        Whether to print discovery progress
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping session_id -> path to results file
    """
    session_result_files = {}
    
    # Look for session_* directories
    session_dirs = glob.glob(os.path.join(rat_dir, "session_*"))
    
    if verbose:
        print(f"🔍 Searching for session results in: {rat_dir}")
        print(f"Found {len(session_dirs)} session directories")
    
    for session_dir in session_dirs:
        session_id = os.path.basename(session_dir).replace("session_", "")
        
        # Look for nm_roi_theta_analysis_results.pkl
        result_file = os.path.join(session_dir, "nm_roi_theta_analysis_results.pkl")
        
        if os.path.exists(result_file):
            session_result_files[session_id] = result_file
            if verbose:
                print(f"  ✓ Found results for session {session_id}: {result_file}")
        else:
            if verbose:
                print(f"  ❌ No results file found for session {session_id} in {session_dir}")
    
    return session_result_files

def load_session_results(session_result_files: Dict[str, str], verbose: bool = True) -> List[Dict]:
    """
    Load individual session results from pickle files.
    
    Parameters:
    -----------
    session_result_files : Dict[str, str]
        Dictionary mapping session_id -> path to results file
    verbose : bool
        Whether to print loading progress
        
    Returns:
    --------
    List[Dict]
        List of loaded session results
    """
    session_results = []
    
    if verbose:
        print(f"\n📂 Loading {len(session_result_files)} session result files...")
    
    for session_id, result_file in session_result_files.items():
        try:
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            # Validate that this is a session result
            if 'nm_windows' not in results:
                if verbose:
                    print(f"  ⚠️  Session {session_id}: No 'nm_windows' found, skipping")
                continue
            
            session_results.append(results)
            
            if verbose:
                n_nm_sizes = len(results['nm_windows'])
                total_events = sum(data['n_events'] for data in results['nm_windows'].values())
                print(f"  ✓ Session {session_id}: {n_nm_sizes} NM sizes, {total_events} events")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Error loading session {session_id}: {e}")
            continue
    
    return session_results

def check_session_compatibility(session_results: List[Dict], verbose: bool = True) -> bool:
    """
    Check if session results are compatible for aggregation.
    
    Parameters:
    -----------
    session_results : List[Dict]
        List of session results
    verbose : bool
        Whether to print compatibility info
        
    Returns:
    --------
    bool
        True if compatible, False otherwise
    """
    if not session_results:
        if verbose:
            print("❌ No session results to check compatibility")
        return False
    
    if verbose:
        print(f"\n🔍 Checking compatibility of {len(session_results)} sessions...")
    
    # Get reference from first session
    first_results = session_results[0]
    
    reference_frequencies = first_results['frequencies']
    reference_roi_channels = first_results['roi_channels']
    reference_nm_sizes = set(first_results['nm_windows'].keys())
    
    if verbose:
        print(f"  Reference session: {first_results['session_metadata']['session_date']}")
        print(f"  Frequencies: {len(reference_frequencies)} ({reference_frequencies[0]:.2f}-{reference_frequencies[-1]:.2f} Hz)")
        print(f"  ROI channels: {reference_roi_channels}")
        print(f"  NM sizes: {sorted([float(x) for x in reference_nm_sizes])}")
    
    # Check all other sessions
    compatible = True
    for i, results in enumerate(session_results[1:], 1):
        session_date = results['session_metadata']['session_date']
        
        # Check frequencies
        if not len(results['frequencies']) == len(reference_frequencies):
            if verbose:
                print(f"  ❌ Session {session_date}: Different number of frequencies ({len(results['frequencies'])} vs {len(reference_frequencies)})")
            compatible = False
            continue
        
        # Check ROI channels
        if not np.array_equal(results['roi_channels'], reference_roi_channels):
            if verbose:
                print(f"  ❌ Session {session_date}: Different ROI channels")
            compatible = False
            continue
        
        # Check NM sizes (allow subset, as some sessions may not have all NM sizes)
        session_nm_sizes = set(results['nm_windows'].keys())
        if not session_nm_sizes.issubset(reference_nm_sizes) and not reference_nm_sizes.issubset(session_nm_sizes):
            if verbose:
                print(f"  ⚠️  Session {session_date}: Different NM sizes ({sorted([float(x) for x in session_nm_sizes])} vs {sorted([float(x) for x in reference_nm_sizes])})")
            print(f"      This is okay - sessions may have different NM size availability")
        
        if verbose:
            print(f"  ✓ Session {session_date}: Compatible")
    
    return compatible

def aggregate_session_results(session_results: List[Dict], verbose: bool = True) -> Dict:
    """
    Aggregate individual session results into multi-session format.
    This replicates the averaging logic from nm_theta_multi_session.py
    
    Parameters:
    -----------
    session_results : List[Dict]
        List of session results to aggregate
    verbose : bool
        Whether to print aggregation progress
        
    Returns:
    --------
    Dict
        Aggregated multi-session results
    """
    if verbose:
        print(f"\n📊 Aggregating {len(session_results)} session results...")
    
    # Get metadata from first session for reference
    first_result = session_results[0]
    rat_id = first_result['session_metadata']['rat_id']
    roi_channels = first_result['roi_channels']
    roi_specification = first_result['roi_specification']
    frequencies = first_result['frequencies']
    
    # Collect all available NM sizes across sessions
    all_nm_sizes = set()
    for result in session_results:
        all_nm_sizes.update(result['nm_windows'].keys())
    
    if verbose:
        print(f"  Rat ID: {rat_id}")
        print(f"  ROI channels: {roi_channels}")
        print(f"  All NM sizes found: {sorted([float(x) for x in all_nm_sizes])}")
    
    # Initialize aggregation structures
    aggregated_windows = {}
    
    for nm_size in all_nm_sizes:
        # Collect spectrograms and metadata for this NM size
        spectrograms = []
        all_events = 0
        sessions_with_size = 0
        window_times = None
        
        for result in session_results:
            if nm_size in result['nm_windows']:
                session_data = result['nm_windows'][nm_size]
                spectrograms.append(session_data['avg_spectrogram'])
                all_events += session_data['n_events']
                sessions_with_size += 1
                
                # Use window times from first session
                if window_times is None:
                    window_times = session_data['window_times']
        
        if spectrograms:
            # Average spectrograms across sessions
            avg_spectrogram = np.mean(spectrograms, axis=0)
            
            aggregated_windows[nm_size] = {
                'avg_spectrogram': avg_spectrogram,
                'window_times': window_times,
                'n_events': all_events,
                'n_sessions': sessions_with_size
            }
            
            if verbose:
                print(f"    NM size {nm_size}: {sessions_with_size} sessions, {all_events} total events")
    
    # Create aggregated results structure
    aggregated_results = {
        'rat_id': rat_id,
        'roi_specification': roi_specification,
        'roi_channels': roi_channels,
        'frequencies': frequencies,
        'averaged_windows': aggregated_windows,
        'n_sessions_analyzed': len(session_results),
        'analysis_parameters': first_result.get('analysis_parameters', {}),  # Copy from first session
        'session_metadata': [
            {
                'session_date': result['session_metadata']['session_date'],
                'nm_sizes': list(result['nm_windows'].keys()),
                'total_events': sum(data['n_events'] for data in result['nm_windows'].values())
            }
            for result in session_results
        ]
    }
    
    if verbose:
        print(f"  ✅ Aggregation completed for {len(aggregated_windows)} NM sizes")
    
    return aggregated_results

def save_multi_session_results(results: Dict, save_path: str):
    """Save multi-session analysis results (same as nm_theta_multi_session.py)."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save main results
    results_file = os.path.join(save_path, 'multi_session_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary = {
        'rat_id': results['rat_id'],
        'roi_specification': results['roi_specification'],
        'roi_channels': results['roi_channels'],
        'n_sessions': results['n_sessions_analyzed'],
        'frequency_range': f"{results['frequencies'][0]:.1f}-{results['frequencies'][-1]:.1f} Hz",
        'nm_sizes': list(results['averaged_windows'].keys()),
        'total_events_per_size': {size: data['n_events'] for size, data in results['averaged_windows'].items()},
        'sessions_per_size': {size: data['n_sessions'] for size, data in results['averaged_windows'].items()}
    }
    
    summary_file = os.path.join(save_path, 'multi_session_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Session NM Theta Analysis Summary - Rat {results['rat_id']}\n")
        f.write("=" * 60 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Multi-session results saved to {save_path}")
    print(f"Main results: {results_file}")
    print(f"Summary: {summary_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate individual session results into multi_session_results.pkl')
    parser.add_argument('--rat_dir', required=True, help='Path to rat directory containing session_* subdirectories')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔬 Session to Multi-Session Aggregation")
    print("=" * 60)
    print(f"Rat directory: {args.rat_dir}")
    print("=" * 60)
    
    # Validate rat directory
    if not os.path.exists(args.rat_dir):
        print(f"❌ Rat directory not found: {args.rat_dir}")
        sys.exit(1)
    
    # Extract rat ID from directory name
    rat_dir_name = os.path.basename(args.rat_dir)
    if '_multi_session_' in rat_dir_name:
        rat_id = rat_dir_name.split('_')[1]  # e.g., "rat_531_multi_session_mne" -> "531"
    else:
        print(f"❌ Could not extract rat ID from directory name: {rat_dir_name}")
        print("Expected format: rat_<ID>_multi_session_<method>")
        sys.exit(1)
    
    print(f"Detected rat ID: {rat_id}")
    
    # Step 1: Find session result files
    session_result_files = find_session_results(args.rat_dir, verbose=args.verbose)
    
    if not session_result_files:
        print("❌ No session result files found!")
        print("Make sure you have session_* directories with nm_roi_theta_analysis_results.pkl files")
        sys.exit(1)
    
    # Step 2: Load session results
    session_results = load_session_results(session_result_files, verbose=args.verbose)
    
    if not session_results:
        print("❌ No valid session results loaded!")
        sys.exit(1)
    
    # Step 3: Check compatibility
    if not check_session_compatibility(session_results, verbose=args.verbose):
        print("❌ Session results are not compatible for aggregation!")
        print("Make sure all sessions were processed with the same parameters")
        sys.exit(1)
    
    # Step 4: Aggregate results
    print(f"\n📊 Aggregating results from {len(session_results)} sessions...")
    
    try:
        aggregated_results = aggregate_session_results(session_results, verbose=args.verbose)
        
        print(f"✅ Aggregation completed successfully!")
        
    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Save results
    print(f"\n💾 Saving multi-session results to {args.rat_dir}...")
    
    try:
        save_multi_session_results(aggregated_results, args.rat_dir)
        
        print(f"✅ Multi-session results saved successfully!")
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 6: Summary
    print(f"\n" + "=" * 60)
    print("📊 AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Rat ID: {aggregated_results['rat_id']}")
    print(f"Sessions processed: {aggregated_results['n_sessions_analyzed']}")
    print(f"NM sizes: {[float(key) for key in aggregated_results['averaged_windows'].keys()]}")
    print(f"ROI channels: {aggregated_results['roi_channels']}")
    print(f"Total events per size: {[data['n_events'] for data in aggregated_results['averaged_windows'].values()]}")
    print(f"Results directory: {args.rat_dir}")
    
    print(f"\n✅ Session aggregation completed successfully!")
    print(f"\n📝 Next steps:")
    print(f"   - The multi_session_results.pkl file is now available for cross-rat aggregation")
    print(f"   - You can run scripts/aggregate_results.py to aggregate across multiple rats")

if __name__ == "__main__":
    main()