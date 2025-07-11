#!/usr/bin/env python3
"""
Simple Batch Session Aggregation - Direct Processing

This script directly aggregates sessions without subprocess overhead.
Much simpler and more efficient than the subprocess-based version.

Usage:
    python scripts/simple_batch_aggregate.py --results_dir results/
"""

import os
import sys
import glob
import pickle
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def find_session_results(rat_dir: str) -> Dict[str, str]:
    """Find all individual session result files in a rat directory."""
    session_result_files = {}
    session_dirs = glob.glob(os.path.join(rat_dir, "session_*"))
    
    for session_dir in session_dirs:
        session_id = os.path.basename(session_dir).replace("session_", "")
        result_file = os.path.join(session_dir, "nm_roi_theta_analysis_results.pkl")
        
        if os.path.exists(result_file):
            session_result_files[session_id] = result_file
    
    return session_result_files

def load_session_results(session_result_files: Dict[str, str]) -> List[Dict]:
    """Load individual session results from pickle files."""
    session_results = []
    
    for session_id, result_file in session_result_files.items():
        try:
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            if 'normalized_windows' not in results:
                continue
            
            session_results.append(results)
                
        except Exception as e:
            print(f"    ⚠️  Error loading session {session_id}: {e}")
            continue
    
    return session_results

def aggregate_session_results(session_results: List[Dict]) -> Dict:
    """Aggregate individual session results into multi-session format."""
    if not session_results:
        raise ValueError("No session results to aggregate")
    
    # Get metadata from first session
    first_result = session_results[0]
    rat_id = first_result['session_metadata']['rat_id']
    roi_channels = first_result['roi_channels']
    roi_specification = first_result['roi_specification']
    frequencies = first_result['frequencies']
    
    # Collect all available NM sizes
    all_nm_sizes = set()
    for result in session_results:
        all_nm_sizes.update(result['normalized_windows'].keys())
    
    # Initialize aggregation
    aggregated_windows = {}
    
    for nm_size in all_nm_sizes:
        spectrograms = []
        all_events = 0
        sessions_with_size = 0
        window_times = None
        
        for result in session_results:
            if nm_size in result['normalized_windows']:
                session_data = result['normalized_windows'][nm_size]
                spectrograms.append(session_data['avg_spectrogram'])
                all_events += session_data['n_events']
                sessions_with_size += 1
                
                if window_times is None:
                    window_times = session_data['window_times']
        
        if spectrograms:
            avg_spectrogram = np.mean(spectrograms, axis=0)
            
            aggregated_windows[nm_size] = {
                'avg_spectrogram': avg_spectrogram,
                'window_times': window_times,
                'n_events': all_events,
                'n_sessions': sessions_with_size
            }
    
    # Create final results
    aggregated_results = {
        'rat_id': rat_id,
        'roi_specification': roi_specification,
        'roi_channels': roi_channels,
        'frequencies': frequencies,
        'averaged_windows': aggregated_windows,
        'n_sessions_analyzed': len(session_results),
        'session_metadata': [
            {
                'session_date': result['session_metadata']['session_date'],
                'nm_sizes': list(result['normalized_windows'].keys()),
                'total_events': sum(data['n_events'] for data in result['normalized_windows'].values())
            }
            for result in session_results
        ]
    }
    
    return aggregated_results

def save_multi_session_results(results: Dict, save_path: str):
    """Save multi-session analysis results."""
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

def process_rat_directory(rat_dir: str, verbose: bool = True) -> bool:
    """Process a single rat directory for session aggregation."""
    rat_name = os.path.basename(rat_dir)
    
    if verbose:
        print(f"\n--- Processing {rat_name} ---")
    
    # Check if already has multi_session_results.pkl
    multi_result_file = os.path.join(rat_dir, "multi_session_results.pkl")
    if os.path.exists(multi_result_file):
        if verbose:
            print(f"  ⏭️  Already has multi_session_results.pkl - skipping")
        return True
    
    # Find session results
    session_result_files = find_session_results(rat_dir)
    
    if not session_result_files:
        if verbose:
            print(f"  ❌ No session result files found")
        return False
    
    if verbose:
        print(f"  📂 Found {len(session_result_files)} session results")
    
    # Load session results
    session_results = load_session_results(session_result_files)
    
    if not session_results:
        if verbose:
            print(f"  ❌ No valid session results loaded")
        return False
    
    if verbose:
        print(f"  📊 Loaded {len(session_results)} valid sessions")
    
    # Aggregate results
    try:
        aggregated_results = aggregate_session_results(session_results)
        
        if verbose:
            nm_sizes = list(aggregated_results['averaged_windows'].keys())
            total_events = sum(data['n_events'] for data in aggregated_results['averaged_windows'].values())
            print(f"  ✓ Aggregated {len(nm_sizes)} NM sizes, {total_events} total events")
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Aggregation failed: {e}")
        return False
    
    # Save results
    try:
        save_multi_session_results(aggregated_results, rat_dir)
        
        if verbose:
            print(f"  💾 Saved multi_session_results.pkl")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Save failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple batch session aggregation without subprocess overhead')
    parser.add_argument('--results_dir', required=True, help='Base directory containing rat directories')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be processed without actually running')
    
    args = parser.parse_args()
    
    print("🔬 Simple Batch Session Aggregation")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Find rat directories
    rat_dirs = glob.glob(os.path.join(args.results_dir, "rat_*/rat_*_multi_session_*"))
    
    if not rat_dirs:
        print(f"❌ No rat directories found")
        sys.exit(1)
    
    print(f"Found {len(rat_dirs)} rat directories")
    
    # Check which need processing
    needs_processing = []
    
    for rat_dir in rat_dirs:
        rat_name = os.path.basename(rat_dir)
        multi_result_file = os.path.join(rat_dir, "multi_session_results.pkl")
        session_results = glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl"))
        
        if os.path.exists(multi_result_file):
            if args.verbose:
                print(f"  ⏭️  {rat_name}: Already complete")
        elif session_results:
            needs_processing.append(rat_dir)
            if args.verbose:
                print(f"  🔄 {rat_name}: Needs aggregation ({len(session_results)} sessions)")
        else:
            if args.verbose:
                print(f"  ❌ {rat_name}: No session results found")
    
    if not needs_processing:
        print(f"\n✅ All rats already have multi_session_results.pkl!")
        return
    
    print(f"\n📊 {len(needs_processing)} rats need session aggregation")
    
    if args.dry_run:
        print(f"🔍 DRY RUN: Would process:")
        for rat_dir in needs_processing:
            print(f"  - {os.path.basename(rat_dir)}")
        return
    
    # Process rats
    successful = 0
    failed = 0
    
    for rat_dir in needs_processing:
        if process_rat_directory(rat_dir, verbose=args.verbose):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Rats processed: {len(needs_processing)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful:
        print(f"\n✅ {successful} rats now have multi_session_results.pkl files!")
        print(f"📝 Next: Run cross-rat aggregation with scripts/aggregate_results.py")
    
    if failed:
        print(f"\n⚠️  {failed} rats failed - check individual errors above")

if __name__ == "__main__":
    main()