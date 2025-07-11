#!/usr/bin/env python3
"""
Batch Session Aggregation Script

This script finds all rat directories with partial session results and
aggregates them into multi_session_results.pkl files for cross-rat analysis.

Usage:
    python batch_aggregate_sessions.py --results_dir /path/to/results
    python batch_aggregate_sessions.py --results_dir /path/to/results --pattern "rat_*_multi_session_*"
"""

import os
import sys
import glob
import subprocess
from typing import List

def find_rat_directories(results_dir: str, pattern: str = "rat_*/rat_*_multi_session_*") -> List[str]:
    """
    Find all rat directories matching the pattern.
    
    Parameters:
    -----------
    results_dir : str
        Base directory to search in
    pattern : str
        Glob pattern for rat directories
        
    Returns:
    --------
    List[str]
        List of rat directory paths
    """
    search_pattern = os.path.join(results_dir, pattern)
    rat_dirs = glob.glob(search_pattern)
    return sorted(rat_dirs)

def check_needs_aggregation(rat_dir: str) -> bool:
    """
    Check if a rat directory needs session aggregation.
    
    Returns True if:
    - Has session_* subdirectories with results
    - Does not have multi_session_results.pkl OR it's older than session results
    
    Parameters:
    -----------
    rat_dir : str
        Path to rat directory
        
    Returns:
    --------
    bool
        True if aggregation is needed
    """
    # Check for session directories
    session_dirs = glob.glob(os.path.join(rat_dir, "session_*"))
    if not session_dirs:
        return False
    
    # Check for session result files
    session_results = []
    for session_dir in session_dirs:
        result_file = os.path.join(session_dir, "nm_roi_theta_analysis_results.pkl")
        if os.path.exists(result_file):
            session_results.append(result_file)
    
    if not session_results:
        return False
    
    # Check if multi_session_results.pkl exists
    multi_result_file = os.path.join(rat_dir, "multi_session_results.pkl")
    if not os.path.exists(multi_result_file):
        return True
    
    # Check if multi_session_results.pkl is older than any session result
    multi_mtime = os.path.getmtime(multi_result_file)
    for session_result in session_results:
        if os.path.getmtime(session_result) > multi_mtime:
            return True
    
    return False

def run_aggregation(rat_dir: str, verbose: bool = True) -> bool:
    """
    Run session aggregation for a rat directory.
    
    Parameters:
    -----------
    rat_dir : str
        Path to rat directory
    verbose : bool
        Whether to use verbose output
        
    Returns:
    --------
    bool
        True if successful
    """
    script_path = os.path.join(os.path.dirname(__file__), "aggregate_sessions_to_multi.py")
    
    cmd = ["python", script_path, "--rat_dir", rat_dir]
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        if verbose:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {rat_dir}:")
        print(e.stderr)
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch aggregate session results across multiple rats')
    parser.add_argument('--results_dir', required=True, help='Base directory containing rat directories')
    parser.add_argument('--pattern', default='rat_*/rat_*_multi_session_*', help='Glob pattern for rat directories')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be processed without actually running')
    
    args = parser.parse_args()
    
    print("🔬 Batch Session Aggregation")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Find rat directories
    rat_dirs = find_rat_directories(args.results_dir, args.pattern)
    
    if not rat_dirs:
        print(f"❌ No rat directories found matching pattern: {args.pattern}")
        sys.exit(1)
    
    print(f"Found {len(rat_dirs)} rat directories:")
    for rat_dir in rat_dirs:
        print(f"  {os.path.basename(rat_dir)}")
    
    # Check which need aggregation
    needs_aggregation = []
    
    print(f"\n🔍 Checking which rats need session aggregation...")
    
    for rat_dir in rat_dirs:
        rat_name = os.path.basename(rat_dir)
        
        if check_needs_aggregation(rat_dir):
            needs_aggregation.append(rat_dir)
            session_count = len(glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl")))
            multi_exists = os.path.exists(os.path.join(rat_dir, "multi_session_results.pkl"))
            status = "missing" if not multi_exists else "outdated"
            print(f"  ✓ {rat_name}: {session_count} sessions, multi_session_results.pkl {status}")
        else:
            print(f"  ⏭️  {rat_name}: Already up to date")
    
    if not needs_aggregation:
        print(f"\n✅ All rat directories are already up to date!")
        return
    
    print(f"\n📊 {len(needs_aggregation)} rats need session aggregation:")
    for rat_dir in needs_aggregation:
        print(f"  - {os.path.basename(rat_dir)}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN: Would process {len(needs_aggregation)} rats")
        return
    
    # Process rats that need aggregation
    print(f"\n🚀 Processing {len(needs_aggregation)} rats...")
    
    successful = []
    failed = []
    
    for i, rat_dir in enumerate(needs_aggregation, 1):
        rat_name = os.path.basename(rat_dir)
        print(f"\n--- Processing {i}/{len(needs_aggregation)}: {rat_name} ---")
        
        if run_aggregation(rat_dir, verbose=args.verbose):
            successful.append(rat_name)
            print(f"✅ {rat_name}: Success")
        else:
            failed.append(rat_name)
            print(f"❌ {rat_name}: Failed")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 BATCH AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Total rat directories found: {len(rat_dirs)}")
    print(f"Needed aggregation: {len(needs_aggregation)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successful rats:")
        for rat_name in successful:
            print(f"  - {rat_name}")
    
    if failed:
        print(f"\n❌ Failed rats:")
        for rat_name in failed:
            print(f"  - {rat_name}")
    
    if successful and not failed:
        print(f"\n🎉 All aggregations completed successfully!")
        print(f"\n📝 Next steps:")
        print(f"   - All rats now have multi_session_results.pkl files")
        print(f"   - You can run scripts/aggregate_results.py to aggregate across all rats")
        print(f"   - Example: python scripts/aggregate_results.py --results_path {args.results_dir}")
    elif successful:
        print(f"\n⚠️  Some aggregations failed. Check the errors above and retry failed rats individually.")
    else:
        print(f"\n❌ All aggregations failed. Check your data and try again.")

if __name__ == "__main__":
    main()