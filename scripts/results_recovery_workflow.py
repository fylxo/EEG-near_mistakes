#!/usr/bin/env python3
"""
Results Recovery Workflow

This script helps recover from partial SLURM job failures by:
1. Finding rats with successful individual sessions but missing multi_session_results.pkl
2. Aggregating individual session results into multi_session_results.pkl
3. Running cross-rat aggregation to create final visualizations

This is a comprehensive workflow script that handles the complete pipeline
from partial session results to final cross-rat analysis.

Usage:
    python results_recovery_workflow.py --results_dir /path/to/results
    python results_recovery_workflow.py --results_dir /path/to/results --roi "1,2,3" --freq_range 3 8
"""

import os
import sys
import subprocess
import glob
from typing import List, Tuple

def check_environment():
    """Check if required scripts are available."""
    script_dir = os.path.dirname(__file__)
    required_scripts = [
        "aggregate_sessions_to_multi.py",
        "batch_aggregate_sessions.py", 
        "aggregate_results.py"
    ]
    
    missing = []
    for script in required_scripts:
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            missing.append(script)
    
    if missing:
        print(f"❌ Missing required scripts: {missing}")
        print(f"Make sure these scripts are in the scripts/ directory")
        return False
    
    return True

def analyze_results_directory(results_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyze the results directory to categorize rat directories.
    
    Returns:
    --------
    Tuple[List[str], List[str], List[str]]
        (completed_rats, partial_rats, failed_rats)
    """
    # Find all rat directories
    rat_dirs = glob.glob(os.path.join(results_dir, "rat_*/rat_*_multi_session_*"))
    
    completed_rats = []
    partial_rats = []
    failed_rats = []
    
    for rat_dir in rat_dirs:
        rat_name = os.path.basename(rat_dir)
        
        # Check for multi_session_results.pkl (completed)
        multi_result = os.path.join(rat_dir, "multi_session_results.pkl")
        
        # Check for individual session results
        session_results = glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl"))
        
        if os.path.exists(multi_result):
            completed_rats.append(rat_name)
        elif session_results:
            partial_rats.append(rat_name)
        else:
            failed_rats.append(rat_name)
    
    return completed_rats, partial_rats, failed_rats

def run_session_aggregation(results_dir: str, verbose: bool = True) -> bool:
    """Run batch session aggregation."""
    script_path = os.path.join(os.path.dirname(__file__), "batch_aggregate_sessions.py")
    
    cmd = ["python", script_path, "--results_dir", results_dir]
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def run_cross_rat_aggregation(results_dir: str, roi: str, freq_min: float, freq_max: float, verbose: bool = True) -> bool:
    """Run cross-rat aggregation."""
    script_path = os.path.join(os.path.dirname(__file__), "aggregate_results.py")
    
    cmd = [
        "python", script_path,
        "--results_path", results_dir,
        "--roi", roi,
        "--freq_min", str(freq_min),
        "--freq_max", str(freq_max)
    ]
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete results recovery workflow from partial SLURM failures')
    parser.add_argument('--results_dir', required=True, help='Base directory containing rat results')
    parser.add_argument('--roi', default='1,2,3', help='ROI specification for cross-rat analysis')
    parser.add_argument('--freq_min', type=float, default=3.0, help='Minimum frequency for analysis')
    parser.add_argument('--freq_max', type=float, default=8.0, help='Maximum frequency for analysis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip_session_aggregation', action='store_true', help='Skip session aggregation step')
    parser.add_argument('--skip_cross_rat', action='store_true', help='Skip cross-rat aggregation step')
    
    args = parser.parse_args()
    
    print("🔬 Results Recovery Workflow")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"ROI: {args.roi}")
    print(f"Frequency range: {args.freq_min}-{args.freq_max} Hz")
    print("=" * 80)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Step 1: Analyze current state
    print(f"\n🔍 STEP 1: Analyzing current results state...")
    completed_rats, partial_rats, failed_rats = analyze_results_directory(args.results_dir)
    
    print(f"\n📊 RESULTS ANALYSIS:")
    print(f"  ✅ Completed rats (have multi_session_results.pkl): {len(completed_rats)}")
    if completed_rats:
        for rat in completed_rats[:5]:  # Show first 5
            print(f"    - {rat}")
        if len(completed_rats) > 5:
            print(f"    ... and {len(completed_rats) - 5} more")
    
    print(f"  🔄 Partial rats (have session results, missing multi_session): {len(partial_rats)}")
    if partial_rats:
        for rat in partial_rats:
            rat_dir = os.path.join(args.results_dir, rat)
            session_count = len(glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl")))
            print(f"    - {rat} ({session_count} sessions)")
    
    print(f"  ❌ Failed rats (no usable results): {len(failed_rats)}")
    if failed_rats:
        for rat in failed_rats[:3]:  # Show first 3
            print(f"    - {rat}")
        if len(failed_rats) > 3:
            print(f"    ... and {len(failed_rats) - 3} more")
    
    # Step 2: Session aggregation
    if not args.skip_session_aggregation and partial_rats:
        print(f"\n🔄 STEP 2: Aggregating individual sessions for {len(partial_rats)} partial rats...")
        
        if run_session_aggregation(args.results_dir, verbose=args.verbose):
            print(f"✅ Session aggregation completed successfully!")
            
            # Re-analyze to see progress
            completed_rats_after, partial_rats_after, failed_rats_after = analyze_results_directory(args.results_dir)
            recovered = len(completed_rats_after) - len(completed_rats)
            print(f"📈 Recovered {recovered} rats from partial results")
            
            # Update counts
            completed_rats = completed_rats_after
            partial_rats = partial_rats_after
            
        else:
            print(f"❌ Session aggregation failed!")
            print(f"Check the errors above and try running batch_aggregate_sessions.py manually")
    
    elif args.skip_session_aggregation:
        print(f"\n⏭️  STEP 2: Skipping session aggregation (--skip_session_aggregation)")
    
    elif not partial_rats:
        print(f"\n⏭️  STEP 2: No partial rats found - skipping session aggregation")
    
    # Step 3: Cross-rat aggregation
    if not args.skip_cross_rat and completed_rats:
        print(f"\n📊 STEP 3: Running cross-rat aggregation for {len(completed_rats)} completed rats...")
        
        if run_cross_rat_aggregation(args.results_dir, args.roi, args.freq_min, args.freq_max, verbose=args.verbose):
            print(f"✅ Cross-rat aggregation completed successfully!")
            
            # Check for output
            cross_rat_dir = os.path.join(args.results_dir, "cross_rats_aggregated")
            if os.path.exists(cross_rat_dir):
                files = os.listdir(cross_rat_dir)
                print(f"📁 Cross-rat results saved to: {cross_rat_dir}")
                print(f"📁 Generated files: {files}")
            
        else:
            print(f"❌ Cross-rat aggregation failed!")
            print(f"Check the errors above and try running aggregate_results.py manually")
    
    elif args.skip_cross_rat:
        print(f"\n⏭️  STEP 3: Skipping cross-rat aggregation (--skip_cross_rat)")
    
    elif not completed_rats:
        print(f"\n⚠️  STEP 3: No completed rats found - cannot run cross-rat aggregation")
        print(f"You may need to run session aggregation first or check your SLURM job results")
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("🎯 WORKFLOW SUMMARY")
    print("=" * 80)
    
    final_completed, final_partial, final_failed = analyze_results_directory(args.results_dir)
    
    print(f"Final state:")
    print(f"  ✅ Completed rats: {len(final_completed)}")
    print(f"  🔄 Partial rats: {len(final_partial)}")
    print(f"  ❌ Failed rats: {len(final_failed)}")
    
    # Check for final cross-rat results
    cross_rat_dir = os.path.join(args.results_dir, "cross_rats_aggregated")
    if os.path.exists(cross_rat_dir):
        print(f"\n🎉 SUCCESS! Final cross-rat results available at:")
        print(f"   {cross_rat_dir}")
        
        # List key files
        key_files = ["cross_rats_aggregated_results.pkl", "cross_rats_nm_theta_analysis.png", "cross_rats_summary.txt"]
        available_files = []
        for file in key_files:
            file_path = os.path.join(cross_rat_dir, file)
            if os.path.exists(file_path):
                available_files.append(file)
        
        if available_files:
            print(f"   Key files: {available_files}")
    
    else:
        print(f"\n⚠️  No final cross-rat results found.")
        if final_completed:
            print(f"   You can run cross-rat aggregation manually:")
            print(f"   python scripts/aggregate_results.py --results_path {args.results_dir} --roi {args.roi}")
        else:
            print(f"   You need to get more rats to completed state first.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if final_partial:
        print(f"   - {len(final_partial)} rats still have partial results")
        print(f"   - Run session aggregation: python scripts/batch_aggregate_sessions.py --results_dir {args.results_dir}")
    
    if final_failed:
        print(f"   - {len(final_failed)} rats have no usable results")
        print(f"   - Consider re-running SLURM jobs for these rats")
    
    if len(final_completed) < 3:
        print(f"   - Only {len(final_completed)} completed rats found")
        print(f"   - Cross-rat analysis works best with 3+ rats")
    
    print(f"\n✅ Results recovery workflow completed!")

if __name__ == "__main__":
    main()