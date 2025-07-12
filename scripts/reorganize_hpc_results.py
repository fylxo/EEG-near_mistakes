#!/usr/bin/env python3
"""
Script to reorganize HPC results folder structure for aggregation.

This script moves multi_session_results.pkl files from:
results/rat_XXX/rat_XXX_multi_session_mne/multi_session_results.pkl
to:
results/rat_XXX/multi_session_results.pkl

And removes the now-empty intermediate directories.
"""

import os
import shutil
import glob

def reorganize_results(results_path):
    """
    Reorganize the results directory structure.
    
    Parameters:
    -----------
    results_path : str
        Path to the results directory containing rat_* subdirectories
    """
    print(f"🔄 Reorganizing results in: {results_path}")
    
    # Find all rat_* directories
    rat_dirs = glob.glob(os.path.join(results_path, "rat_*"))
    
    if not rat_dirs:
        print("❌ No rat_* directories found!")
        return
    
    print(f"Found {len(rat_dirs)} rat directories")
    
    moved_count = 0
    
    for rat_dir in rat_dirs:
        rat_id = os.path.basename(rat_dir)
        
        # Look for the nested structure
        nested_dir = os.path.join(rat_dir, f"{rat_id}_multi_session_mne")
        source_file = os.path.join(nested_dir, "multi_session_results.pkl")
        target_file = os.path.join(rat_dir, "multi_session_results.pkl")
        
        if os.path.exists(source_file):
            print(f"  📁 {rat_id}: Moving multi_session_results.pkl")
            
            # Move the file
            shutil.move(source_file, target_file)
            
            # Remove the now-empty nested directory
            if os.path.exists(nested_dir) and not os.listdir(nested_dir):
                os.rmdir(nested_dir)
                print(f"    ✓ Removed empty directory: {rat_id}_multi_session_mne")
            
            moved_count += 1
            
        elif os.path.exists(target_file):
            print(f"  ✓ {rat_id}: Already reorganized")
            
        else:
            print(f"  ❌ {rat_id}: No multi_session_results.pkl found")
    
    print(f"\n✅ Reorganization complete!")
    print(f"Moved {moved_count} files")
    
    # List final structure
    print(f"\n📋 Final structure:")
    for rat_dir in sorted(rat_dirs):
        rat_id = os.path.basename(rat_dir)
        target_file = os.path.join(rat_dir, "multi_session_results.pkl")
        if os.path.exists(target_file):
            print(f"  ✓ {rat_id}/multi_session_results.pkl")
        else:
            print(f"  ❌ {rat_id}/ (no results file)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default to current directory if run from results/
        results_path = "."
    
    if not os.path.exists(results_path):
        print(f"❌ Results path does not exist: {results_path}")
        sys.exit(1)
    
    reorganize_results(results_path)