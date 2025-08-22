#!/usr/bin/env python3
"""
Multi-Electrode Cross-Rats Analysis Runner

This script runs the cross-rats theta analysis for multiple electrodes sequentially.
For each electrode, it:
1. Runs nm_theta_cross_rats analysis
2. Moves results to electrode-specific subfolder  
3. Cleans up temporary cross_rats folder
4. Proceeds to next electrode

Usage:
    python run_multi_electrode_analysis.py [--electrodes 1,2,3,4] [--pkl_path path/to/data.pkl]
"""

import os
import sys
import shutil
import argparse
import time
from typing import List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'core'))

# Import the main analysis function
from nm_theta_cross_rats import run_cross_rats_analysis

def move_cross_rats_results_to_electrode_folder(electrode: str, results_base_dir: str = "results") -> bool:
    """
    Move cross_rats main results to electrode-specific subfolder.
    Moves files directly from results/ directory, not from cross_rats/ subfolder.
    
    Parameters:
    -----------
    electrode : str
        Electrode number/specification (e.g., "2", "8,9,6,11")
    results_base_dir : str
        Base results directory (default: "results")
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    electrode_dest = os.path.join(results_base_dir, "cross_rats_results", f"electrode_{electrode}")
    
    # Create electrode destination folder
    os.makedirs(electrode_dest, exist_ok=True)
    
    # Debug: List what's actually in the results directory
    print(f"🔍 Debug: Contents of {results_base_dir}:")
    try:
        all_items = os.listdir(results_base_dir)
        for item in all_items:
            item_path = os.path.join(results_base_dir, item)
            if os.path.isfile(item_path):
                print(f"  📄 File: {item}")
            else:
                print(f"  📁 Directory: {item}")
    except Exception as e:
        print(f"  ❌ Error listing contents: {e}")
    
    # Files to move (look directly in results directory)
    files_to_move = [
        "cross_rats_aggregated_results.pkl",
        "cross_rats_spectrograms.png", 
        "cross_rats_summary.json"
    ]
    
    # Find HTML files (interactive plots) in results directory
    try:
        html_files = [f for f in os.listdir(results_base_dir) if f.endswith('.html') and 'interactive_nm_size' in f]
        files_to_move.extend(html_files)
        print(f"🔍 Found {len(html_files)} HTML files: {html_files}")
    except Exception as e:
        print(f"❌ Error finding HTML files: {e}")
    
    moved_files = []
    for filename in files_to_move:
        source_path = os.path.join(results_base_dir, filename)
        dest_path = os.path.join(electrode_dest, filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            moved_files.append(filename)
            print(f"  ✓ Moved: {filename}")
        else:
            print(f"  ⚠️ File not found: {filename}")
    
    print(f"📁 Results moved to: {electrode_dest}")
    print(f"   Files moved: {len(moved_files)}")
    
    return len(moved_files) > 0

def cleanup_results_folder(results_base_dir: str = "results") -> bool:
    """
    Clean up ALL content in results directory except csv_exports and cross_rats_results.
    This prepares for the next electrode analysis.
    
    Parameters:
    -----------
    results_base_dir : str
        Base results directory (default: "results")
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Items to preserve (don't delete)
    preserve_items = {"csv_exports", "cross_rats_results"}
    
    if not os.path.exists(results_base_dir):
        print(f"ℹ️  Results directory doesn't exist: {results_base_dir}")
        return True
    
    print(f"🧹 Cleaning up {results_base_dir} (preserving: {preserve_items})")
    
    try:
        items_to_remove = []
        all_items = os.listdir(results_base_dir)
        
        for item in all_items:
            if item not in preserve_items:
                item_path = os.path.join(results_base_dir, item)
                items_to_remove.append((item, item_path))
        
        print(f"   Items to remove: {len(items_to_remove)}")
        
        for item_name, item_path in items_to_remove:
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"   ✓ Removed file: {item_name}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"   ✓ Removed directory: {item_name}")
        
        print(f"🧹 Cleanup complete - removed {len(items_to_remove)} items")
        return True
        
    except Exception as e:
        print(f"❌ Failed to cleanup {results_base_dir}: {e}")
        return False

def run_electrode_analysis(electrode: str, pkl_path: str, results_base_dir: str = "results", verbose: bool = True) -> bool:
    """
    Run complete analysis for one electrode: analysis + file organization + cleanup.
    
    Parameters:
    -----------
    electrode : str
        Electrode specification (e.g., "2", "8,9,6,11")
    pkl_path : str
        Path to the data pickle file
    results_base_dir : str
        Base results directory
    verbose : bool
        Enable verbose output
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print(f"\n" + "="*80)
    print(f"🔬 ANALYZING ELECTRODE: {electrode}")
    print(f"="*80)
    
    try:
        # Step 1: Run cross-rats analysis
        print(f"\n📊 Step 1: Running cross-rats analysis for electrode {electrode}...")
        
        results = run_cross_rats_analysis(
            pkl_path=pkl_path,
            roi=electrode,
            freq_min=3.0,
            freq_max=7.0,
            n_freqs=30,
            window_duration=2.0,
            save_path=results_base_dir,
            show_plots=False,  # No plots for batch processing
            verbose=verbose,
            cleanup_intermediate_files=True  # Clean up session folders
        )
        
        if not results:
            print(f"❌ Analysis failed for electrode {electrode}")
            return False
        
        print(f"✅ Analysis completed for electrode {electrode}")
        
        # Step 2: Move results to electrode-specific folder
        print(f"\n📁 Step 2: Moving results to electrode folder...")
        move_success = move_cross_rats_results_to_electrode_folder(electrode, results_base_dir)
        
        if not move_success:
            print(f"⚠️  Warning: Failed to move some results for electrode {electrode}")
        
        # Step 3: Cleanup ALL results folder content (except csv_exports and cross_rats_results)
        print(f"\n🧹 Step 3: Cleaning up results folder for next electrode...")
        cleanup_success = cleanup_results_folder(results_base_dir)
        
        if not cleanup_success:
            print(f"⚠️  Warning: Failed to cleanup cross_rats folder")
        
        print(f"✅ Electrode {electrode} processing complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing electrode {electrode}: {e}")
        return False

def main():
    """Main function to run multi-electrode analysis."""
    
    # =============================================================================
    # CONFIGURATION - Edit these values to run from IDE
    # =============================================================================
    
    # Default settings (used when running from IDE)
    DEFAULT_PKL_PATH = "data/processed/all_eeg_data.pkl" 
    DEFAULT_ELECTRODES = [str(i) for i in range(1, 33)]  # Electrodes 1-32
    DEFAULT_RESULTS_DIR = "results"
    DEFAULT_VERBOSE = True  # Set to False for less output
    
    # =============================================================================
    # Command line argument parsing (optional - fallback to defaults if no args)
    # =============================================================================
    
    parser = argparse.ArgumentParser(description='Run cross-rats analysis for multiple electrodes')
    
    parser.add_argument('--electrodes', default=",".join(DEFAULT_ELECTRODES),
                       help='Comma-separated list of electrodes to analyze (default: "1,2,3,...,32")')
    parser.add_argument('--pkl_path', default=DEFAULT_PKL_PATH,
                       help=f'Path to the data pickle file (default: "{DEFAULT_PKL_PATH}")')
    parser.add_argument('--results_dir', default=DEFAULT_RESULTS_DIR,
                       help=f'Base results directory (default: "{DEFAULT_RESULTS_DIR}")')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbose output')
    
    # Try to parse args, but don't fail if running from IDE
    try:
        args = parser.parse_args()
        electrode_list = [e.strip() for e in args.electrodes.split(',')]
        pkl_path = args.pkl_path
        results_dir = args.results_dir
        verbose = not args.quiet
    except SystemExit:
        # Running from IDE - use defaults
        print("📝 Running from IDE - using default configuration")
        electrode_list = DEFAULT_ELECTRODES
        pkl_path = DEFAULT_PKL_PATH
        results_dir = DEFAULT_RESULTS_DIR
        verbose = DEFAULT_VERBOSE
    
    print(f"🚀 Multi-Electrode Cross-Rats Analysis")
    print(f"   Data file: {pkl_path}")
    print(f"   Electrodes: {electrode_list}")
    print(f"   Results dir: {results_dir}")
    print(f"   Total electrodes: {len(electrode_list)}")
    
    # Process each electrode
    start_time = time.time()
    successful_electrodes = []
    failed_electrodes = []
    
    for i, electrode in enumerate(electrode_list, 1):
        print(f"\n🔄 Processing electrode {i}/{len(electrode_list)}: {electrode}")
        
        electrode_start_time = time.time()
        success = run_electrode_analysis(
            electrode=electrode,
            pkl_path=pkl_path,
            results_base_dir=results_dir,
            verbose=verbose
        )
        electrode_end_time = time.time()
        
        if success:
            successful_electrodes.append(electrode)
            print(f"✅ Electrode {electrode} completed in {electrode_end_time - electrode_start_time:.1f}s")
        else:
            failed_electrodes.append(electrode)
            print(f"❌ Electrode {electrode} failed")
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "="*80)
    print(f"📋 FINAL SUMMARY")
    print(f"="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful electrodes ({len(successful_electrodes)}): {successful_electrodes}")
    
    if failed_electrodes:
        print(f"Failed electrodes ({len(failed_electrodes)}): {failed_electrodes}")
        print(f"❌ {len(failed_electrodes)}/{len(electrode_list)} electrodes failed")
    else:
        print(f"✅ All {len(successful_electrodes)} electrodes processed successfully!")
    
    print(f"\n📁 Results saved in:")
    for electrode in successful_electrodes:
        electrode_dir = os.path.join(results_dir, "cross_rats_results", f"electrode_{electrode}")
        print(f"   - {electrode_dir}")

if __name__ == "__main__":
    main()