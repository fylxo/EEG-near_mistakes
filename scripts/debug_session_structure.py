#!/usr/bin/env python3
"""
Debug Session Structure

This script examines the structure of individual session result files
to understand what keys they contain.

Usage:
    python scripts/debug_session_structure.py --rat_dir results/rat_422/rat_422_multi_session_mne
"""

import os
import sys
import glob
import pickle
from pprint import pprint

def examine_session_file(session_file: str):
    """Examine the structure of a single session result file."""
    print(f"\n📁 Examining: {session_file}")
    
    try:
        with open(session_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"✓ Successfully loaded pickle file")
        print(f"📊 Type: {type(results)}")
        
        if isinstance(results, dict):
            print(f"🔑 Top-level keys: {list(results.keys())}")
            
            # Look for common keys we might expect
            expected_keys = ['normalized_windows', 'nm_windows', 'spectrograms', 'frequencies', 'roi_channels']
            found_keys = []
            missing_keys = []
            
            for key in expected_keys:
                if key in results:
                    found_keys.append(key)
                else:
                    missing_keys.append(key)
            
            if found_keys:
                print(f"✓ Found expected keys: {found_keys}")
            if missing_keys:
                print(f"❌ Missing expected keys: {missing_keys}")
            
            # Examine each key in detail
            for key, value in results.items():
                if isinstance(value, dict):
                    print(f"  📂 {key}: dict with {len(value)} items")
                    if len(value) <= 5:  # Show sub-keys if not too many
                        print(f"     Sub-keys: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"  📋 {key}: list with {len(value)} items")
                elif hasattr(value, 'shape'):  # numpy array
                    print(f"  🔢 {key}: array with shape {value.shape}")
                else:
                    print(f"  📄 {key}: {type(value)} = {str(value)[:100]}...")
        
        else:
            print(f"⚠️  Not a dictionary: {type(results)}")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug individual session result file structure')
    parser.add_argument('--rat_dir', required=True, help='Path to rat directory')
    parser.add_argument('--max_files', type=int, default=3, help='Maximum number of files to examine')
    
    args = parser.parse_args()
    
    print("🔍 Session Structure Debugger")
    print("=" * 60)
    print(f"Rat directory: {args.rat_dir}")
    print("=" * 60)
    
    # Find session result files
    session_files = glob.glob(os.path.join(args.rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl"))
    
    if not session_files:
        print(f"❌ No session result files found in {args.rat_dir}")
        sys.exit(1)
    
    print(f"Found {len(session_files)} session result files")
    
    # Examine first few files
    files_to_examine = session_files[:args.max_files]
    
    for session_file in files_to_examine:
        examine_session_file(session_file)
    
    if len(session_files) > args.max_files:
        print(f"\n... and {len(session_files) - args.max_files} more files")
    
    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print(f"If the files are missing 'normalized_windows' but have other keys like")
    print(f"'nm_windows' or 'spectrograms', we need to update the aggregation script")
    print(f"to use the correct key names.")

if __name__ == "__main__":
    main()