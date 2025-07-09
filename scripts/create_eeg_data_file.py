#!/usr/bin/env python3
"""
Create EEG Data File from MAT Files

This script reads all .mat files and creates the all_eeg_data.pkl file
needed for the cross-rats analysis.
"""

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
import argparse

def ascii_to_str(arr):
    """
    Convert ASCII array or bytes to string.
    
    Parameters:
    -----------
    arr : various types
        Input array, bytes, or string to convert
        
    Returns:
    --------
    str
        Converted string
    """
    if isinstance(arr, bytes):
        return arr.decode('utf-8')
    if isinstance(arr, np.ndarray):
        # Flatten and cast to uint8 (just in case)
        arr = arr.flatten()
        try:
            return ''.join(chr(c) for c in arr if c != 0)
        except Exception:
            # Fallback for char arrays stored as bytes
            if arr.dtype.char == 'S':
                return b''.join(arr).decode('utf-8')
    return str(arr)

def read_mat_file(file_path):
    """
    Read a single MAT file and extract EEG data.
    
    Parameters:
    -----------
    file_path : str
        Path to the MAT file
        
    Returns:
    --------
    dict
        Dictionary containing all extracted data
    """
    with h5py.File(file_path, 'r') as f:
        data_eeg = f['Data_eeg']
        
        def get_cell(col):
            ref = data_eeg[col, 0]
            arr = f[ref]
            data = arr[()]
            if isinstance(data, bytes):
                return data.decode('utf-8')
            elif isinstance(data, np.ndarray) and data.dtype.char == 'S':
                return b''.join(data).decode('utf-8')
            else:
                return np.array(data)
        
        # Extract columns
        rat_id = ascii_to_str(get_cell(0))
        session_date = ascii_to_str(get_cell(1))
        eeg = get_cell(2).T  # Transpose!
        eeg_time = get_cell(3).flatten().reshape(1, -1)  # 1D array
        velocity_trace = get_cell(4).flatten()  # 1D array
        velocity_time = get_cell(5).flatten()
        nm_peak_times = get_cell(6).flatten()
        nm_sizes = get_cell(7).flatten()
        iti_peak_times = get_cell(8).flatten()
        iti_sizes = get_cell(9).flatten()
        
        # Pack result
        return {
            'file_path': file_path,
            'rat_id': rat_id,
            'session_date': session_date,
            'eeg': eeg,
            'eeg_time': eeg_time,
            'velocity_trace': velocity_trace,
            'velocity_time': velocity_time,
            'nm_peak_times': nm_peak_times,
            'nm_sizes': nm_sizes,
            'iti_peak_times': iti_peak_times,
            'iti_sizes': iti_sizes
        }

def create_eeg_data_file(mat_dir, output_file, verbose=True):
    """
    Create the all_eeg_data.pkl file from MAT files.
    
    Parameters:
    -----------
    mat_dir : str
        Directory containing .mat files
    output_file : str
        Path to save the output .pkl file
    verbose : bool
        Whether to print progress information
    """
    if verbose:
        print("🧠 Creating EEG Data File from MAT Files")
        print("=" * 50)
        print(f"MAT files directory: {mat_dir}")
        print(f"Output file: {output_file}")
        print("=" * 50)
    
    # Find all MAT files
    mat_files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
    
    if not mat_files:
        print(f"❌ No .mat files found in {mat_dir}")
        return False
    
    if verbose:
        print(f"Found {len(mat_files)} MAT files")
    
    # Process files
    all_data = []
    failed_files = []
    
    progress_bar = tqdm(mat_files, desc='Loading MAT files') if verbose else mat_files
    
    for file_path in progress_bar:
        try:
            data = read_mat_file(file_path)
            all_data.append(data)
            
            if verbose and not hasattr(progress_bar, 'write'):
                print(f"✓ Loaded: {os.path.basename(file_path)} (rat {data['rat_id']}, session {data['session_date']})")
        
        except Exception as e:
            failed_files.append((file_path, str(e)))
            if verbose:
                print(f"❌ Failed to load {os.path.basename(file_path)}: {e}")
    
    if verbose:
        print(f"\n📊 Processing Summary:")
        print(f"  Successfully loaded: {len(all_data)} files")
        print(f"  Failed to load: {len(failed_files)} files")
        
        if failed_files:
            print(f"\nFailed files:")
            for file_path, error in failed_files:
                print(f"  - {os.path.basename(file_path)}: {error}")
    
    if not all_data:
        print("❌ No data successfully loaded!")
        return False
    
    # Save to pickle file
    if verbose:
        print(f"\n💾 Saving data to {output_file}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        
        if verbose:
            print(f"✅ Successfully saved {len(all_data)} sessions to {output_file}")
            
            # Get file size
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"File size: {file_size_mb:.1f} MB")
            
            # Show rat summary
            rat_ids = set()
            sessions_per_rat = {}
            for data in all_data:
                rat_id = data['rat_id']
                rat_ids.add(rat_id)
                sessions_per_rat[rat_id] = sessions_per_rat.get(rat_id, 0) + 1
            
            print(f"\n📋 Data Summary:")
            print(f"  Total sessions: {len(all_data)}")
            print(f"  Unique rats: {len(rat_ids)}")
            print(f"  Rat IDs: {sorted(rat_ids)}")
            print(f"  Sessions per rat: {dict(sorted(sessions_per_rat.items()))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create all_eeg_data.pkl from MAT files')
    parser.add_argument('--mat_dir', required=True, help='Directory containing .mat files')
    parser.add_argument('--output', default='data/processed/all_eeg_data.pkl', 
                       help='Output pickle file path (default: data/processed/all_eeg_data.pkl)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Check if MAT directory exists
    if not os.path.exists(args.mat_dir):
        print(f"❌ MAT directory does not exist: {args.mat_dir}")
        return 1
    
    # Create EEG data file
    verbose = not args.quiet
    success = create_eeg_data_file(args.mat_dir, args.output, verbose=verbose)
    
    if success:
        if verbose:
            print(f"\n🎉 EEG data file created successfully!")
            print(f"You can now use this file with the cross-rats analysis:")
            print(f"  python scripts/submit_array_jobs.py --data_path {args.output} ...")
        return 0
    else:
        print(f"\n❌ Failed to create EEG data file")
        return 1

if __name__ == "__main__":
    exit(main())