#!/usr/bin/env python3
"""
Replicate Previous Results Analysis - Fixed Version
==================================================

This script processes EEG theta power data to replicate previous results by:
1. For each rat: Average theta power across all sessions
2. For each electrode and NM type: Average across all rats

Output: Final averaged values for each electrode-NM_type combination
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_all_csv_data(csv_exports_dir):
    """
    Load all CSV files from the csv_exports directory
    
    Returns:
    - DataFrame with columns: rat_ID, session_number, electrode_number, NM_size1.0, NM_size2.0, NM_size3.0
    """
    all_data = []
    
    # Get all electrode directories using os.listdir to handle brackets properly
    electrode_dirs = []
    for item in os.listdir(csv_exports_dir):
        if item.startswith("electrode_[") and item.endswith("]"):
            electrode_dirs.append(os.path.join(csv_exports_dir, item))
    
    electrode_dirs.sort()
    print(f"Found {len(electrode_dirs)} electrode directories")
    
    for electrode_dir in electrode_dirs:
        # Extract electrode number from directory name
        dir_name = os.path.basename(electrode_dir)
        electrode_num = dir_name.split("electrode_[")[1].split("]")[0]
        
        # Get all CSV files in this electrode directory
        csv_files = []
        if os.path.exists(electrode_dir):
            for file in os.listdir(electrode_dir):
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(electrode_dir, file))
        
        print(f"Processing electrode {electrode_num}: {len(csv_files)} files")
        
        for csv_file in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(csv_file, sep=';')
                
                # Add electrode info if not present or fix it
                df['electrode_number'] = f"[{electrode_num}]"
                
                all_data.append(df)
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records loaded: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError("No data was loaded successfully")

def calculate_averages(df):
    """
    Calculate averages following the specified procedure:
    1. Average across sessions for each rat
    2. Average across rats for each electrode-NM_type combination
    """
    
    print("Step 1: Averaging across sessions for each rat...")
    
    # Group by rat_ID and electrode_number, then average across sessions
    rat_averages = df.groupby(['rat_ID', 'electrode_number'])[['NM_size1.0', 'NM_size2.0', 'NM_size3.0']].mean().reset_index()
    
    print(f"Rat-level averages calculated: {len(rat_averages)} records")
    print(f"Unique rats: {rat_averages['rat_ID'].nunique()}")
    print(f"Unique electrodes: {rat_averages['electrode_number'].nunique()}")
    
    print("\nStep 2: Averaging across rats for each electrode-NM_type combination...")
    
    # Group by electrode_number, then average across rats
    final_averages = rat_averages.groupby('electrode_number')[['NM_size1.0', 'NM_size2.0', 'NM_size3.0']].mean().reset_index()
    
    # Also calculate standard errors for each electrode-NM_type combination
    final_std_errors = rat_averages.groupby('electrode_number')[['NM_size1.0', 'NM_size2.0', 'NM_size3.0']].sem().reset_index()
    final_std_errors.columns = ['electrode_number', 'NM_size1.0_SE', 'NM_size2.0_SE', 'NM_size3.0_SE']
    
    # Merge averages and standard errors
    final_results = pd.merge(final_averages, final_std_errors, on='electrode_number')
    
    # Count number of rats contributing to each electrode
    rat_counts = rat_averages.groupby('electrode_number').size().reset_index(name='n_rats')
    final_results = pd.merge(final_results, rat_counts, on='electrode_number')
    
    print(f"Final averages calculated: {len(final_results)} electrodes")
    
    return final_results, rat_averages

def save_results(final_results, rat_averages, output_dir):
    """Save results to CSV files"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save final cross-rat averages
    final_output_path = os.path.join(output_dir, "cross_rat_averages_by_electrode.csv")
    final_results.to_csv(final_output_path, index=False)
    print(f"Final cross-rat averages saved to: {final_output_path}")
    
    # Save rat-level averages (intermediate results)
    rat_output_path = os.path.join(output_dir, "rat_level_averages_by_electrode.csv")
    rat_averages.to_csv(rat_output_path, index=False)
    print(f"Rat-level averages saved to: {rat_output_path}")
    
    # Create a summary table in a more readable format
    summary_df = final_results.copy()
    
    # Extract electrode number for sorting
    summary_df['electrode_num'] = summary_df['electrode_number'].str.extract(r'\[(\d+)\]').astype(int)
    summary_df = summary_df.sort_values('electrode_num')
    
    # Create formatted summary
    summary_output_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_output_path, 'w') as f:
        f.write("REPLICATED RESULTS - CROSS-RAT AVERAGES BY ELECTRODE\n")
        f.write("=" * 60 + "\n\n")
        f.write("Methodology:\n")
        f.write("1. For each rat: Average theta power across all sessions\n")
        f.write("2. For each electrode: Average across all rats\n\n")
        f.write("Results Format: Mean ± Standard Error (n_rats)\n")
        f.write("-" * 60 + "\n\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"Electrode {row['electrode_number']}:\n")
            f.write(f"  NM_size1.0: {row['NM_size1.0']:.6f} ± {row['NM_size1.0_SE']:.6f} (n={row['n_rats']})\n")
            f.write(f"  NM_size2.0: {row['NM_size2.0']:.6f} ± {row['NM_size2.0_SE']:.6f} (n={row['n_rats']})\n")
            f.write(f"  NM_size3.0: {row['NM_size3.0']:.6f} ± {row['NM_size3.0_SE']:.6f} (n={row['n_rats']})\n")
            f.write("\n")
    
    print(f"Summary report saved to: {summary_output_path}")

def print_data_overview(df):
    """Print overview of the loaded data"""
    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Unique rats: {df['rat_ID'].nunique()}")
    print(f"Unique electrodes: {df['electrode_number'].nunique()}")
    print(f"Unique sessions: {df['session_number'].nunique()}")
    
    print(f"\nRat distribution:")
    rat_counts = df['rat_ID'].value_counts().sort_index()
    for rat_id, count in rat_counts.items():
        sessions = df[df['rat_ID'] == rat_id]['session_number'].nunique()
        print(f"  Rat {rat_id}: {count} records ({sessions} sessions)")
    
    print(f"\nElectrode distribution:")
    electrode_counts = df['electrode_number'].value_counts().sort_index()
    for electrode, count in electrode_counts.head(10).items():
        print(f"  {electrode}: {count} records")
    if len(electrode_counts) > 10:
        print(f"  ... and {len(electrode_counts) - 10} more electrodes")
    
    print(f"\nData quality check:")
    print(f"  Missing NM_size1.0: {df['NM_size1.0'].isna().sum()}")
    print(f"  Missing NM_size2.0: {df['NM_size2.0'].isna().sum()}")
    print(f"  Missing NM_size3.0: {df['NM_size3.0'].isna().sum()}")

def main():
    """Main execution function"""
    
    # Set paths
    csv_exports_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/csv_exports"
    output_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/replicated_analysis"
    
    print("Starting analysis to replicate previous results...")
    print(f"Input directory: {csv_exports_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load all data
        print("\nLoading all CSV data...")
        df = load_all_csv_data(csv_exports_dir)
        
        # Print data overview
        print_data_overview(df)
        
        # Calculate averages
        print("\nCalculating averages...")
        final_results, rat_averages = calculate_averages(df)
        
        # Save results
        print("\nSaving results...")
        save_results(final_results, rat_averages, output_dir)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final results: {len(final_results)} electrode averages")
        print(f"Based on: {rat_averages['rat_ID'].nunique()} rats")
        print(f"Total sessions processed: {df['session_number'].nunique()}")
        
        # Display first few results as preview
        print("\nPreview of results (first 5 electrodes):")
        print("-" * 40)
        preview = final_results.head()
        for _, row in preview.iterrows():
            print(f"Electrode {row['electrode_number']}:")
            print(f"  NM1.0: {row['NM_size1.0']:.6f}, NM2.0: {row['NM_size2.0']:.6f}, NM3.0: {row['NM_size3.0']:.6f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()