#!/usr/bin/env python3
"""
Format Results as Matrix and Calculate Differences
=================================================

This script formats the cross-rat averages into a 32x3 matrix format
and calculates differences between NM types.
"""

import pandas as pd
import numpy as np

def load_and_format_results():
    """Load the cross-rat averages and format as requested matrix"""
    
    # Load the results
    results_file = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/replicated_analysis/cross_rat_averages_by_electrode.csv"
    df = pd.read_csv(results_file)
    
    # Extract electrode number for proper sorting
    df['electrode_num'] = df['electrode_number'].str.extract(r'\[(\d+)\]').astype(int)
    df_sorted = df.sort_values('electrode_num')
    
    # Create the matrix (32 rows x 3 columns)
    matrix_data = []
    for _, row in df_sorted.iterrows():
        electrode_row = [
            row['NM_size1.0'],
            row['NM_size2.0'], 
            row['NM_size3.0']
        ]
        matrix_data.append(electrode_row)
    
    matrix = np.array(matrix_data)
    return matrix, df_sorted

def format_matrix_matlab_style(matrix):
    """Format matrix in MATLAB style as requested"""
    
    output = "% 32x3 data matrix: rows = electrodes (1..32), cols = scenarios (type1..3)\n"
    output += "M = [\n"
    
    for i, row in enumerate(matrix):
        output += f" {row[0]:5.2f} {row[1]:5.2f} {row[2]:5.2f}"
        if i < len(matrix) - 1:
            output += "\n"
    
    output += "\n];"
    return output

def calculate_differences(matrix):
    """Calculate differences between NM types"""
    
    # Extract columns
    nm1 = matrix[:, 0]
    nm2 = matrix[:, 1] 
    nm3 = matrix[:, 2]
    
    # Calculate differences
    diff_21 = nm2 - nm1  # NM_size2.0 - NM_size1.0
    diff_31 = nm3 - nm1  # NM_size3.0 - NM_size1.0
    diff_32 = nm3 - nm2  # NM_size3.0 - NM_size2.0
    
    return diff_21, diff_31, diff_32

def format_difference_matrix(diff_array, title):
    """Format difference array as matrix"""
    
    output = f"% {title}\n"
    output += f"{title.split()[0].lower()}_diff = [\n"
    
    for i, val in enumerate(diff_array):
        output += f" {val:6.3f}"
        if i < len(diff_array) - 1:
            output += "\n"
    
    output += "\n];"
    return output

def main():
    """Main execution function"""
    
    print("Loading and formatting results...")
    matrix, df_sorted = load_and_format_results()
    
    print("Creating matrix format...")
    matrix_output = format_matrix_matlab_style(matrix)
    
    print("Calculating differences...")
    diff_21, diff_31, diff_32 = calculate_differences(matrix)
    
    # Format difference matrices
    diff_21_output = format_difference_matrix(diff_21, "NM2-NM1 differences (NM_size2.0 - NM_size1.0)")
    diff_31_output = format_difference_matrix(diff_31, "NM3-NM1 differences (NM_size3.0 - NM_size1.0)")  
    diff_32_output = format_difference_matrix(diff_32, "NM3-NM2 differences (NM_size3.0 - NM_size2.0)")
    
    # Create complete output
    complete_output = matrix_output + "\n\n" + diff_21_output + "\n\n" + diff_31_output + "\n\n" + diff_32_output
    
    # Save to file
    output_file = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/replicated_analysis/matrix_format_results.txt"
    with open(output_file, 'w') as f:
        f.write(complete_output)
    
    print(f"Results saved to: {output_file}")
    
    # Print to console
    print("\n" + "="*60)
    print("MATRIX FORMAT RESULTS")
    print("="*60)
    print(complete_output)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Mean NM_size1.0: {matrix[:, 0].mean():.4f} ± {matrix[:, 0].std():.4f}")
    print(f"Mean NM_size2.0: {matrix[:, 1].mean():.4f} ± {matrix[:, 1].std():.4f}")
    print(f"Mean NM_size3.0: {matrix[:, 2].mean():.4f} ± {matrix[:, 2].std():.4f}")
    
    print(f"\nMean differences:")
    print(f"NM2-NM1: {diff_21.mean():.4f} ± {diff_21.std():.4f}")
    print(f"NM3-NM1: {diff_31.mean():.4f} ± {diff_31.std():.4f}")
    print(f"NM3-NM2: {diff_32.mean():.4f} ± {diff_32.std():.4f}")
    
    # Identify electrodes with largest differences
    print(f"\nElectrodes with largest NM3-NM1 differences:")
    top_indices = np.argsort(diff_31)[-5:][::-1]  # Top 5
    for idx in top_indices:
        electrode_num = idx + 1
        print(f"  Electrode {electrode_num}: {diff_31[idx]:.4f}")
    
    print(f"\nElectrodes with smallest NM3-NM1 differences:")
    bottom_indices = np.argsort(diff_31)[:5]  # Bottom 5
    for idx in bottom_indices:
        electrode_num = idx + 1
        print(f"  Electrode {electrode_num}: {diff_31[idx]:.4f}")

if __name__ == "__main__":
    main()