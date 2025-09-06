#!/usr/bin/env python3
"""
ROI Analysis for EEG Near Mistakes Data
=======================================

This script implements ROI-based analysis following the exact methodology described:
1. For each session and electrode: trial-averaged power (already computed in our CSVs)
2. For each session: average across electrodes in the ROI
3. For each rat: average across sessions
4. Across rats: calculate mean and SEM

The script is flexible to handle any combination of electrodes for different ROIs.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from scipy import stats

class ROIAnalyzer:
    def __init__(self, csv_exports_dir):
        """
        Initialize ROI analyzer
        
        Parameters:
        - csv_exports_dir: Path to the CSV exports directory
        """
        self.csv_exports_dir = csv_exports_dir
        self.data = None
        
    def load_data(self):
        """Load all CSV data from the exports directory"""
        
        print("Loading all CSV data...")
        all_data = []
        
        # Get all electrode directories
        electrode_dirs = []
        for item in os.listdir(self.csv_exports_dir):
            if item.startswith("electrode_[") and item.endswith("]"):
                electrode_dirs.append(os.path.join(self.csv_exports_dir, item))
        
        electrode_dirs.sort()
        print(f"Found {len(electrode_dirs)} electrode directories")
        
        for electrode_dir in electrode_dirs:
            # Extract electrode number from directory name
            dir_name = os.path.basename(electrode_dir)
            electrode_num = int(dir_name.split("electrode_[")[1].split("]")[0])
            
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
                    
                    # Add electrode number as integer for easier filtering
                    df['electrode_num'] = electrode_num
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue
        
        # Combine all data
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"Total records loaded: {len(self.data)}")
            print(f"Unique rats: {self.data['rat_ID'].nunique()}")
            print(f"Unique electrodes: {self.data['electrode_num'].nunique()}")
        else:
            raise ValueError("No data was loaded successfully")
    
    def analyze_roi(self, roi_electrodes, roi_name="ROI"):
        """
        Analyze a specific ROI following the described methodology
        
        Parameters:
        - roi_electrodes: List of electrode numbers (e.g., [8, 9, 6, 11])
        - roi_name: Name for the ROI (for output files)
        
        Returns:
        - Dictionary with results including means, SEMs, and detailed data
        """
        
        print(f"\n" + "="*50)
        print(f"ANALYZING {roi_name.upper()}")
        print("="*50)
        print(f"ROI electrodes: {roi_electrodes}")
        
        # Filter data for the ROI electrodes
        roi_data = self.data[self.data['electrode_num'].isin(roi_electrodes)].copy()
        
        if len(roi_data) == 0:
            raise ValueError(f"No data found for ROI electrodes: {roi_electrodes}")
        
        print(f"ROI data records: {len(roi_data)}")
        print(f"Available electrodes in ROI: {sorted(roi_data['electrode_num'].unique())}")
        
        # Step 1: For each session, average across electrodes in the ROI
        print("\nStep 1: Averaging across electrodes within each session...")
        
        session_roi_averages = roi_data.groupby(['rat_ID', 'session_number'])[
            ['NM_size1.0', 'NM_size2.0', 'NM_size3.0']
        ].mean().reset_index()
        
        print(f"Session-ROI averages: {len(session_roi_averages)} records")
        
        # Step 2: For each rat, average across sessions
        print("Step 2: Averaging across sessions for each rat...")
        
        rat_roi_averages = session_roi_averages.groupby('rat_ID')[
            ['NM_size1.0', 'NM_size2.0', 'NM_size3.0']
        ].mean().reset_index()
        
        print(f"Rat-ROI averages: {len(rat_roi_averages)} records")
        print(f"Rats in ROI analysis: {sorted(rat_roi_averages['rat_ID'].unique())}")
        
        # Step 3: Calculate cross-rat statistics
        print("Step 3: Calculating cross-rat statistics...")
        
        # Calculate means
        means = {
            'NM_size1.0': rat_roi_averages['NM_size1.0'].mean(),
            'NM_size2.0': rat_roi_averages['NM_size2.0'].mean(),
            'NM_size3.0': rat_roi_averages['NM_size3.0'].mean()
        }
        
        # Calculate standard errors of the mean (SEM)
        sems = {
            'NM_size1.0': rat_roi_averages['NM_size1.0'].sem(),
            'NM_size2.0': rat_roi_averages['NM_size2.0'].sem(),
            'NM_size3.0': rat_roi_averages['NM_size3.0'].sem()
        }
        
        # Calculate standard deviations
        stds = {
            'NM_size1.0': rat_roi_averages['NM_size1.0'].std(),
            'NM_size2.0': rat_roi_averages['NM_size2.0'].std(),
            'NM_size3.0': rat_roi_averages['NM_size3.0'].std()
        }
        
        # Sample sizes
        n_rats = len(rat_roi_averages)
        n_sessions_per_rat = session_roi_averages.groupby('rat_ID').size()
        total_sessions = len(session_roi_averages)
        
        # Statistical tests between conditions
        # Paired t-tests between NM conditions
        t_stat_21, p_val_21 = stats.ttest_rel(
            rat_roi_averages['NM_size2.0'], 
            rat_roi_averages['NM_size1.0']
        )
        
        t_stat_31, p_val_31 = stats.ttest_rel(
            rat_roi_averages['NM_size3.0'], 
            rat_roi_averages['NM_size1.0']
        )
        
        t_stat_32, p_val_32 = stats.ttest_rel(
            rat_roi_averages['NM_size3.0'], 
            rat_roi_averages['NM_size2.0']
        )
        
        # Prepare results dictionary
        results = {
            'roi_name': roi_name,
            'roi_electrodes': roi_electrodes,
            'means': means,
            'sems': sems,
            'stds': stds,
            'n_rats': n_rats,
            'n_sessions_per_rat': n_sessions_per_rat,
            'total_sessions': total_sessions,
            'statistical_tests': {
                'NM2_vs_NM1': {'t_stat': t_stat_21, 'p_value': p_val_21},
                'NM3_vs_NM1': {'t_stat': t_stat_31, 'p_value': p_val_31},
                'NM3_vs_NM2': {'t_stat': t_stat_32, 'p_value': p_val_32}
            },
            'rat_averages': rat_roi_averages,
            'session_averages': session_roi_averages
        }
        
        return results
    
    def save_roi_results(self, results, output_dir):
        """Save ROI analysis results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        roi_name = results['roi_name']
        
        # Save rat-level averages
        rat_file = os.path.join(output_dir, f"{roi_name}_rat_averages.csv")
        results['rat_averages'].to_csv(rat_file, index=False)
        print(f"Rat averages saved to: {rat_file}")
        
        # Save session-level averages
        session_file = os.path.join(output_dir, f"{roi_name}_session_averages.csv")
        results['session_averages'].to_csv(session_file, index=False)
        print(f"Session averages saved to: {session_file}")
        
        # Create detailed summary report
        summary_file = os.path.join(output_dir, f"{roi_name}_analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"{roi_name.upper()} ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ROI Configuration:\n")
            f.write(f"- ROI Name: {roi_name}\n")
            f.write(f"- Electrodes: {results['roi_electrodes']}\n")
            f.write(f"- Number of rats: {results['n_rats']}\n")
            f.write(f"- Total sessions: {results['total_sessions']}\n")
            f.write(f"- Sessions per rat: {results['n_sessions_per_rat'].mean():.1f} ± {results['n_sessions_per_rat'].std():.1f}\n\n")
            
            f.write("Methodology:\n")
            f.write("1. For each session: Average across ROI electrodes\n")
            f.write("2. For each rat: Average across sessions\n")
            f.write("3. Across rats: Calculate mean ± SEM\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"NM_size1.0: {results['means']['NM_size1.0']:.6f} ± {results['sems']['NM_size1.0']:.6f}\n")
            f.write(f"NM_size2.0: {results['means']['NM_size2.0']:.6f} ± {results['sems']['NM_size2.0']:.6f}\n")
            f.write(f"NM_size3.0: {results['means']['NM_size3.0']:.6f} ± {results['sems']['NM_size3.0']:.6f}\n\n")
            
            f.write("STATISTICAL COMPARISONS (Paired t-tests):\n")
            f.write("-" * 40 + "\n")
            
            for comparison, stats_info in results['statistical_tests'].items():
                t_stat = stats_info['t_stat']
                p_val = stats_info['p_value']
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                f.write(f"{comparison}: t = {t_stat:.4f}, p = {p_val:.6f} {significance}\n")
            
            f.write("\nEffect Sizes (Cohen's d):\n")
            f.write("-" * 25 + "\n")
            
            # Calculate Cohen's d for each comparison
            nm1 = results['rat_averages']['NM_size1.0']
            nm2 = results['rat_averages']['NM_size2.0']
            nm3 = results['rat_averages']['NM_size3.0']
            
            d_21 = (nm2.mean() - nm1.mean()) / np.sqrt(((nm2.var() + nm1.var()) / 2))
            d_31 = (nm3.mean() - nm1.mean()) / np.sqrt(((nm3.var() + nm1.var()) / 2))
            d_32 = (nm3.mean() - nm2.mean()) / np.sqrt(((nm3.var() + nm2.var()) / 2))
            
            f.write(f"NM2 vs NM1: d = {d_21:.4f}\n")
            f.write(f"NM3 vs NM1: d = {d_31:.4f}\n")
            f.write(f"NM3 vs NM2: d = {d_32:.4f}\n")
            
        print(f"Analysis summary saved to: {summary_file}")
        
        return summary_file
    
    def print_roi_summary(self, results):
        """Print a summary of ROI results to console"""
        
        roi_name = results['roi_name']
        means = results['means']
        sems = results['sems']
        
        print(f"\n{roi_name.upper()} RESULTS SUMMARY:")
        print("-" * 40)
        print(f"NM_size1.0: {means['NM_size1.0']:.6f} ± {sems['NM_size1.0']:.6f}")
        print(f"NM_size2.0: {means['NM_size2.0']:.6f} ± {sems['NM_size2.0']:.6f}")
        print(f"NM_size3.0: {means['NM_size3.0']:.6f} ± {sems['NM_size3.0']:.6f}")
        
        print(f"\nStatistical tests:")
        for comparison, stats_info in results['statistical_tests'].items():
            p_val = stats_info['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{comparison}: p = {p_val:.6f} {significance}")

def main():
    """Main execution function"""
    
    # Configuration
    csv_exports_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/csv_exports"
    output_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/roi_analysis"
    
    # ROI definitions - can be easily extended
    roi_definitions = {
        'Frontal_ROI': [8, 9, 6, 11],  # Main hypothesis area
        # Add more ROIs here as needed:
        # 'Parietal_ROI': [1, 2, 3, 4],
        # 'Central_ROI': [12, 13, 14, 15],
        # etc.
    }
    
    print("Starting ROI Analysis...")
    print(f"Input directory: {csv_exports_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ROIAnalyzer(csv_exports_dir)
    analyzer.load_data()
    
    # Analyze each ROI
    all_results = {}
    
    for roi_name, electrode_list in roi_definitions.items():
        try:
            print(f"\nAnalyzing {roi_name}...")
            results = analyzer.analyze_roi(electrode_list, roi_name)
            analyzer.save_roi_results(results, output_dir)
            analyzer.print_roi_summary(results)
            
            all_results[roi_name] = results
            
        except Exception as e:
            print(f"Error analyzing {roi_name}: {e}")
            continue
    
    # Create a comparative summary across ROIs
    if len(all_results) > 1:
        print(f"\n" + "="*60)
        print("COMPARATIVE SUMMARY ACROSS ROIS")
        print("="*60)
        
        comparative_file = os.path.join(output_dir, "roi_comparative_summary.txt")
        with open(comparative_file, 'w') as f:
            f.write("COMPARATIVE SUMMARY ACROSS ROIS\n")
            f.write("=" * 40 + "\n\n")
            
            for roi_name, results in all_results.items():
                f.write(f"{roi_name}:\n")
                f.write(f"  NM1: {results['means']['NM_size1.0']:.4f} ± {results['sems']['NM_size1.0']:.4f}\n")
                f.write(f"  NM2: {results['means']['NM_size2.0']:.4f} ± {results['sems']['NM_size2.0']:.4f}\n")
                f.write(f"  NM3: {results['means']['NM_size3.0']:.4f} ± {results['sems']['NM_size3.0']:.4f}\n")
                f.write("\n")
        
        print(f"Comparative summary saved to: {comparative_file}")
    
    print(f"\n" + "="*60)
    print("ROI ANALYSIS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()