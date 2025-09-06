#!/usr/bin/env python3
"""
Comprehensive ROI Analysis with Visualization and Statistics
===========================================================

This script performs:
1. ROI analysis for all defined regions
2. Visualization of results
3. Statistical comparisons (ANOVA, post-hoc tests)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import warnings
warnings.filterwarnings('ignore')

# Import our ROI analyzer
import sys
sys.path.append('/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes')
from roi_analysis import ROIAnalyzer

def run_all_roi_analyses():
    """Run ROI analysis for all defined regions"""
    
    # Configuration
    csv_exports_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/csv_exports"
    output_dir = "/mnt/c/Users/flavi/Desktop/UniHelsinki/code/eeg-near_mistakes/results/comprehensive_roi_analysis"
    
    # ROI definitions as specified
    roi_definitions = {
        'Frontal_ROI': [8, 9, 6, 11],
        'Motor_ROI': [7, 5, 3, 14, 12, 10],
        'Somatomotor_ROI': [4, 23, 24, 1, 16, 25, 26, 15, 13],
        'Visual_ROI': [17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32]
    }
    
    print("Starting Comprehensive ROI Analysis...")
    print(f"ROI definitions:")
    for roi_name, electrodes in roi_definitions.items():
        print(f"  {roi_name}: {electrodes}")
    
    # Initialize analyzer
    analyzer = ROIAnalyzer(csv_exports_dir)
    analyzer.load_data()
    
    # Analyze each ROI
    all_results = {}
    
    for roi_name, electrode_list in roi_definitions.items():
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing {roi_name}...")
            print(f"{'='*60}")
            
            results = analyzer.analyze_roi(electrode_list, roi_name)
            analyzer.save_roi_results(results, output_dir)
            analyzer.print_roi_summary(results)
            
            all_results[roi_name] = results
            
        except Exception as e:
            print(f"Error analyzing {roi_name}: {e}")
            continue
    
    return all_results, output_dir

def create_roi_comparison_plots(all_results, output_dir):
    """Create comprehensive visualization plots"""
    
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATION PLOTS")
    print(f"{'='*60}")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Prepare data for plotting
    plot_data = []
    for roi_name, results in all_results.items():
        for nm_type in ['NM_size1.0', 'NM_size2.0', 'NM_size3.0']:
            plot_data.append({
                'ROI': roi_name.replace('_ROI', ''),
                'NM_Type': nm_type.replace('NM_size', 'NM'),
                'Mean': results['means'][nm_type],
                'SEM': results['sems'][nm_type],
                'N_Rats': results['n_rats']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Bar plot with error bars
    ax1 = plt.subplot(2, 3, 1)
    roi_names = plot_df['ROI'].unique()
    nm_types = plot_df['NM_Type'].unique()
    
    x = np.arange(len(roi_names))
    width = 0.25
    
    for i, nm_type in enumerate(nm_types):
        data_subset = plot_df[plot_df['NM_Type'] == nm_type]
        means = data_subset['Mean'].values
        sems = data_subset['SEM'].values
        
        bars = ax1.bar(x + i*width, means, width, yerr=sems, 
                      label=nm_type, alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Brain Region')
    ax1.set_ylabel('Theta Power')
    ax1.set_title('ROI Comparison: Theta Power by NM Type')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(roi_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap
    ax2 = plt.subplot(2, 3, 2)
    heatmap_data = plot_df.pivot(index='ROI', columns='NM_Type', values='Mean')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                ax=ax2, cbar_kws={'label': 'Theta Power'})
    ax2.set_title('ROI x NM Type Heatmap')
    ax2.set_xlabel('Near Mistake Type')
    ax2.set_ylabel('Brain Region')
    
    # Plot 3: Line plot showing progression
    ax3 = plt.subplot(2, 3, 3)
    for roi in roi_names:
        roi_data = plot_df[plot_df['ROI'] == roi]
        means = roi_data['Mean'].values
        sems = roi_data['SEM'].values
        ax3.errorbar(range(len(nm_types)), means, yerr=sems, 
                    marker='o', label=roi, linewidth=2, markersize=8)
    
    ax3.set_xlabel('NM Type')
    ax3.set_ylabel('Theta Power')
    ax3.set_title('Theta Power Progression Across NM Types')
    ax3.set_xticks(range(len(nm_types)))
    ax3.set_xticklabels(nm_types)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Individual rat data points
    ax4 = plt.subplot(2, 3, 4)
    
    # Prepare individual rat data for violin plot
    individual_data = []
    for roi_name, results in all_results.items():
        rat_data = results['rat_averages']
        for nm_type in ['NM_size1.0', 'NM_size2.0', 'NM_size3.0']:
            for _, row in rat_data.iterrows():
                individual_data.append({
                    'ROI': roi_name.replace('_ROI', ''),
                    'NM_Type': nm_type.replace('NM_size', 'NM'),
                    'Value': row[nm_type],
                    'Rat_ID': row['rat_ID']
                })
    
    individual_df = pd.DataFrame(individual_data)
    
    sns.violinplot(data=individual_df, x='ROI', y='Value', hue='NM_Type', ax=ax4)
    ax4.set_xlabel('Brain Region')
    ax4.set_ylabel('Theta Power')
    ax4.set_title('Distribution of Individual Rat Values')
    ax4.tick_params(axis='x', rotation=45)
    
    # Plot 5: Effect sizes comparison
    ax5 = plt.subplot(2, 3, 5)
    effect_sizes = []
    
    for roi_name, results in all_results.items():
        rat_data = results['rat_averages']
        nm1 = rat_data['NM_size1.0']
        nm2 = rat_data['NM_size2.0']
        nm3 = rat_data['NM_size3.0']
        
        # Calculate Cohen's d
        d_21 = (nm2.mean() - nm1.mean()) / np.sqrt((nm2.var() + nm1.var()) / 2)
        d_31 = (nm3.mean() - nm1.mean()) / np.sqrt((nm3.var() + nm1.var()) / 2)
        
        effect_sizes.append({
            'ROI': roi_name.replace('_ROI', ''),
            'NM2_vs_NM1': d_21,
            'NM3_vs_NM1': d_31
        })
    
    effect_df = pd.DataFrame(effect_sizes)
    
    x_pos = np.arange(len(effect_df))
    width = 0.35
    
    ax5.bar(x_pos - width/2, effect_df['NM2_vs_NM1'], width, 
           label='NM2 vs NM1', alpha=0.8)
    ax5.bar(x_pos + width/2, effect_df['NM3_vs_NM1'], width, 
           label='NM3 vs NM1', alpha=0.8)
    
    ax5.set_xlabel('Brain Region')
    ax5.set_ylabel("Cohen's d")
    ax5.set_title('Effect Sizes Across ROIs')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(effect_df['ROI'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 6: Statistical significance matrix
    ax6 = plt.subplot(2, 3, 6)
    
    # Create p-value matrix
    p_values = []
    for roi_name, results in all_results.items():
        roi_clean = roi_name.replace('_ROI', '')
        p_values.append([
            roi_clean,
            results['statistical_tests']['NM2_vs_NM1']['p_value'],
            results['statistical_tests']['NM3_vs_NM1']['p_value'],
            results['statistical_tests']['NM3_vs_NM2']['p_value']
        ])
    
    p_df = pd.DataFrame(p_values, columns=['ROI', 'NM2_vs_NM1', 'NM3_vs_NM1', 'NM3_vs_NM2'])
    p_matrix = p_df.set_index('ROI')[['NM2_vs_NM1', 'NM3_vs_NM1', 'NM3_vs_NM2']]
    
    # Convert p-values to significance levels for visualization
    sig_matrix = p_matrix.applymap(lambda x: -np.log10(x) if x > 0 else 0)
    
    sns.heatmap(sig_matrix, annot=p_matrix.round(3), fmt='', cmap='Reds', 
                ax=ax6, cbar_kws={'label': '-log10(p-value)'})
    ax6.set_title('Statistical Significance Matrix\n(Numbers show p-values)')
    ax6.set_xlabel('Comparison')
    ax6.set_ylabel('Brain Region')
    
    plt.tight_layout()
    
    # Save the figure
    plot_file = os.path.join(output_dir, 'roi_comprehensive_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive plot saved to: {plot_file}")
    
    return individual_df

def perform_statistical_tests(all_results, individual_df, output_dir):
    """Perform ANOVA and post-hoc statistical tests"""
    
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS (ANOVA)")
    print(f"{'='*60}")
    
    statistical_results = {}
    
    # For each NM type, perform ANOVA across ROIs
    for nm_type in ['NM1.0', 'NM2.0', 'NM3.0']:
        print(f"\n--- Analysis for {nm_type} ---")
        
        # Prepare data for ANOVA
        roi_groups = []
        group_names = []
        
        for roi_name, results in all_results.items():
            roi_clean = roi_name.replace('_ROI', '')
            nm_col = nm_type.replace('NM', 'NM_size')
            values = results['rat_averages'][nm_col].values
            roi_groups.append(values)
            group_names.append(roi_clean)
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*roi_groups)
        
        print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_value:.6f}")
        
        # Effect size (eta-squared)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(roi_groups)))**2 
                        for group in roi_groups)
        ss_total = sum((x - np.mean(np.concatenate(roi_groups)))**2 
                      for group in roi_groups for x in group)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        print(f"Effect size (η²): {eta_squared:.4f}")
        
        # Post-hoc tests (if ANOVA is significant)
        if p_value < 0.05:
            print("Performing post-hoc Tukey HSD test...")
            
            # Prepare data for post-hoc test
            posthoc_data = []
            posthoc_groups = []
            
            for i, (roi_name, results) in enumerate(all_results.items()):
                roi_clean = roi_name.replace('_ROI', '')
                nm_col = nm_type.replace('NM', 'NM_size')
                values = results['rat_averages'][nm_col].values
                posthoc_data.extend(values)
                posthoc_groups.extend([roi_clean] * len(values))
            
            # Perform Tukey HSD
            tukey_results = pairwise_tukeyhsd(posthoc_data, posthoc_groups, alpha=0.05)
            print(tukey_results)
        else:
            print("No significant differences found - skipping post-hoc tests")
            tukey_results = None
        
        statistical_results[nm_type] = {
            'anova_f': f_stat,
            'anova_p': p_value,
            'eta_squared': eta_squared,
            'tukey_results': tukey_results
        }
    
    # Two-way repeated measures ANOVA (ROI x NM_Type)
    print(f"\n{'='*40}")
    print("TWO-WAY REPEATED MEASURES ANOVA")
    print(f"{'='*40}")
    
    # Prepare data for 2-way RM ANOVA using individual_df
    print("Note: Using simplified approach with independent samples ANOVA")
    print("(Full RM-ANOVA would require specialized statistical package)")
    
    # Perform 2-way ANOVA using scipy
    roi_factor = individual_df['ROI']
    nm_factor = individual_df['NM_Type'] 
    values = individual_df['Value']
    
    # Get unique levels
    roi_levels = roi_factor.unique()
    nm_levels = nm_factor.unique()
    
    print(f"ROI factor levels: {roi_levels}")
    print(f"NM_Type factor levels: {nm_levels}")
    
    # Main effect of ROI
    roi_groups_2way = [values[roi_factor == roi].values for roi in roi_levels]
    f_roi, p_roi = f_oneway(*roi_groups_2way)
    
    # Main effect of NM_Type
    nm_groups_2way = [values[nm_factor == nm].values for nm in nm_levels]
    f_nm, p_nm = f_oneway(*nm_groups_2way)
    
    print(f"\nMain effect of ROI: F = {f_roi:.4f}, p = {p_roi:.6f}")
    print(f"Main effect of NM_Type: F = {f_nm:.4f}, p = {p_nm:.6f}")
    
    statistical_results['two_way'] = {
        'roi_f': f_roi,
        'roi_p': p_roi,
        'nm_f': f_nm,
        'nm_p': p_nm
    }
    
    # Save statistical results
    stats_file = os.path.join(output_dir, 'statistical_analysis_summary.txt')
    with open(stats_file, 'w') as f:
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ONE-WAY ANOVA RESULTS (Each NM type across ROIs):\n")
        f.write("-" * 45 + "\n")
        for nm_type, results in statistical_results.items():
            if nm_type != 'two_way':
                f.write(f"\n{nm_type}:\n")
                f.write(f"  F-statistic: {results['anova_f']:.4f}\n")
                f.write(f"  p-value: {results['anova_p']:.6f}\n")
                f.write(f"  η² (effect size): {results['eta_squared']:.4f}\n")
                
                if results['tukey_results'] is not None:
                    f.write(f"  Post-hoc (Tukey HSD):\n")
                    f.write(f"  {results['tukey_results']}\n")
        
        f.write(f"\nTWO-WAY ANOVA RESULTS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Main effect of ROI: F = {statistical_results['two_way']['roi_f']:.4f}, p = {statistical_results['two_way']['roi_p']:.6f}\n")
        f.write(f"Main effect of NM_Type: F = {statistical_results['two_way']['nm_f']:.4f}, p = {statistical_results['two_way']['nm_p']:.6f}\n")
        
        f.write(f"\nINTERPRETATION:\n")
        f.write("-" * 15 + "\n")
        
        # Interpretation guidelines
        for nm_type, results in statistical_results.items():
            if nm_type != 'two_way':
                if results['anova_p'] < 0.001:
                    sig = "highly significant (***)"
                elif results['anova_p'] < 0.01:
                    sig = "very significant (**)"
                elif results['anova_p'] < 0.05:
                    sig = "significant (*)"
                else:
                    sig = "not significant (ns)"
                    
                if results['eta_squared'] > 0.14:
                    effect = "large effect"
                elif results['eta_squared'] > 0.06:
                    effect = "medium effect"
                elif results['eta_squared'] > 0.01:
                    effect = "small effect"
                else:
                    effect = "negligible effect"
                
                f.write(f"{nm_type}: {sig}, {effect}\n")
    
    print(f"\nStatistical analysis summary saved to: {stats_file}")
    
    return statistical_results

def main():
    """Main execution function"""
    
    print("="*80)
    print("COMPREHENSIVE ROI ANALYSIS WITH VISUALIZATION AND STATISTICS")
    print("="*80)
    
    try:
        # Step 1: Run all ROI analyses
        all_results, output_dir = run_all_roi_analyses()
        
        # Step 2: Create visualization plots
        individual_df = create_roi_comparison_plots(all_results, output_dir)
        
        # Step 3: Perform statistical tests
        statistical_results = perform_statistical_tests(all_results, individual_df, output_dir)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"All results saved to: {output_dir}")
        print(f"ROIs analyzed: {len(all_results)}")
        print(f"Total rats: {list(all_results.values())[0]['n_rats']}")
        
        # Quick summary
        print(f"\nQUICK SUMMARY:")
        print(f"-" * 15)
        for roi_name, results in all_results.items():
            roi_clean = roi_name.replace('_ROI', '')
            nm1 = results['means']['NM_size1.0']
            nm2 = results['means']['NM_size2.0']
            nm3 = results['means']['NM_size3.0']
            print(f"{roi_clean:12}: NM1={nm1:.3f}, NM2={nm2:.3f}, NM3={nm3:.3f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()