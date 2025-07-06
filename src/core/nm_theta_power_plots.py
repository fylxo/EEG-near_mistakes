#!/usr/bin/env python3
"""
NM Theta Power Plots

This script loads existing cross-rats results and creates theta band power plots
showing mean ± 1 SE for different near-mistake types across brain regions.

Author: Generated for EEG near-mistake theta power analysis
"""

import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse

# Add parent directory to path for config imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import AnalysisConfig, DataConfig, PlottingConfig


def extract_theta_power(spectrograms: np.ndarray, frequencies: np.ndarray, 
                        theta_range: Tuple[float, float] = None,
                        time_window: Optional[Tuple[float, float]] = None,
                        window_times: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract theta band power from spectrograms.
    
    Parameters:
    -----------
    spectrograms : np.ndarray
        Array of spectrograms with shape (n_rats, n_freqs, n_times)
    frequencies : np.ndarray
        Frequency values for each frequency bin
    theta_range : Tuple[float, float], optional
        Frequency range for theta band (default: from AnalysisConfig)
    time_window : Optional[Tuple[float, float]]
        Time window to extract power from (default: None for all times)
    window_times : Optional[np.ndarray]
        Time values for each time bin (needed if time_window is specified)
        
    Returns:
    --------
    theta_power : np.ndarray
        Theta power values with shape (n_rats,) - one value per rat
    """
    # Apply configuration default if not provided
    if theta_range is None:
        theta_range = AnalysisConfig.get_theta_range()
    
    # Find frequency indices for theta band
    theta_mask = (frequencies >= theta_range[0]) & (frequencies <= theta_range[1])
    
    if not np.any(theta_mask):
        raise ValueError(f"No frequencies found in theta range {theta_range}")
    
    # Extract theta frequencies
    theta_spectrograms = spectrograms[:, theta_mask, :]
    
    # Average across theta frequencies
    theta_power_time_series = np.mean(theta_spectrograms, axis=1)  # Shape: (n_rats, n_times)
    
    # Extract specific time window if requested
    if time_window is not None and window_times is not None:
        time_mask = (window_times >= time_window[0]) & (window_times <= time_window[1])
        if np.any(time_mask):
            theta_power_time_series = theta_power_time_series[:, time_mask]
    
    # Average across time to get single power value per rat
    theta_power = np.mean(theta_power_time_series, axis=1)  # Shape: (n_rats,)
    
    return theta_power


def create_theta_power_plots(results: Dict, save_path: str, 
                           theta_range: Tuple[float, float] = None,
                           time_window: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None,
                           verbose: bool = True):
    """
    Create theta band power plots with mean ± 1 SE for different NM types.
    
    Parameters:
    -----------
    results : Dict
        Cross-rats aggregated results
    save_path : str
        Directory to save plots
    theta_range : Tuple[float, float], optional
        Frequency range for theta band (default: from AnalysisConfig)
    time_window : Optional[Tuple[float, float]]
        Time window to extract power from (default: None for all times)
    ylim : Optional[Tuple[float, float]], optional
        Y-axis limits for the plot (default: from PlottingConfig)
    verbose : bool
        Whether to print progress information
    """
    # Apply configuration defaults
    if theta_range is None:
        theta_range = AnalysisConfig.get_theta_range()
    if ylim is None:
        ylim = PlottingConfig.get_theta_power_ylim()
    
    if verbose:
        print(f"\n📊 Creating theta power plots (mean ± 1 SE)")
        print(f"Theta range: {theta_range[0]}-{theta_range[1]} Hz")
        if time_window:
            print(f"Time window: {time_window[0]}-{time_window[1]} s")
        print("=" * 60)
    
    frequencies = results['frequencies']
    averaged_windows = results['averaged_windows']
    
    # Extract theta power for each NM type
    nm_types = []
    theta_powers = []
    
    for nm_size, window_data in averaged_windows.items():
        nm_types.append(float(nm_size))
        
        # Get individual rat spectrograms
        individual_spectrograms = window_data['individual_spectrograms']
        window_times = window_data['window_times']
        
        # Extract theta power for each rat
        rat_theta_powers = extract_theta_power(
            spectrograms=individual_spectrograms,
            frequencies=frequencies,
            theta_range=theta_range,
            time_window=time_window,
            window_times=window_times
        )
        
        theta_powers.append(rat_theta_powers)
    
    # Calculate mean and standard error for each NM type
    nm_means = []
    nm_ses = []
    
    for powers in theta_powers:
        mean_power = np.mean(powers)
        se_power = np.std(powers, ddof=1) / np.sqrt(len(powers))  # Standard error
        nm_means.append(mean_power)
        nm_ses.append(se_power)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=PlottingConfig.get_figure_size('default'))
    
    # Create bar plot with error bars
    bars = ax.bar(nm_types, nm_means, yerr=nm_ses, 
                  capsize=PlottingConfig.BAR_CAPSIZE, 
                  color=PlottingConfig.BAR_COLOR_DEFAULT, 
                  alpha=PlottingConfig.BAR_ALPHA, 
                  edgecolor=PlottingConfig.BAR_EDGE_COLOR)
    
    # Customize the plot
    ax.set_xlabel('Near-Mistake Type', fontsize=PlottingConfig.AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Theta Power (Z-score)', fontsize=PlottingConfig.AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Theta Band Power by Near-Mistake Type\n'
                f'Mean ± 1 SE (n={results["n_rats"]} rats)', 
                fontsize=PlottingConfig.TITLE_FONTSIZE)
    
    # Set x-axis ticks to show NM types
    ax.set_xticks(nm_types)
    ax.set_xticklabels([f'NM {int(nm)}' for nm in nm_types])
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add value labels on bars
    for bar, mean_val, se_val in zip(bars, nm_means, nm_ses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + se_val + PlottingConfig.VALUE_LABEL_OFFSET,
                f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=PlottingConfig.VALUE_LABEL_FONTSIZE)
    
    # Add grid for better readability
    if PlottingConfig.GRID_ENABLE:
        ax.grid(True, alpha=PlottingConfig.GRID_ALPHA)
    
    # Add ROI and frequency info using configuration
    info_text = PlottingConfig.get_info_box_text(
        roi=results['roi_specification'],
        freq_range=theta_range,
        time_window=time_window,
        n_rats=results['n_rats']
    )
    
    ax.text(PlottingConfig.INFO_BOX_POSITION[0], PlottingConfig.INFO_BOX_POSITION[1], 
            info_text, transform=ax.transAxes, 
            fontsize=PlottingConfig.INFO_BOX_FONTSIZE,
            verticalalignment=PlottingConfig.INFO_BOX_VERTICAL_ALIGNMENT,
            bbox=PlottingConfig.INFO_BOX_STYLE)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(save_path, 'theta_power_by_nm_type.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"✓ Theta power plot saved to: {plot_file}")
        print(f"Summary statistics:")
        for nm_type, mean_val, se_val in zip(nm_types, nm_means, nm_ses):
            print(f"  NM {int(nm_type)}: {mean_val:.3f} ± {se_val:.3f}")
    
    plt.show()
    
    # Save numerical results
    theta_results = {
        'nm_types': nm_types,
        'means': nm_means,
        'standard_errors': nm_ses,
        'individual_powers': [powers.tolist() for powers in theta_powers],
        'theta_range': theta_range,
        'time_window': time_window,
        'n_rats': results['n_rats'],
        'roi_specification': results['roi_specification']
    }
    
    results_file = os.path.join(save_path, 'theta_power_results.json')
    with open(results_file, 'w') as f:
        json.dump(theta_results, f, indent=2)
    
    if verbose:
        print(f"✓ Theta power results saved to: {results_file}")


def load_cross_rats_results(results_path: str) -> Dict:
    """
    Load cross-rats results from pickle file.
    
    Parameters:
    -----------
    results_path : str
        Path to the cross_rats_aggregated_results.pkl file
        
    Returns:
    --------
    results : Dict
        Cross-rats aggregated results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


def create_theta_power_analysis(results_path: str, 
                               theta_range: Tuple[float, float] = (4.0, 8.0),
                               time_window: Optional[Tuple[float, float]] = None,
                               ylim: Optional[Tuple[float, float]] = (-0.3, 0.3),
                               save_path: Optional[str] = None,
                               verbose: bool = True) -> Dict:
    """
    Main function to create theta power analysis from existing results.
    
    Parameters:
    -----------
    results_path : str
        Path to the cross_rats_aggregated_results.pkl file
    theta_range : Tuple[float, float]
        Frequency range for theta band (default: 4-8 Hz)
    time_window : Optional[Tuple[float, float]]
        Time window to extract power from (default: None for all times)
    ylim : Optional[Tuple[float, float]]
        Y-axis limits for the plot (default: (-0.3, 0.3))
    save_path : Optional[str]
        Directory to save plots (default: same as results directory)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    theta_results : Dict
        Theta power analysis results
    """
    if verbose:
        print("🧠 NM Theta Power Analysis")
        print("=" * 60)
        print(f"Loading results from: {results_path}")
    
    # Load results
    results = load_cross_rats_results(results_path)
    
    if verbose:
        print(f"✓ Loaded results for {results['n_rats']} rats")
        print(f"  ROI: {results['roi_specification']}")
        print(f"  NM types: {list(results['averaged_windows'].keys())}")
    
    # Set save path if not provided
    if save_path is None:
        save_path = os.path.dirname(results_path)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create theta power plots
    create_theta_power_plots(
        results=results,
        save_path=save_path,
        theta_range=theta_range,
        time_window=time_window,
        ylim=ylim,
        verbose=verbose
    )
    
    # Load the saved theta results
    theta_results_file = os.path.join(save_path, 'theta_power_results.json')
    with open(theta_results_file, 'r') as f:
        theta_results = json.load(f)
    
    return theta_results


def main():
    """
    Main function for command line usage.
    """
    parser = argparse.ArgumentParser(
        description='Create theta power plots from cross-rats NM analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default theta range (4-8 Hz)
  python nm_theta_power_plots.py --results_path results/cross_rats/cross_rats_aggregated_results.pkl
  
  # Custom theta range
  python nm_theta_power_plots.py --results_path results/cross_rats/cross_rats_aggregated_results.pkl --theta_min 3 --theta_max 7
  
  # Extract power from specific time window
  python nm_theta_power_plots.py --results_path results/cross_rats/cross_rats_aggregated_results.pkl --time_min 0.2 --time_max 0.8
  
  # Custom save path
  python nm_theta_power_plots.py --results_path results/cross_rats/cross_rats_aggregated_results.pkl --save_path results/theta_plots
        """
    )
    
    # Required parameters
    parser.add_argument('--results_path', required=True,
                       help='Path to cross_rats_aggregated_results.pkl file')
    
    # Theta band parameters
    parser.add_argument('--theta_min', type=float, default=AnalysisConfig.THETA_MIN_FREQ,
                       help=f'Minimum theta frequency (Hz) (default: {AnalysisConfig.THETA_MIN_FREQ})')
    parser.add_argument('--theta_max', type=float, default=AnalysisConfig.THETA_MAX_FREQ,
                       help=f'Maximum theta frequency (Hz) (default: {AnalysisConfig.THETA_MAX_FREQ})')
    
    # Time window parameters
    parser.add_argument('--time_min', type=float, default=None,
                       help='Start time for power extraction (s)')
    parser.add_argument('--time_max', type=float, default=None,
                       help='End time for power extraction (s)')
    
    # Plot parameters
    parser.add_argument('--ylim_min', type=float, default=PlottingConfig.THETA_POWER_YLIM_MIN,
                       help=f'Minimum y-axis limit (default: {PlottingConfig.THETA_POWER_YLIM_MIN})')
    parser.add_argument('--ylim_max', type=float, default=PlottingConfig.THETA_POWER_YLIM_MAX,
                       help=f'Maximum y-axis limit (default: {PlottingConfig.THETA_POWER_YLIM_MAX})')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default=None,
                       help='Directory to save plots (default: same as results directory)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Parse time window
    time_window = None
    if args.time_min is not None and args.time_max is not None:
        time_window = (args.time_min, args.time_max)
    
    # Parse y-axis limits
    ylim = (args.ylim_min, args.ylim_max)
    
    # Run the analysis
    theta_results = create_theta_power_analysis(
        results_path=args.results_path,
        theta_range=(args.theta_min, args.theta_max),
        time_window=time_window,
        ylim=ylim,
        save_path=args.save_path,
        verbose=not args.quiet
    )
    
    print(f"\n✅ Theta power analysis completed!")
    print(f"Results saved to: {args.save_path or os.path.dirname(args.results_path)}")


if __name__ == "__main__":
    # Check if running from command line or directly
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # Direct usage example
        print("🧠 NM Theta Power Analysis - Direct Usage Example")
        print("=" * 60)
        
        # Example usage - modify these paths as needed
        results_path = "results/cross_rats/cross_rats_aggregated_results.pkl"
        
        # Check if example results exist
        if os.path.exists(results_path):
            print(f"Loading results from: {results_path}")
            
            # Run theta power analysis
            theta_results = create_theta_power_analysis(
                results_path=results_path,
                theta_range=(7.0, 10.0),        # Theta frequency range
                time_window=(-0.10,0.10),              # Use all time points, or specify (start, end) in seconds
                save_path=None,                # Save in same directory as results
                verbose=True
            )
            
            print(f"\n✅ Analysis completed!")
            print(f"Theta power results: {theta_results}")
            
        else:
            print(f"❌ Example results file not found: {results_path}")
            print("Please run nm_theta_cross_rats.py first to generate results, then use this script.")
            print("\nCommand line usage:")
            print("python nm_theta_power_plots.py --results_path path/to/cross_rats_aggregated_results.pkl")