#!/usr/bin/env python3
"""
Memory-Efficient Cross-Rats NM Theta Analysis

This script performs theta oscillation analysis around Near Mistake (NM) events
across multiple rats using a memory-efficient approach. It:

1. Loads the main data file once
2. For each rat: processes all sessions, saves multi-session results, discards from memory
3. Aggregates multi-session results across all rats
4. Saves final cross-rats averaged results

Key Features:
- Memory-efficient: processes one rat at a time
- Uses nm_theta_analyzer.py for consistent multi-session analysis  
- Automatic rat discovery from dataset
- Cross-rats statistical aggregation
- Comprehensive result saving and visualization

Usage:
    python nm_theta_cross_rats.py --roi frontal --freq_min 3 --freq_max 8

Author: Generated for EEG near-mistake cross-rats analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pandas as pd
import argparse
import gc
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings
from datetime import datetime
import json

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

# Import our modules
from nm_theta_analyzer import run_analysis


def check_rat_channel_count(rat_id: str, pkl_path: str) -> int:
    """
    Check the number of channels for a specific rat by examining their EEG data.
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to check
    pkl_path : str
        Path to the main EEG data file
        
    Returns:
    --------
    channel_count : int
        Number of channels for this rat
    """
    print(f"🔍 Checking channel count for rat {rat_id}")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Find first session for this rat
    for session_data in all_data:
        if str(session_data.get('rat_id')) == str(rat_id):
            eeg_data = session_data.get('eeg')
            if eeg_data is not None:
                channel_count = eeg_data.shape[0]
                print(f"✓ Rat {rat_id}: {channel_count} channels")
                return channel_count
    
    print(f"⚠️  No EEG data found for rat {rat_id}")
    return 0


def discover_rat_ids(pkl_path: str, exclude_20_channel_rats: bool = True) -> List[str]:
    """
    Discover all unique rat IDs from the dataset.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the main EEG data file
    exclude_20_channel_rats : bool
        Whether to exclude rats with 20 channels (default: True)
        
    Returns:
    --------
    rat_ids : List[str]
        List of unique rat IDs found in the dataset
    """
    print(f"🔍 Discovering rat IDs from {pkl_path}")
    print("Loading data to scan for rat IDs...")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    rat_ids = set()
    for session_data in all_data:
        rat_id = session_data.get('rat_id')
        if rat_id is not None:
            rat_ids.add(str(rat_id))
    
    rat_ids_list = sorted(list(rat_ids))
    print(f"✓ Found {len(rat_ids_list)} unique rats: {rat_ids_list}")
    
    # Check channel counts and filter if requested
    if exclude_20_channel_rats:
        print(f"\n🔍 Checking channel counts (excluding 20-channel rats)...")
        valid_rat_ids = []
        excluded_rat_ids = []
        
        for rat_id in rat_ids_list:
            channel_count = check_rat_channel_count(rat_id, pkl_path)
            if channel_count == 20:
                excluded_rat_ids.append(rat_id)
                print(f"❌ Excluding rat {rat_id} (20 channels)")
            else:
                valid_rat_ids.append(rat_id)
                print(f"✅ Including rat {rat_id} ({channel_count} channels)")
        
        print(f"\n📊 Channel count filtering results:")
        print(f"  Total rats found: {len(rat_ids_list)}")
        print(f"  Rats with 20 channels (excluded): {len(excluded_rat_ids)}")
        print(f"  Valid rats (included): {len(valid_rat_ids)}")
        
        if excluded_rat_ids:
            print(f"  Excluded rat IDs: {excluded_rat_ids}")
        
        rat_ids_list = valid_rat_ids
    
    # Clean up memory
    del all_data
    gc.collect()
    
    return rat_ids_list


def process_single_rat_multi_session(
    rat_id: str,
    roi_or_channels: Union[str, List[int]],
    pkl_path: str,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 30,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    base_save_path: str = 'results/cross_rats',
    show_plots: bool = False
) -> Tuple[str, Optional[Dict]]:
    """
    Process multi-session analysis for a single rat using nm_theta_analyzer.
    
    Parameters:
    -----------
    rat_id : str
        Rat ID to process
    roi_or_channels : Union[str, List[int]]
        ROI specification
    pkl_path : str
        Path to main data file
    freq_range : Tuple[float, float]
        Frequency range for analysis
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration
    n_cycles_factor : float
        Cycles factor for spectrograms
    base_save_path : str
        Base directory for saving results
    show_plots : bool
        Whether to show plots during processing
        
    Returns:
    --------
    rat_id : str
        The processed rat ID
    results : Optional[Dict]
        Multi-session results for the rat, or None if failed
    """
    print(f"\n🐀 Processing rat {rat_id} - Multi-session analysis")
    print("=" * 60)
    
    try:
        # Create save path for this rat
        rat_save_path = os.path.join(base_save_path, f'rat_{rat_id}_multi_session')
        
        # Use nm_theta_analyzer to run multi-session analysis
        results = run_analysis(
            mode='multi',
            method='basic',  # method ignored for multi-session
            parallel_type=None,
            pkl_path=pkl_path,
            roi=roi_or_channels if isinstance(roi_or_channels, str) else ','.join(map(str, roi_or_channels)),
            session_index=None,  # ignored for multi-session
            rat_id=rat_id,
            freq_min=freq_range[0],
            freq_max=freq_range[1],
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            n_jobs=None,
            batch_size=8,
            save_path=rat_save_path,
            show_plots=show_plots,
            show_frequency_profiles=False
        )
        
        print(f"✓ Successfully processed rat {rat_id}")
        print(f"  Sessions analyzed: {results.get('n_sessions_analyzed', 'unknown')}")
        print(f"  Results saved to: {rat_save_path}")
        
        # Force garbage collection
        gc.collect()
        
        return rat_id, results
        
    except Exception as e:
        print(f"❌ Error processing rat {rat_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return rat_id, None


def aggregate_cross_rats_results(
    rat_results: Dict[str, Dict],
    roi_specification: Union[str, List[int]],
    freq_range: Tuple[float, float],
    save_path: str
) -> Dict:
    """
    Aggregate multi-session results across all rats.
    
    Parameters:
    -----------
    rat_results : Dict[str, Dict]
        Dictionary mapping rat_id -> multi-session results
    roi_specification : Union[str, List[int]]
        ROI specification used for analysis
    freq_range : Tuple[float, float]
        Frequency range used
    save_path : str
        Path to save aggregated results
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    print(f"\n📊 Aggregating results across {len(rat_results)} rats")
    print("=" * 60)
    
    # Filter out failed results
    valid_results = {rat_id: results for rat_id, results in rat_results.items() if results is not None}
    n_valid_rats = len(valid_results)
    
    if n_valid_rats == 0:
        raise ValueError("No valid rat results to aggregate")
    
    print(f"Valid results from {n_valid_rats} rats: {list(valid_results.keys())}")
    
    # Get reference data from first valid result
    first_result = next(iter(valid_results.values()))
    frequencies = first_result['frequencies']
    n_freqs = len(frequencies)
    
    # Initialize aggregation containers
    aggregated_windows = defaultdict(lambda: {
        'spectrograms': [],
        'total_events': [],
        'n_sessions': [],
        'rat_ids': []
    })
    
    # Collect spectrograms from all rats
    for rat_id, results in valid_results.items():
        print(f"Processing results from rat {rat_id}")
        
        for nm_size, window_data in results['aggregated_windows'].items():
            spectrograms = aggregated_windows[nm_size]['spectrograms']
            spectrograms.append(window_data['avg_spectrogram'])
            
            aggregated_windows[nm_size]['total_events'].append(window_data['total_events'])
            aggregated_windows[nm_size]['n_sessions'].append(window_data['n_sessions'])
            aggregated_windows[nm_size]['rat_ids'].append(rat_id)
    
    # Compute cross-rats averages
    final_aggregated_windows = {}
    
    for nm_size, data in aggregated_windows.items():
        spectrograms = np.array(data['spectrograms'])  # Shape: (n_rats, n_freqs, n_times)
        
        print(f"NM size {nm_size}: {spectrograms.shape[0]} rats, "
              f"spectrogram shape: {spectrograms.shape[1:]}") 
        
        # Average across rats
        avg_spectrogram = np.mean(spectrograms, axis=0)
        sem_spectrogram = np.std(spectrograms, axis=0) / np.sqrt(spectrograms.shape[0])
        
        final_aggregated_windows[nm_size] = {
            'avg_spectrogram': avg_spectrogram,
            'sem_spectrogram': sem_spectrogram,
            'individual_spectrograms': spectrograms,
            'window_times': first_result['aggregated_windows'][nm_size]['window_times'],
            'total_events_per_rat': data['total_events'],
            'n_sessions_per_rat': data['n_sessions'],
            'rat_ids': data['rat_ids'],
            'n_rats': spectrograms.shape[0],
            'total_events_all_rats': sum(data['total_events']),
            'total_sessions_all_rats': sum(data['n_sessions'])
        }
    
    # Create comprehensive results dictionary
    aggregated_results = {
        'analysis_type': 'cross_rats',
        'rat_ids': list(valid_results.keys()),
        'n_rats': n_valid_rats,
        'roi_specification': roi_specification,
        'roi_channels': first_result['roi_channels'],
        'frequencies': frequencies,
        'aggregated_windows': final_aggregated_windows,
        'analysis_parameters': {
            'frequency_range': freq_range,
            'n_frequencies': n_freqs,
            'window_duration': first_result['analysis_parameters']['window_duration'],
            'normalization': first_result['analysis_parameters']['normalization']
        },
        'processing_info': {
            'n_rats_attempted': len(rat_results),
            'n_rats_successful': n_valid_rats,
            'failed_rats': [rat_id for rat_id, results in rat_results.items() if results is None],
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Save aggregated results
    os.makedirs(save_path, exist_ok=True)
    
    results_file = os.path.join(save_path, 'cross_rats_aggregated_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated_results, f)
    
    print(f"✓ Cross-rats results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = os.path.join(save_path, 'cross_rats_summary.json')
    summary_data = {
        'n_rats': n_valid_rats,
        'rat_ids': list(valid_results.keys()),
        'nm_sizes_analyzed': list(final_aggregated_windows.keys()),
        'total_events_all_rats': {nm_size: data['total_events_all_rats'] 
                                  for nm_size, data in final_aggregated_windows.items()},
        'total_sessions_all_rats': {nm_size: data['total_sessions_all_rats']
                                    for nm_size, data in final_aggregated_windows.items()},
        'frequency_range': freq_range,
        'roi_specification': str(roi_specification),
        'roi_channels': first_result['roi_channels'].tolist() if hasattr(first_result['roi_channels'], 'tolist') else first_result['roi_channels']
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✓ Summary statistics saved to: {summary_file}")
    
    return aggregated_results


def create_cross_rats_visualizations(results: Dict, save_path: str):
    """
    Create comprehensive visualizations for cross-rats results.
    
    Parameters:
    -----------
    results : Dict
        Cross-rats aggregated results
    save_path : str
        Directory to save visualizations
    """
    print(f"\n📈 Creating cross-rats visualizations")
    print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    
    frequencies = results['frequencies']
    n_rats = results['n_rats']
    roi_channels = results['roi_channels']
    
    # Create figure with subplots for each NM size
    nm_sizes = list(results['aggregated_windows'].keys())
    n_nm_sizes = len(nm_sizes)
    
    if n_nm_sizes == 0:
        print("⚠️  No NM sizes to plot")
        return
    
    fig, axes = plt.subplots(n_nm_sizes, 2, figsize=(15, 5 * n_nm_sizes))
    if n_nm_sizes == 1:
        axes = axes.reshape(1, -1)
    
    for i, nm_size in enumerate(nm_sizes):
        window_data = results['aggregated_windows'][nm_size]
        avg_spectrogram = window_data['avg_spectrogram']
        sem_spectrogram = window_data['sem_spectrogram']
        window_times = window_data['window_times']
        
        # Plot 1: Average spectrogram across rats
        ax1 = axes[i, 0]
        im1 = ax1.imshow(avg_spectrogram, aspect='auto', origin='lower',
                        extent=[window_times[0], window_times[-1], 
                               frequencies[0], frequencies[-1]],
                        cmap='RdBu_r', vmin=-2, vmax=2)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title(f'NM Size {nm_size} - Average Across {n_rats} Rats\n'
                     f'Total Events: {window_data["total_events_all_rats"]}, '
                     f'Sessions: {window_data["total_sessions_all_rats"]}')
        
        plt.colorbar(im1, ax=ax1, label='Z-score')
        
        # Plot 2: Standard error across rats
        ax2 = axes[i, 1]
        im2 = ax2.imshow(sem_spectrogram, aspect='auto', origin='lower',
                        extent=[window_times[0], window_times[-1],
                               frequencies[0], frequencies[-1]],
                        cmap='viridis')
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title(f'NM Size {nm_size} - Standard Error Across Rats')
        
        plt.colorbar(im2, ax=ax2, label='SEM Z-score')
    
    # Add overall title
    roi_str = f"ROI: {results['roi_specification']} (channels: {roi_channels})"
    freq_str = f"Freq: {results['analysis_parameters']['frequency_range'][0]}-{results['analysis_parameters']['frequency_range'][1]} Hz"
    
    plt.suptitle(f'Cross-Rats NM Theta Analysis\n{roi_str}, {freq_str}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_path, 'cross_rats_spectrograms.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Spectrograms saved to: {plot_file}")
    
    plt.show()
    
    # Create individual rat comparison plot
    if n_nm_sizes == 1:  # Only create if single NM size to avoid complexity
        nm_size = nm_sizes[0]
        window_data = results['aggregated_windows'][nm_size]
        individual_spectrograms = window_data['individual_spectrograms']
        rat_ids = window_data['rat_ids']
        
        n_rats_to_show = min(6, len(rat_ids))  # Show up to 6 rats
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for j in range(n_rats_to_show):
            ax = axes[j]
            spectrogram = individual_spectrograms[j]
            rat_id = rat_ids[j]
            
            im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                          extent=[window_times[0], window_times[-1],
                                 frequencies[0], frequencies[-1]],
                          cmap='RdBu_r', vmin=-2, vmax=2)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Rat {rat_id}\nEvents: {window_data["total_events_per_rat"][j]}, '
                        f'Sessions: {window_data["n_sessions_per_rat"][j]}')
            
            plt.colorbar(im, ax=ax, label='Z-score')
        
        # Hide unused subplots
        for j in range(n_rats_to_show, 6):
            axes[j].set_visible(False)
        
        plt.suptitle(f'Individual Rat Spectrograms - NM Size {nm_size}\n{roi_str}, {freq_str}', fontsize=14)
        plt.tight_layout()
        
        individual_plot_file = os.path.join(save_path, f'individual_rats_nm_{nm_size}.png')
        plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Individual rats plot saved to: {individual_plot_file}")
        
        plt.show()


def run_cross_rats_analysis(
    roi: str,
    pkl_path: str = 'data/processed/all_eeg_data.pkl',
    freq_min: float = 3.0,
    freq_max: float = 8.0,
    n_freqs: int = 30,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    rat_ids: Optional[List[str]] = None,
    save_path: str = 'results/cross_rats',
    show_plots: bool = False
) -> Dict:
    """
    Run cross-rats NM theta analysis with direct parameter specification.
    
    Parameters:
    -----------
    roi : str
        ROI name ("frontal", "hippocampus") or channels ("1,2,3")
    pkl_path : str
        Path to main EEG data file
    freq_min : float
        Minimum frequency (Hz)
    freq_max : float
        Maximum frequency (Hz)
    n_freqs : int
        Number of frequencies
    window_duration : float
        Event window duration (s)
    n_cycles_factor : float
        Cycles factor for spectrograms
    rat_ids : Optional[List[str]]
        Specific rat IDs to process, None for all rats
    save_path : str
        Base directory for saving results
    show_plots : bool
        Show plots during processing
        
    Returns:
    --------
    aggregated_results : Dict
        Cross-rats aggregated results
    """
    print("🧠 Cross-Rats NM Theta Analysis")
    print("=" * 80)
    print(f"Data file: {pkl_path}")
    print(f"ROI: {roi}")
    print(f"Frequency range: {freq_min}-{freq_max} Hz ({n_freqs} freqs)")
    print(f"Window duration: {window_duration}s")
    print(f"Save path: {save_path}")
    print("=" * 80)
    
    # Discover rat IDs
    if rat_ids:
        print(f"Using specified rat IDs: {rat_ids}")
    else:
        rat_ids = discover_rat_ids(pkl_path)
    
    # Process each rat individually
    rat_results = {}
    
    for rat_id in rat_ids:
        rat_id_str, results = process_single_rat_multi_session(
            rat_id=rat_id,
            roi_or_channels=roi,
            pkl_path=pkl_path,
            freq_range=(freq_min, freq_max),
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            base_save_path=save_path,
            show_plots=show_plots
        )
        
        rat_results[rat_id_str] = results
        
        # Force garbage collection after each rat
        gc.collect()
    
    # Aggregate results across rats
    aggregated_results = aggregate_cross_rats_results(
        rat_results=rat_results,
        roi_specification=roi,
        freq_range=(freq_min, freq_max),
        save_path=save_path
    )
    
    # Create visualizations
    create_cross_rats_visualizations(
        results=aggregated_results,
        save_path=save_path
    )
    
    print("\n✅ Cross-rats analysis completed successfully!")
    print(f"Results saved to: {save_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Rats processed: {aggregated_results['n_rats']}")
    print(f"  Rat IDs: {aggregated_results['rat_ids']}")
    print(f"  NM sizes analyzed: {list(aggregated_results['aggregated_windows'].keys())}")
    print(f"  ROI channels: {aggregated_results['roi_channels']}")
    
    return aggregated_results


def main():
    """
    Main function for cross-rats NM theta analysis.
    Supports both command line arguments and direct parameter modification.
    """
    parser = argparse.ArgumentParser(
        description='Cross-Rats NM Theta Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze frontal ROI across all rats
  python nm_theta_cross_rats.py --roi frontal --freq_min 3 --freq_max 8
  
  # Analyze specific channels across all rats
  python nm_theta_cross_rats.py --roi "1,2,3" --freq_min 1 --freq_max 12
  
  # Analyze hippocampus with custom parameters
  python nm_theta_cross_rats.py --roi hippocampus --freq_min 6 --freq_max 10 --n_freqs 20
        """
    )
    
    # Data parameters
    parser.add_argument('--pkl_path', default='data/processed/all_eeg_data.pkl',
                       help='Path to main EEG data file')
    parser.add_argument('--roi', required=True,
                       help='ROI name ("frontal", "hippocampus") or channels ("1,2,3")')
    
    # Analysis parameters
    parser.add_argument('--freq_min', type=float, default=3.0,
                       help='Minimum frequency (Hz)')
    parser.add_argument('--freq_max', type=float, default=8.0,
                       help='Maximum frequency (Hz)')
    parser.add_argument('--n_freqs', type=int, default=30,
                       help='Number of frequencies')
    parser.add_argument('--window_duration', type=float, default=1.0,
                       help='Event window duration (s)')
    parser.add_argument('--n_cycles_factor', type=float, default=3.0,
                       help='Cycles factor for spectrograms')
    
    # Processing parameters
    parser.add_argument('--rat_ids', type=str, default=None,
                       help='Specific rat IDs to process (comma-separated), default: all rats')
    parser.add_argument('--save_path', default='results/cross_rats',
                       help='Base directory for saving results')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show plots during processing')
    
    args = parser.parse_args()
    
    # Parse rat_ids if provided
    rat_ids = None
    if args.rat_ids:
        rat_ids = [r.strip() for r in args.rat_ids.split(',')]
    
    # Run the analysis
    return run_cross_rats_analysis(
        roi=args.roi,
        pkl_path=args.pkl_path,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        n_freqs=args.n_freqs,
        window_duration=args.window_duration,
        n_cycles_factor=args.n_cycles_factor,
        rat_ids=rat_ids,
        save_path=args.save_path,
        show_plots=args.show_plots
    )


# Example usage for IDE/notebook environment
if __name__ == "__main__":
    # Check if we're running in IDE mode (with direct function calls)
    # or command line mode
    import sys
    
    # If there are command line arguments, use command line mode
    if len(sys.argv) > 1:
        # For command line usage, call main()
        main()
    else:
        # For IDE usage, run the analysis directly
        results = run_cross_rats_analysis(
            roi="2",                    # ROI specification
            pkl_path="data/processed/all_eeg_data.pkl",  # Data file path
            freq_min=1.0,                     # Minimum frequency
            freq_max=45.0,                     # Maximum frequency
            n_freqs=30,                       # Number of frequencies
            window_duration=2.0,              # Event window duration
            n_cycles_factor=3.0,              # Cycles factor
            rat_ids=None,                     # None for all rats, or ["rat1", "rat2"] for specific rats
            save_path="results/cross_rats",   # Save directory
            show_plots=False                  # Show plots during processing
        )