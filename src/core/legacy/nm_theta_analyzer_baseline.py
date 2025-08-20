#!/usr/bin/env python3
"""
NM Theta Analysis Orchestrator - Baseline Normalization Version

This script provides a unified interface for all NM theta analysis methods using
baseline normalization instead of global statistics.

Key Difference from Original:
- Uses pre-event baseline normalization (-1.0 to -0.5 seconds) instead of global statistics
- Extracts statistics from -1.0 to -0.5 seconds window before each event
- Normalizes the full -1 to +1 second window using these baseline statistics

It automatically selects the appropriate implementation based on your requirements:

- Single session + basic analysis → nm_theta_single_basic_baseline.py
- Multi-session analysis → nm_theta_multi_session_baseline.py

Usage Methods:
    1. IDE Usage: Modify parameters in code and run directly
    2. Command Line Usage: Use command-line arguments

Unified interface for EEG near-mistake analysis with baseline normalization
"""

import argparse
import sys
import os
from typing import Union, List, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import baseline normalization functions
from normalization.baseline_normalization import (
    compute_baseline_statistics, 
    normalize_windows_baseline,
    extract_nm_event_windows_with_baseline,
    analyze_session_with_baseline_normalization
)


def print_method_info(mode: str, method: str, parallel_type: str = None):
    """Print information about the selected analysis method."""
    print("🔧 ANALYSIS METHOD SELECTION - BASELINE NORMALIZATION")
    print("=" * 50)
    print("⚠️  NORMALIZATION: Using pre-event baseline (-1.0 to -0.5 seconds)")
    print("    instead of global statistics")
    print()
    
    if mode == "single":
        if method == "basic":
            print("📋 Selected: Single Session Basic Analysis (Baseline)")
            print("   • Implementation: nm_theta_single_basic_baseline.py")
            print("   • Processing: Sequential, single-threaded")
            print("   • Memory usage: Standard")
            print("   • Speed: Baseline")
            print("   • Normalization: Pre-event baseline (-1.0 to -0.5s)")
            print("   • Best for: Small datasets, testing, simple analysis")
            
        elif method == "parallel":
            print("📋 Selected: Single Session Parallel Analysis (Baseline)")
            print("   • Implementation: nm_theta_single_parallel_baseline.py")
            print(f"   • Processing: Multi-threaded ({parallel_type})")
            print("   • Memory usage: Standard")
            print("   • Speed: Fast (parallelized channels)")
            print("   • Normalization: Pre-event baseline (-1.0 to -0.5s)")
            print("   • Best for: Multi-core systems, faster processing")
            
        elif method == "vectorized":
            print("📋 Selected: Single Session Vectorized Analysis (Baseline)")
            print("   • Implementation: nm_theta_single_vectorized_baseline.py")
            print("   • Processing: Vectorized CWT (scipy)")
            print("   • Memory usage: Optimized with chunking")
            print("   • Speed: Fastest (batch frequency processing)")
            print("   • Normalization: Pre-event baseline (-1.0 to -0.5s)")
            print("   • Best for: Maximum speed, large frequency ranges")
            
    elif mode == "multi":
        print("📋 Selected: Multi-Session Analysis (Baseline)")
        print("   • Implementation: nm_theta_multi_session_baseline.py")
        print("   • Processing: Memory-efficient sequential")
        print("   • Memory usage: Minimal (load-process-save)")
        print("   • Speed: Moderate (optimized for memory)")
        print("   • Normalization: Pre-event baseline (-1.0 to -0.5s)")
        print("   • Best for: Multiple sessions, memory-constrained systems")
    
    print("=" * 50)


def validate_parameters(mode, method, parallel_type, session_index, rat_id, pkl_path, freq_min, freq_max):
    """Validate parameter combinations."""
    errors = []
    
    # Mode-specific validations
    if mode == "single":
        if session_index is None:
            errors.append("Single session mode requires session_index")
        if rat_id is not None:
            print("⚠️  Warning: rat_id ignored in single session mode")
            
    elif mode == "multi":
        if rat_id is None:
            errors.append("Multi-session mode requires rat_id")
        if session_index is not None:
            print("⚠️  Warning: session_index ignored in multi-session mode")
        if method != "basic":
            print("⚠️  Warning: Multi-session mode uses memory-efficient processing (method ignored)")
    
    # Method-specific validations
    if method == "parallel" and parallel_type is None:
        errors.append("Parallel method requires parallel_type (threading or multiprocessing)")
    
    if method != "parallel" and parallel_type is not None:
        print("⚠️  Warning: parallel_type ignored (only used with parallel method)")
    
    # File existence
    if not os.path.exists(pkl_path):
        errors.append(f"Data file not found: {pkl_path}")
    
    # Frequency range
    if freq_min >= freq_max:
        errors.append("freq_min must be less than freq_max")
    
    if len(errors) > 0:
        print("❌ Parameter validation errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    return True


def parse_roi_specification(roi_str: str) -> Union[str, List[int]]:
    """Parse ROI specification from string."""
    try:
        # Try to parse as comma-separated channel numbers
        channels = [int(x.strip()) for x in roi_str.split(',')]
        return channels
    except ValueError:
        # Use as ROI name
        return roi_str


def run_analysis(mode, method, parallel_type, pkl_path, roi, session_index, rat_id,
                freq_min, freq_max, n_freqs, window_duration, n_cycles_factor,
                n_jobs, batch_size, save_path, show_plots, show_frequency_profiles):
    """Run the analysis with the specified parameters using baseline normalization."""
    
    roi_specification = parse_roi_specification(roi)
    
    if mode == "single":
        if method == "basic":
            from nm_theta_single_basic import analyze_session_nm_theta_roi, load_session_data
            
            print("Loading session data...")
            session_data = load_session_data(pkl_path, session_index)
            
            save_dir = save_path or f'../../results/single_session_baseline/session_{session_index}_basic'
            
            print("Running basic analysis with baseline normalization...")
            results = analyze_session_nm_theta_roi(
                session_data=session_data,
                roi_or_channels=roi_specification,
                freq_range=(freq_min, freq_max),
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                save_path=save_dir,
                show_plots=show_plots,
                show_frequency_profiles=show_frequency_profiles
            )
            
        elif method == "parallel":
            from nm_theta_single_parallel import analyze_session_nm_theta_roi_parallel, load_session_data
            
            print("Loading session data...")
            session_data = load_session_data(pkl_path, session_index)
            
            save_dir = save_path or f'../../results/single_session_baseline/session_{session_index}_parallel_{parallel_type}'
            
            print("Running parallel analysis with baseline normalization...")
            results = analyze_session_nm_theta_roi_parallel(
                session_data=session_data,
                roi_or_channels=roi_specification,
                freq_range=(freq_min, freq_max),
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                save_path=save_dir,
                show_plots=show_plots,
                show_frequency_profiles=show_frequency_profiles,
                n_jobs=n_jobs,
                parallel_method=parallel_type
            )
            
        elif method == "vectorized":
            from nm_theta_single_vectorized import analyze_session_nm_theta_roi_vectorized, load_session_data
            
            print("Loading session data...")
            session_data = load_session_data(pkl_path, session_index)
            
            save_dir = save_path or f'../../results/single_session_baseline/session_{session_index}_vectorized_cwt'
            
            print("Running vectorized analysis with baseline normalization...")
            results = analyze_session_nm_theta_roi_vectorized(
                session_data=session_data,
                roi_or_channels=roi_specification,
                freq_range=(freq_min, freq_max),
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                save_path=save_dir,
                show_plots=show_plots,
                show_frequency_profiles=show_frequency_profiles,
                n_jobs=n_jobs,
                batch_size=batch_size,
                method='cwt'  # Always use CWT for vectorized mode
            )
            
    elif mode == "multi":
        from nm_theta_multi_session import analyze_rat_multi_session_memory_efficient
        
        save_dir = save_path or f'../../results/multi_session_baseline/rat_{rat_id}_memory_efficient'
        
        print("Running multi-session analysis with baseline normalization...")
        results = analyze_rat_multi_session_memory_efficient(
            rat_id=rat_id,
            roi_or_channels=roi_specification,
            pkl_path=pkl_path,
            freq_range=(freq_min, freq_max),
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            save_path=save_dir,
            show_plots=show_plots
        )
    
    return results


def main():
    """
    Main orchestrator function with dual-mode support for baseline normalization.
    
    🎯 FOR IDE USAGE: Modify the parameters below and run directly
    🖥️ FOR COMMAND LINE: Use arguments like --mode single --roi frontal etc.
    """
    
    # =============================================================================
    # 🎯 IDE CONFIGURATION - MODIFY THESE PARAMETERS FOR IDE USAGE
    # =============================================================================
    
    # Analysis mode and method
    mode = 'single'              # 'single' or 'multi'
    method = 'basic'             # 'basic', 'parallel', or 'vectorized'
    parallel_type = 'threading'  # 'threading' or 'multiprocessing' (for parallel method)
    
    # Data parameters
    pkl_path = 'data/processed/all_eeg_data.pkl'
    roi = 'frontal'             # ROI name ('frontal', 'hippocampus') or custom channels like '2,3,5'
    
    # Session parameters
    session_index = 0            # For single session mode (0-based)
    rat_id = '10501'            # For multi-session mode
    
    # Analysis parameters
    freq_min = 3.0              # Minimum frequency (Hz)
    freq_max = 8.0              # Maximum frequency (Hz)  
    n_freqs = 20                # Number of logarithmically spaced frequencies
    window_duration = 1.0       # Event window duration (±half around event)
    n_cycles_factor = 3.0       # Factor for adaptive n_cycles
    
    # Processing parameters
    n_jobs = None               # Number of parallel jobs (None = auto-detect)
    batch_size = 8              # Batch size for vectorized processing
    
    # Output parameters
    save_path = None            # Directory to save results (None = auto-generate)
    show_plots = True           # Whether to display plots
    show_frequency_profiles = False  # Show frequency profiles (single session only)
    
    # =============================================================================
    # 🖥️ COMMAND LINE DETECTION AND PARSING
    # =============================================================================
    
    # Check if command line arguments were provided
    use_command_line = len(sys.argv) > 1
    
    if use_command_line:
        print("📋 Using command line arguments...")
        parser = argparse.ArgumentParser(
            description='Unified NM Theta Analysis Interface - Baseline Normalization',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Single session, basic analysis with baseline normalization
  python nm_theta_analyzer_baseline.py --mode single --method basic --session_index 0 --roi frontal
  
  # Single session, parallel with threading and baseline normalization
  python nm_theta_analyzer_baseline.py --mode single --method parallel --parallel_type threading --session_index 0 --roi frontal --n_jobs 4
  
  # Multi-session analysis with baseline normalization
  python nm_theta_analyzer_baseline.py --mode multi --rat_id 10501 --roi frontal --freq_min 3 --freq_max 8
  
  # Custom channels with baseline normalization
  python nm_theta_analyzer_baseline.py --mode single --method basic --session_index 0 --roi "2,3,5,7"
            """
        )
        
        # Core parameters
        parser.add_argument('--mode', choices=['single', 'multi'], required=True,
                           help='Analysis mode: single session or multi-session')
        parser.add_argument('--method', choices=['basic', 'parallel', 'vectorized'], default='basic',
                           help='Processing method (ignored for multi-session mode)')
        
        # Data parameters
        parser.add_argument('--pkl_path', type=str, default='data/processed/all_eeg_data.pkl',
                           help='Path to EEG data pickle file')
        parser.add_argument('--roi', type=str, required=True,
                           help='ROI specification: ROI name (frontal, hippocampus) or comma-separated channel numbers (1,2,3)')
        
        # Session parameters
        parser.add_argument('--session_index', type=int, default=None,
                           help='Session index for single session mode (0-based)')
        parser.add_argument('--rat_id', type=str, default=None,
                           help='Rat ID for multi-session mode')
        
        # Analysis parameters
        parser.add_argument('--freq_min', type=float, default=3.0,
                           help='Minimum frequency (Hz)')
        parser.add_argument('--freq_max', type=float, default=8.0,
                           help='Maximum frequency (Hz)')
        parser.add_argument('--n_freqs', type=int, default=20,
                           help='Number of logarithmically spaced frequencies')
        parser.add_argument('--window_duration', type=float, default=1.0,
                           help='Event window duration in seconds (±half around event)')
        parser.add_argument('--n_cycles_factor', type=float, default=3.0,
                           help='Factor for adaptive n_cycles in spectrograms')
        
        # Processing parameters
        parser.add_argument('--parallel_type', choices=['threading', 'multiprocessing'], default=None,
                           help='Parallelization method (required for parallel mode)')
        parser.add_argument('--n_jobs', type=int, default=None,
                           help='Number of parallel jobs (None = auto-detect)')
        parser.add_argument('--batch_size', type=int, default=8,
                           help='Batch size for vectorized processing')
        
        # Output parameters
        parser.add_argument('--save_path', type=str, default=None,
                           help='Directory to save results (None = auto-generate)')
        parser.add_argument('--no_plots', action='store_true',
                           help='Skip generating plots')
        parser.add_argument('--show_frequency_profiles', action='store_true',
                           help='Show frequency profile plots (single session only)')
        
        # Parse and override variables
        args = parser.parse_args()
        
        mode = args.mode
        method = args.method
        parallel_type = args.parallel_type
        pkl_path = args.pkl_path
        roi = args.roi
        session_index = args.session_index
        rat_id = args.rat_id
        freq_min = args.freq_min
        freq_max = args.freq_max
        n_freqs = args.n_freqs
        window_duration = args.window_duration
        n_cycles_factor = args.n_cycles_factor
        n_jobs = args.n_jobs
        batch_size = args.batch_size
        save_path = args.save_path
        show_plots = not args.no_plots
        show_frequency_profiles = args.show_frequency_profiles
        
    else:
        print("📋 Using IDE parameter configuration...")
        print(f"   Mode: {mode}")
        print(f"   Method: {method}")
        print(f"   ROI: {roi}")
        print(f"   Normalization: Baseline (-1.0 to -0.5 seconds)")
        if mode == 'single':
            print(f"   Session index: {session_index}")
        else:
            print(f"   Rat ID: {rat_id}")
    
    # =============================================================================
    # 🚀 RUN ANALYSIS
    # =============================================================================
    
    # Print header
    print("\n🧠 NM THETA ANALYSIS ORCHESTRATOR - BASELINE NORMALIZATION")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Method: {method}")
    print(f"Normalization: Pre-event baseline (-1.0 to -0.5 seconds)")
    if parallel_type and method == 'parallel':
        print(f"Parallel type: {parallel_type}")
    print(f"ROI: {roi}")
    print(f"Frequency range: {freq_min}-{freq_max} Hz")
    print()
    
    # Validate parameters
    if not validate_parameters(mode, method, parallel_type, session_index, rat_id, pkl_path, freq_min, freq_max):
        return False
    
    # Print method information
    print_method_info(mode, method, parallel_type)
    print()
    
    try:
        # Run analysis
        results = run_analysis(
            mode, method, parallel_type, pkl_path, roi, session_index, rat_id,
            freq_min, freq_max, n_freqs, window_duration, n_cycles_factor,
            n_jobs, batch_size, save_path, show_plots, show_frequency_profiles
        )
        
        # Print success summary
        print("\n🎉 BASELINE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✓ Normalization method: Pre-event baseline (-1.0 to -0.5 seconds)")
        
        if mode == "single":
            print(f"✓ Session analyzed: {session_index}")
            print(f"✓ ROI channels: {results.get('roi_channels', 'N/A')}")
            if 'normalized_windows' in results:
                print(f"✓ NM sizes found: {list(results['normalized_windows'].keys())}")
                total_events = sum(data['n_events'] for data in results['normalized_windows'].values())
                print(f"✓ Total events: {total_events}")
        
        elif mode == "multi":
            print(f"✓ Rat ID: {rat_id}")
            print(f"✓ Sessions analyzed: {results.get('n_sessions_analyzed', 'N/A')}")
            if 'aggregated_windows' in results:
                print(f"✓ NM sizes found: {list(results['aggregated_windows'].keys())}")
                for nm_size, data in results['aggregated_windows'].items():
                    print(f"  - NM size {nm_size}: {data['total_events']} events from {data['n_sessions']} sessions")
        
        print(f"✓ Results saved to: {save_path if save_path else 'auto-generated path'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ BASELINE ANALYSIS FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)