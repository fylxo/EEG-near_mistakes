"""
Batch processing script for spectral analysis pipeline

This script processes multiple sessions from all_eeg_data.pkl using the spectral analysis pipeline.
It includes functions for batch processing, progress tracking, and result aggregation.
"""

import numpy as np
import pickle
import os
import time
from typing import List, Optional, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from spectral_analysis_pipeline import (
    load_session_data,
    process_single_session,
    save_analysis_results
)

def get_dataset_info(pkl_path: str = 'data/processed/all_eeg_data.pkl') -> Dict:
    """
    Get basic information about the dataset without loading all data.
    
    Parameters:
    -----------
    pkl_path : str
        Path to the dataset file
    
    Returns:
    --------
    info : Dict
        Dataset information
    """
    print("Getting dataset information...")
    
    try:
        with open(pkl_path, 'rb') as f:
            # Try to load just the first session to understand structure
            all_data = pickle.load(f)
            n_sessions = len(all_data)
            
            # Sample a few sessions to get statistics
            sample_indices = np.linspace(0, n_sessions-1, min(5, n_sessions), dtype=int)
            
            info = {
                'n_sessions': n_sessions,
                'session_examples': []
            }
            
            for i in sample_indices:
                session = all_data[i]
                session_info = {
                    'index': i,
                    'rat_id': session.get('rat_id', 'unknown'),
                    'session_date': session.get('session_date', 'unknown'),
                    'eeg_shape': session['eeg'].shape,
                    'recording_duration': session['eeg_time'].flatten()[-1] - session['eeg_time'].flatten()[0],
                    'n_nm_events': len(session['nm_peak_times']),
                    'n_iti_events': len(session['iti_peak_times']),
                    'nm_sizes': np.unique(session['nm_sizes']).tolist(),
                    'iti_sizes': np.unique(session['iti_sizes']).tolist()
                }
                info['session_examples'].append(session_info)
            
            return info
            
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return None


def process_session_batch(pkl_path: str,
                         session_indices: List[int],
                         output_dir: str = 'batch_results',
                         sfreq: float = 200.0,
                         freqs: Optional[np.ndarray] = None,
                         n_cycles: int = 7,
                         channels: Optional[List[int]] = None,
                         roi_channels: Optional[List[int]] = None,
                         window_duration: float = 4.0,
                         theta_band: tuple = (4, 8),
                         plot_results: bool = True,
                         save_individual: bool = True) -> Dict:
    """
    Process a batch of sessions.
    
    Parameters:
    -----------
    pkl_path : str
        Path to all_eeg_data.pkl
    session_indices : List[int]
        List of session indices to process
    output_dir : str
        Directory to save results
    ... (other parameters same as process_single_session)
    
    Returns:
    --------
    batch_results : Dict
        Results from all processed sessions
    """
    print(f"Processing batch of {len(session_indices)} sessions...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    batch_results = {
        'processed_sessions': {},
        'failed_sessions': {},
        'batch_summary': {
            'total_sessions': len(session_indices),
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }
    }
    
    # Load the dataset once
    print("Loading dataset...")
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    for i, session_idx in enumerate(session_indices):
        print(f"\n--- Processing session {i+1}/{len(session_indices)} (index {session_idx}) ---")
        
        try:
            # Get session data
            session_data = all_data[session_idx]
            session_id = f"{session_data.get('rat_id', f'session_{session_idx}')}_{session_data.get('session_date', 'unknown')}"
            
            # Process session
            start_time = time.time()
            results = process_single_session(
                session_data=session_data,
                session_id=session_id,
                sfreq=sfreq,
                freqs=freqs,
                n_cycles=n_cycles,
                channels=channels,
                roi_channels=roi_channels,
                window_duration=window_duration,
                theta_band=theta_band,
                save_path=output_dir if save_individual else None,
                plot_results=plot_results
            )
            
            processing_time = time.time() - start_time
            
            # Store results
            batch_results['processed_sessions'][session_idx] = {
                'results': results,
                'processing_time': processing_time,
                'session_id': session_id
            }
            
            batch_results['batch_summary']['successful'] += 1
            print(f"✓ Session {session_idx} completed in {processing_time:.1f}s")
            
        except Exception as e:
            print(f"✗ Error processing session {session_idx}: {e}")
            batch_results['failed_sessions'][session_idx] = str(e)
            batch_results['batch_summary']['failed'] += 1
    
    batch_results['batch_summary']['end_time'] = time.time()
    batch_results['batch_summary']['total_time'] = (
        batch_results['batch_summary']['end_time'] - batch_results['batch_summary']['start_time']
    )
    
    # Save batch results
    with open(f'{output_dir}/batch_results.pkl', 'wb') as f:
        pickle.dump(batch_results, f)
    
    print(f"\nBatch processing complete!")
    print(f"Successful: {batch_results['batch_summary']['successful']}")
    print(f"Failed: {batch_results['batch_summary']['failed']}")
    print(f"Total time: {batch_results['batch_summary']['total_time']:.1f}s")
    
    return batch_results


def process_all_sessions(pkl_path: str = 'data/processed/all_eeg_data.pkl',
                        output_dir: str = 'all_sessions_results',
                        max_sessions: Optional[int] = None,
                        **kwargs) -> Dict:
    """
    Process all sessions in the dataset.
    
    Parameters:
    -----------
    pkl_path : str
        Path to all_eeg_data.pkl
    output_dir : str
        Directory to save results
    max_sessions : Optional[int]
        Maximum number of sessions to process (for testing)
    **kwargs : Additional arguments for processing
    
    Returns:
    --------
    results : Dict
        Batch processing results
    """
    # Get dataset info
    info = get_dataset_info(pkl_path)
    if info is None:
        print("Could not get dataset information")
        return None
    
    n_sessions = info['n_sessions']
    if max_sessions is not None:
        n_sessions = min(n_sessions, max_sessions)
    
    print(f"Dataset contains {info['n_sessions']} sessions total")
    print(f"Processing {n_sessions} sessions")
    
    # Process all sessions
    session_indices = list(range(n_sessions))
    return process_session_batch(pkl_path, session_indices, output_dir, **kwargs)


def aggregate_batch_results(batch_results: Dict, 
                           generate_variability_plots: bool = True,
                           save_path: Optional[str] = None) -> Dict:
    """
    Aggregate results across all processed sessions with optional variability plots.
    
    Parameters:
    -----------
    batch_results : Dict
        Results from process_session_batch
    generate_variability_plots : bool
        Whether to generate multi-session variability plots
    save_path : Optional[str]
        Path to save aggregated plots
    
    Returns:
    --------
    aggregated : Dict
        Aggregated statistics and results
    """
    print("Aggregating batch results...")
    
    aggregated = {
        'summary': batch_results['batch_summary'].copy(),
        'group_statistics': {},
        'session_statistics': {},
        'multi_session_theta_data': []
    }
    
    # Initialize group statistics
    for event_type in ['NM', 'ITI']:
        for size in [1, 2, 3]:
            key = f'{event_type}_size_{size}'
            aggregated['group_statistics'][key] = {
                'n_sessions_with_data': 0,
                'total_events': 0,
                'events_per_session': []
            }
    
    # Aggregate across sessions
    for session_idx, session_data in batch_results['processed_sessions'].items():
        results = session_data['results']
        
        # Session-level statistics
        aggregated['session_statistics'][session_idx] = {
            'session_id': session_data['session_id'],
            'processing_time': session_data['processing_time'],
            'n_channels': len(results['channels']),
            'recording_duration': results['times'][-1] - results['times'][0]
        }
        
        # Collect theta data for variability analysis
        if 'theta_data' in results:
            aggregated['multi_session_theta_data'].append(results['theta_data'])
        
        # Group-level statistics
        if 'group_averages' in results:
            for event_type in results['group_averages']:
                for size in results['group_averages'][event_type]:
                    key = f'{event_type}_size_{size}'
                    n_events = results['group_averages'][event_type][size]['n_events']
                    
                    aggregated['group_statistics'][key]['n_sessions_with_data'] += 1
                    aggregated['group_statistics'][key]['total_events'] += n_events
                    aggregated['group_statistics'][key]['events_per_session'].append(n_events)
    
    # Compute summary statistics
    for key in aggregated['group_statistics']:
        events_per_session = aggregated['group_statistics'][key]['events_per_session']
        if events_per_session:
            aggregated['group_statistics'][key]['mean_events_per_session'] = np.mean(events_per_session)
            aggregated['group_statistics'][key]['std_events_per_session'] = np.std(events_per_session)
            aggregated['group_statistics'][key]['min_events_per_session'] = np.min(events_per_session)
            aggregated['group_statistics'][key]['max_events_per_session'] = np.max(events_per_session)
    
    # Generate variability plots if requested and we have multiple sessions
    if generate_variability_plots and len(aggregated['multi_session_theta_data']) > 1:
        print(f"Generating variability plots for {len(aggregated['multi_session_theta_data'])} sessions...")
        
        # Import plotting functions
        from spectral_analysis_pipeline import plot_theta_timecourses_with_variability
        
        # Get representative channel configuration from first session
        if aggregated['multi_session_theta_data']:
            first_session = list(batch_results['processed_sessions'].values())[0]['results']
            channels = first_session['channels']
            roi_channels = first_session['roi_channels']
            
            # Create plots directory
            plot_save_path = f"{save_path}/variability_plots" if save_path else "variability_plots"
            
            # Generate different types of variability plots
            for plot_type in ['sem', 'boxplot', 'violin']:
                print(f"  Creating {plot_type} plots...")
                plot_theta_timecourses_with_variability(
                    aggregated['multi_session_theta_data'],
                    channels, roi_channels, plot_save_path, plot_type
                )
    
    return aggregated


# Example usage functions
def test_batch_processing():
    """
    Test batch processing on a small subset of sessions.
    """
    print("Testing batch processing...")
    
    # Get dataset info first
    info = get_dataset_info()
    if info is None:
        print("Could not load dataset info")
        return
    
    # Process first 3 sessions as test
    test_indices = [0, 1, 2] if info['n_sessions'] >= 3 else list(range(info['n_sessions']))
    
    batch_results = process_session_batch(
        pkl_path='data/processed/all_eeg_data.pkl',
        session_indices=test_indices,
        output_dir='test_batch_results',
        channels=list(range(8)),  # First 8 channels for speed
        roi_channels=[0, 1, 2, 3],  # First 4 as ROI
        plot_results=True,
        save_individual=True
    )
    
    # Aggregate results with variability plots
    aggregated = aggregate_batch_results(
        batch_results, 
        generate_variability_plots=True,
        save_path='test_batch_results'
    )
    
    # Save aggregated results
    with open('test_batch_results/aggregated_results.pkl', 'wb') as f:
        pickle.dump(aggregated, f)
    
    print("Test batch processing completed!")
    return batch_results, aggregated


if __name__ == "__main__":
    print("Batch Spectral Analysis Pipeline")
    print("=" * 50)
    
    # Show dataset info
    info = get_dataset_info()
    if info:
        print(f"Dataset contains {info['n_sessions']} sessions")
        print("\nExample sessions:")
        for session_info in info['session_examples']:
            print(f"  Session {session_info['index']}: {session_info['rat_id']} - {session_info['session_date']}")
            print(f"    EEG: {session_info['eeg_shape']}, Duration: {session_info['recording_duration']:.1f}s")
            print(f"    NM events: {session_info['n_nm_events']}, ITI events: {session_info['n_iti_events']}")
    
    print("\nAvailable functions:")
    print("- test_batch_processing(): Test on first few sessions")
    print("- process_all_sessions(): Process entire dataset") 
    print("- get_dataset_info(): Get dataset information")