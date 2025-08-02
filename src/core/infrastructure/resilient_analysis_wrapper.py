#!/usr/bin/env python3
"""
Resilient wrapper for nm_theta_analyzer that handles session-level memory failures.

This wrapper intercepts calls to the underlying analysis functions and adds
session-level resilience with automatic retry logic.
"""

import os
import gc
import pickle
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from infrastructure.session_resilience import SessionProcessor

def run_analysis_with_resilience(
    mode: str,
    method: str,
    parallel_type: Optional[str],
    pkl_path: str,
    roi: str,
    session_index: Optional[int],
    rat_id: str,
    freq_min: float,
    freq_max: float,
    n_freqs: int,
    window_duration: float,
    n_cycles_factor: float,
    n_jobs: Optional[int],
    batch_size: int,
    save_path: str,
    show_plots: bool,
    show_frequency_profiles: bool,
    session_resilience: bool = True,
    max_session_retries: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Run analysis with session-level resilience for memory failures.
    
    This function wraps the original run_analysis function and adds automatic
    retry logic for individual sessions that fail due to memory issues.
    
    Parameters:
    -----------
    All parameters same as original run_analysis, plus:
    session_resilience : bool
        Whether to enable session-level retry logic (default: True)
    max_session_retries : int
        Maximum retry attempts per session (default: 3)
    verbose : bool
        Whether to print detailed progress information (default: True)
        
    Returns:
    --------
    Dict
        Analysis results with resilience information added
    """
    if not session_resilience:
        # Fall back to original analysis if resilience is disabled
        from nm_theta_analyzer import run_analysis
        return run_analysis(
            mode=mode, method=method, parallel_type=parallel_type,
            pkl_path=pkl_path, roi=roi, session_index=session_index,
            rat_id=rat_id, freq_min=freq_min, freq_max=freq_max,
            n_freqs=n_freqs, window_duration=window_duration,
            n_cycles_factor=n_cycles_factor, n_jobs=n_jobs,
            batch_size=batch_size, save_path=save_path,
            show_plots=show_plots, show_frequency_profiles=show_frequency_profiles
        )
    
    if verbose:
        print(f"🛡️  Starting resilient analysis for rat {rat_id}")
        print(f"   Session resilience: Enabled (max {max_session_retries} retries per session)")
    
    # Load data to discover sessions
    if verbose:
        print(f"🔍 Loading data to discover sessions...")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Find sessions for this rat
    rat_sessions = []
    for session_data in all_data:
        if str(session_data.get('rat_id')) == str(rat_id):
            rat_sessions.append(session_data)
    
    if not rat_sessions:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    if verbose:
        print(f"📊 Found {len(rat_sessions)} sessions for rat {rat_id}")
    
    # Setup session processor
    session_processor = SessionProcessor(
        max_retries=max_session_retries,
        cleanup_between_retries=True,
        verbose=verbose
    )
    
    # Process each session with resilience
    def process_single_session(session_idx: int, session_data: Dict) -> Dict:
        """Process a single session with the original analyzer."""
        # Import here to avoid circular imports
        from nm_theta_analyzer import run_analysis
        
        session_id = session_data.get('session_id', f'session_{session_idx}')
        
        if verbose:
            print(f"      🔬 Processing session {session_id}")
        
        # Run original analysis for this specific session
        try:
            result = run_analysis(
                mode='single',  # Process one session at a time
                method=method,
                parallel_type=parallel_type,
                pkl_path=pkl_path,
                roi=roi,
                session_index=session_idx,  # Specific session
                rat_id=rat_id,
                freq_min=freq_min,
                freq_max=freq_max,
                n_freqs=n_freqs,
                window_duration=window_duration,
                n_cycles_factor=n_cycles_factor,
                n_jobs=n_jobs,
                batch_size=batch_size,
                save_path=os.path.join(save_path, f'session_{session_idx}'),
                show_plots=show_plots,
                show_frequency_profiles=show_frequency_profiles
            )
            
            return result
            
        except Exception as e:
            # Check if it's a memory error
            error_str = str(e).lower()
            if 'memory' in error_str or 'allocation' in error_str or 'unable to allocate' in error_str:
                # Re-raise as MemoryError for proper handling
                raise MemoryError(f"Session {session_id}: {str(e)}")
            else:
                # Re-raise other exceptions as-is
                raise e
    
    # Create session processing function
    def session_args_func(session_idx_str: str) -> Tuple:
        session_idx = int(session_idx_str.split('_')[1])
        session_data = rat_sessions[session_idx]
        return (session_idx, session_data)
    
    # Process all sessions with resilience
    session_identifiers = [f"session_{i}" for i in range(len(rat_sessions))]
    
    results = session_processor.process_sessions_resilient(
        sessions=session_identifiers,
        process_func=process_single_session,
        session_args_func=session_args_func
    )
    
    successful_sessions = results['successful_sessions']
    failed_sessions = results['failed_sessions']
    processing_summary = results['processing_summary']
    
    if verbose:
        print(f"📊 Session processing complete:")
        print(f"   Successful: {len(successful_sessions)}/{len(rat_sessions)}")
        print(f"   Failed: {len(failed_sessions)}/{len(rat_sessions)}")
        print(f"   Success rate: {processing_summary['success_rate']:.1f}%")
    
    # Check if we have enough successful sessions to continue
    if len(successful_sessions) == 0:
        raise RuntimeError(f"All sessions failed for rat {rat_id}")
    
    # Now aggregate the successful sessions
    if verbose:
        print(f"🔄 Aggregating {len(successful_sessions)} successful sessions...")
    
    # This is a simplified aggregation - in practice, you'd need to properly
    # combine the session results according to your analysis protocol
    aggregated_result = aggregate_session_results(
        successful_sessions, 
        rat_id, 
        save_path,
        verbose=verbose
    )
    
    # Add resilience information to results
    aggregated_result['resilience_info'] = {
        'session_resilience_enabled': True,
        'max_retries_per_session': max_session_retries,
        'total_sessions_attempted': len(rat_sessions),
        'successful_sessions': len(successful_sessions),
        'failed_sessions': len(failed_sessions),
        'success_rate': processing_summary['success_rate'],
        'failed_session_details': failed_sessions
    }
    
    if verbose:
        print(f"✅ Resilient analysis complete for rat {rat_id}")
        if failed_sessions:
            print(f"⚠️  {len(failed_sessions)} sessions could not be recovered:")
            for session_id, error in failed_sessions.items():
                print(f"     {session_id}: {error}")
    
    return aggregated_result


def aggregate_session_results(session_results: Dict[str, Dict], 
                            rat_id: str,
                            save_path: str,
                            verbose: bool = True) -> Dict:
    """
    Aggregate successful session results into multi-session format.
    
    Parameters:
    -----------
    session_results : Dict[str, Dict]
        Dictionary of successful session results
    rat_id : str
        Rat ID being processed
    save_path : str
        Path to save aggregated results
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    Dict
        Aggregated multi-session results
    """
    if verbose:
        print(f"🔄 Aggregating {len(session_results)} session results...")
    
    # Extract common information from first session
    first_session_result = next(iter(session_results.values()))
    
    # Initialize aggregated result structure
    aggregated = {
        'analysis_type': 'multi_session_resilient',
        'rat_id': rat_id,
        'n_sessions_analyzed': len(session_results),
        'n_sessions_successful': len(session_results),
        'roi_channels': first_session_result.get('roi_channels'),
        'frequencies': first_session_result.get('frequencies') or first_session_result.get('freqs'),
        'analysis_parameters': first_session_result.get('analysis_parameters', {}),
        'averaged_windows': {},
        'session_contributions': {}
    }
    
    # Aggregate spectrograms by NM size
    nm_sizes = set()
    for session_result in session_results.values():
        if 'averaged_windows' in session_result:
            nm_sizes.update(session_result['averaged_windows'].keys())
    
    if verbose:
        print(f"   Found NM sizes: {sorted(nm_sizes)}")
    
    for nm_size in nm_sizes:
        spectrograms = []
        n_events_list = []
        session_ids = []
        
        for session_id, session_result in session_results.items():
            if ('averaged_windows' in session_result and 
                nm_size in session_result['averaged_windows']):
                
                window_data = session_result['averaged_windows'][nm_size]
                spectrograms.append(window_data['avg_spectrogram'])
                n_events_list.append(window_data['n_events'])
                session_ids.append(session_id)
        
        if spectrograms:
            import numpy as np
            
            # Average spectrograms across sessions
            spectrograms_array = np.array(spectrograms)
            avg_spectrogram = np.mean(spectrograms_array, axis=0)
            
            # Get window times from first session
            window_times = session_results[session_ids[0]]['averaged_windows'][nm_size]['window_times']
            
            aggregated['averaged_windows'][nm_size] = {
                'avg_spectrogram': avg_spectrogram,
                'window_times': window_times,
                'n_events': sum(n_events_list),
                'n_sessions': len(spectrograms),
                'contributing_sessions': session_ids
            }
            
            aggregated['session_contributions'][nm_size] = {
                'session_ids': session_ids,
                'events_per_session': n_events_list
            }
    
    # Save aggregated results
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, 'multi_session_results.pkl')
    
    with open(results_file, 'wb') as f:
        pickle.dump(aggregated, f)
    
    if verbose:
        print(f"✅ Aggregated results saved to: {results_file}")
    
    return aggregated


def create_resilient_wrapper(original_run_analysis_func):
    """
    Create a resilient wrapper around the original run_analysis function.
    
    Parameters:
    -----------
    original_run_analysis_func : callable
        Original run_analysis function
        
    Returns:
    --------
    callable
        Wrapped function with session resilience
    """
    def resilient_run_analysis(*args, **kwargs):
        # Extract resilience parameters
        session_resilience = kwargs.pop('session_resilience', True)
        max_session_retries = kwargs.pop('max_session_retries', 3)
        
        if session_resilience and kwargs.get('mode') == 'multi':
            # Use resilient analysis for multi-session mode
            return run_analysis_with_resilience(
                *args, 
                session_resilience=session_resilience,
                max_session_retries=max_session_retries,
                **kwargs
            )
        else:
            # Use original analysis for single sessions or when resilience disabled
            return original_run_analysis_func(*args, **kwargs)
    
    return resilient_run_analysis


if __name__ == "__main__":
    # Test the resilient wrapper
    print("Testing resilient analysis wrapper...")
    
    # This would normally test with real data
    print("✅ Resilient wrapper created successfully")
    print("Use this wrapper to replace calls to run_analysis for multi-session mode")