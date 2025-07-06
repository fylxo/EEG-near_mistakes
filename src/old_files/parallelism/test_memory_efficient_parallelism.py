#!/usr/bin/env python3
"""
Memory-Efficient Multi-Session Parallelism Test

Test parallelism on the memory-efficient multi-session analysis pipeline
which is where we expect the biggest performance gains.
"""

import os
import sys
import time
import pickle
from typing import List, Dict, Tuple

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from nm_theta_multi_session_memory_efficient import (
        find_rat_sessions_from_loaded_data,
        process_single_session_and_save,
        aggregate_session_results
    )
    from electrode_utils import load_electrode_mappings
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import tempfile
    import shutil
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def process_session_parallel_wrapper(args):
    """Wrapper for parallel session processing."""
    (session_index, session_data, roi_or_channels, freq_range, n_freqs, 
     window_duration, n_cycles_factor, temp_dir, mapping_df) = args
    
    session_save_dir = os.path.join(temp_dir, f'session_{session_index}')
    
    try:
        summary = process_single_session_and_save(
            session_index=session_index,
            session_data=session_data,
            roi_or_channels=roi_or_channels,
            freq_range=freq_range,
            n_freqs=n_freqs,
            window_duration=window_duration,
            n_cycles_factor=n_cycles_factor,
            session_save_dir=session_save_dir,
            mapping_df=mapping_df
        )
        return {'success': True, 'summary': summary, 'error': None}
    except Exception as e:
        return {'success': False, 'summary': None, 'error': str(e)}


def analyze_rat_memory_efficient_parallel(
    rat_id: str,
    all_data: List[Dict],
    roi_or_channels,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0,
    n_jobs: int = 2,
    method: str = 'multiprocessing'
) -> Dict:
    """
    Parallel version of memory-efficient multi-session analysis.
    """
    print(f"🔄 PARALLEL MEMORY-EFFICIENT ANALYSIS (rat {rat_id})")
    print(f"   Method: {method}, Jobs: {n_jobs}")
    
    # Find sessions for this rat
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    if len(rat_sessions) == 0:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    print(f"   Found {len(rat_sessions)} sessions to process")
    
    # Create temporary directory for results
    temp_dir = tempfile.mkdtemp(prefix=f'parallel_rat_{rat_id}_')
    
    try:
        # Prepare arguments for parallel processing
        mapping_df = load_electrode_mappings()
        
        session_args = []
        for i, (session_index, session_data) in enumerate(zip(session_indices, rat_sessions)):
            args = (session_index, session_data, roi_or_channels, freq_range, n_freqs,
                   window_duration, n_cycles_factor, temp_dir, mapping_df)
            session_args.append(args)
        
        # Process sessions in parallel
        start_time = time.time()
        
        if method == 'multiprocessing':
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_session_parallel_wrapper, session_args))
        elif method == 'threading':
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_session_parallel_wrapper, session_args))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        processing_time = time.time() - start_time
        
        # Collect successful results
        session_summaries = []
        failed_sessions = []
        
        for i, result in enumerate(results):
            if result['success']:
                session_summaries.append(result['summary'])
            else:
                failed_sessions.append((session_indices[i], result['error']))
        
        print(f"   ⏱️ Parallel processing time: {processing_time:.2f}s")
        print(f"   ✅ Successful: {len(session_summaries)}, ❌ Failed: {len(failed_sessions)}")
        
        if failed_sessions:
            print(f"   Failed sessions: {[idx for idx, _ in failed_sessions]}")
        
        if not session_summaries:
            raise ValueError("No sessions processed successfully")
        
        # Aggregate results
        print("   🔄 Aggregating results...")
        aggregated_results = aggregate_session_results(session_summaries, rat_id)
        
        return {
            'aggregated_results': aggregated_results,
            'processing_time': processing_time,
            'successful_sessions': len(session_summaries),
            'failed_sessions': len(failed_sessions),
            'temp_dir': temp_dir  # Keep for cleanup
        }
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def analyze_rat_memory_efficient_sequential(
    rat_id: str,
    all_data: List[Dict],
    roi_or_channels,
    freq_range: Tuple[float, float] = (3, 8),
    n_freqs: int = 20,
    window_duration: float = 1.0,
    n_cycles_factor: float = 3.0
) -> Dict:
    """
    Sequential version for comparison.
    """
    print(f"🔄 SEQUENTIAL MEMORY-EFFICIENT ANALYSIS (rat {rat_id})")
    
    # Find sessions for this rat
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    if len(rat_sessions) == 0:
        raise ValueError(f"No sessions found for rat {rat_id}")
    
    print(f"   Found {len(rat_sessions)} sessions to process")
    
    # Create temporary directory for results
    temp_dir = tempfile.mkdtemp(prefix=f'sequential_rat_{rat_id}_')
    
    try:
        mapping_df = load_electrode_mappings()
        session_summaries = []
        failed_sessions = []
        
        start_time = time.time()
        
        # Process sessions sequentially
        for i, (session_index, session_data) in enumerate(zip(session_indices, rat_sessions)):
            print(f"   Processing session {i+1}/{len(rat_sessions)} (idx {session_index})")
            
            session_save_dir = os.path.join(temp_dir, f'session_{session_index}')
            
            try:
                summary = process_single_session_and_save(
                    session_index=session_index,
                    session_data=session_data,
                    roi_or_channels=roi_or_channels,
                    freq_range=freq_range,
                    n_freqs=n_freqs,
                    window_duration=window_duration,
                    n_cycles_factor=n_cycles_factor,
                    session_save_dir=session_save_dir,
                    mapping_df=mapping_df
                )
                session_summaries.append(summary)
                
            except Exception as e:
                print(f"   ❌ Session {session_index} failed: {e}")
                failed_sessions.append((session_index, str(e)))
        
        processing_time = time.time() - start_time
        
        print(f"   ⏱️ Sequential processing time: {processing_time:.2f}s")
        print(f"   ✅ Successful: {len(session_summaries)}, ❌ Failed: {len(failed_sessions)}")
        
        if not session_summaries:
            raise ValueError("No sessions processed successfully")
        
        # Aggregate results
        print("   🔄 Aggregating results...")
        aggregated_results = aggregate_session_results(session_summaries, rat_id)
        
        return {
            'aggregated_results': aggregated_results,
            'processing_time': processing_time,
            'successful_sessions': len(session_summaries),
            'failed_sessions': len(failed_sessions),
            'temp_dir': temp_dir  # Keep for cleanup
        }
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def test_memory_efficient_parallelism(data_path: str, rat_id: str = "531", max_sessions: int = 8):
    """Test memory-efficient parallelism with realistic workload."""
    print("🧠 MEMORY-EFFICIENT PARALLELISM TEST")
    print("=" * 50)
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Find sessions for rat
    session_indices, rat_sessions = find_rat_sessions_from_loaded_data(all_data, rat_id)
    
    if len(rat_sessions) < 3:
        print(f"❌ Not enough sessions for rat {rat_id} (found {len(rat_sessions)}, need ≥3)")
        return
    
    # Limit sessions for testing
    if max_sessions and len(rat_sessions) > max_sessions:
        rat_sessions = rat_sessions[:max_sessions]
        session_indices = session_indices[:max_sessions]
        all_data_subset = [all_data[i] if i < len(all_data) else None for i in session_indices]
        # Filter out None values
        all_data_subset = [session for session in all_data_subset if session is not None]
    else:
        all_data_subset = all_data
    
    roi_or_channels = 'frontal'  # Use ROI specification
    
    print(f"\nTesting rat {rat_id} with {len(rat_sessions)} sessions")
    print(f"ROI: {roi_or_channels}")
    
    results = {}
    
    # Test sequential
    print(f"\n{'='*60}")
    print("SEQUENTIAL BASELINE")
    print(f"{'='*60}")
    
    try:
        seq_result = analyze_rat_memory_efficient_sequential(
            rat_id, all_data_subset, roi_or_channels,
            freq_range=(3, 8), n_freqs=15, window_duration=1.0
        )
        results['sequential'] = seq_result
        
        print(f"✅ Sequential completed: {seq_result['processing_time']:.2f}s")
        
        # Cleanup temp directory
        shutil.rmtree(seq_result['temp_dir'], ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Sequential failed: {e}")
        return
    
    # Test parallel methods
    parallel_configs = [
        (2, 'multiprocessing'),
        (2, 'threading'),
        (4, 'multiprocessing'),
        (4, 'threading')
    ]
    
    for n_jobs, method in parallel_configs:
        print(f"\n{'='*60}")
        print(f"PARALLEL: {method.upper()} ({n_jobs} jobs)")
        print(f"{'='*60}")
        
        try:
            par_result = analyze_rat_memory_efficient_parallel(
                rat_id, all_data_subset, roi_or_channels,
                freq_range=(3, 8), n_freqs=15, window_duration=1.0,
                n_jobs=n_jobs, method=method
            )
            
            results[f'{method}_{n_jobs}'] = par_result
            
            # Calculate performance metrics
            seq_time = results['sequential']['processing_time']
            par_time = par_result['processing_time']
            speedup = seq_time / par_time
            efficiency = speedup / n_jobs * 100
            
            print(f"✅ Parallel completed: {par_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x {'✅' if speedup > 1.2 else '⚠️'}")
            print(f"   Efficiency: {efficiency:.1f}% {'✅' if efficiency > 60 else '⚠️'}")
            
            # Cleanup temp directory
            shutil.rmtree(par_result['temp_dir'], ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Parallel {method} ({n_jobs}) failed: {e}")
    
    # Results summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Method':<20} {'Time (s)':<10} {'Speedup':<10} {'Efficiency (%)':<15} {'Recommendation'}")
    print("-" * 70)
    
    seq_time = results['sequential']['processing_time']
    print(f"{'Sequential':<20} {seq_time:<10.2f} {'1.00':<10} {'100.0':<15} {'Baseline'}")
    
    recommendations = []
    
    for config_name, result in results.items():
        if config_name == 'sequential':
            continue
        
        par_time = result['processing_time']
        speedup = seq_time / par_time
        
        # Extract method and jobs from config name
        if 'multiprocessing' in config_name:
            method = 'multiprocessing'
            n_jobs = int(config_name.split('_')[-1])
        else:
            method = 'threading'
            n_jobs = int(config_name.split('_')[-1])
        
        efficiency = speedup / n_jobs * 100
        
        if speedup > 1.3:
            recommendation = "✅ Recommended"
            recommendations.append((config_name, speedup))
        elif speedup > 1.1:
            recommendation = "⚠️  Marginal gain"
        else:
            recommendation = "❌ Use sequential"
        
        display_name = f"{method} ({n_jobs}j)"
        print(f"{display_name:<20} {par_time:<10.2f} {speedup:<10.2f} {efficiency:<15.1f} {recommendation}")
    
    # Best recommendation
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if recommendations:
        best_config, best_speedup = max(recommendations, key=lambda x: x[1])
        print(f"✅ Best configuration: {best_config} ({best_speedup:.2f}x speedup)")
        
        # Estimate time savings for full analysis
        estimated_full_sessions = len(session_indices) if len(session_indices) > max_sessions else 20
        if estimated_full_sessions > len(rat_sessions):
            scaling_factor = estimated_full_sessions / len(rat_sessions)
            estimated_seq_time = seq_time * scaling_factor
            estimated_par_time = estimated_seq_time / best_speedup
            time_saved = estimated_seq_time - estimated_par_time
            
            print(f"⏰ For full rat analysis (~{estimated_full_sessions} sessions):")
            print(f"   Sequential time: ~{estimated_seq_time/60:.1f} minutes")
            print(f"   Parallel time: ~{estimated_par_time/60:.1f} minutes")
            print(f"   Time saved: ~{time_saved/60:.1f} minutes")
    else:
        print("⚠️  No significant speedup found. Use sequential processing.")
        print("   Consider testing with more sessions or larger ROIs")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test memory-efficient parallelism')
    parser.add_argument('--data_path', type=str,
                       default='data/processed/all_eeg_data.pkl',
                       help='Path to EEG data file')
    parser.add_argument('--rat_id', type=str, default='531',
                       help='Rat ID to test')
    parser.add_argument('--max_sessions', type=int, default=8,
                       help='Maximum sessions to test (for speed)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    try:
        test_memory_efficient_parallelism(args.data_path, args.rat_id, args.max_sessions)
        
        print(f"\n{'='*60}")
        print("✅ MEMORY-EFFICIENT PARALLELISM TEST COMPLETE")
        print("🎯 This test shows real-world performance on the")
        print("   memory-efficient multi-session pipeline")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)