#!/usr/bin/env python3
"""
Multi-Session Parallelism Test

Test parallelism performance on realistic multi-session scenarios where
we expect the biggest gains.
"""

import os
import sys
import time
import numpy as np
import pickle
from typing import List, Tuple, Dict

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.dirname(__file__))

try:
    from parallel_roi_analysis import analyze_rat_multi_session_parallel
    from nm_theta_single_basic import analyze_session_nm_theta_roi
    from electrode_utils import load_electrode_mappings
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def load_rat_sessions(data_path: str, rat_id: str, max_sessions: int = None) -> List[Tuple[int, Dict]]:
    """Load sessions for a specific rat."""
    print(f"Loading sessions for rat {rat_id}...")
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    
    rat_sessions = []
    for i, session in enumerate(all_data):
        if session.get('rat_id') == rat_id:
            rat_sessions.append((i, session))
            if max_sessions and len(rat_sessions) >= max_sessions:
                break
    
    print(f"Found {len(rat_sessions)} sessions for rat {rat_id}")
    return rat_sessions


def time_sequential_multisession(rat_sessions: List[Tuple[int, Dict]], 
                                roi_channels: List[int]) -> Tuple[float, List]:
    """Time sequential multi-session processing."""
    print(f"\n🔄 SEQUENTIAL MULTI-SESSION PROCESSING")
    print(f"   Processing {len(rat_sessions)} sessions sequentially...")
    
    start_time = time.time()
    results = []
    
    for i, (session_idx, session_data) in enumerate(rat_sessions):
        print(f"   Processing session {i+1}/{len(rat_sessions)} (idx {session_idx})")
        
        try:
            result = analyze_session_nm_theta_roi(
                session_data=session_data,
                roi_or_channels=roi_channels,
                freq_range=(3, 8),
                n_freqs=15,
                window_duration=1.0,
                n_cycles_factor=3.0,
                save_path=None,
                mapping_df=None,
                show_plots=False
            )
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Session {session_idx} failed: {e}")
    
    total_time = time.time() - start_time
    print(f"   ⏱️ Sequential total time: {total_time:.2f}s")
    print(f"   ✅ Successful sessions: {len(results)}/{len(rat_sessions)}")
    
    return total_time, results


def time_parallel_multisession(rat_sessions: List[Tuple[int, Dict]], 
                              roi_channels: List[int],
                              n_jobs: int = 2,
                              batch_size: int = None) -> Tuple[float, Dict]:
    """Time parallel multi-session processing."""
    print(f"\n🔄 PARALLEL MULTI-SESSION PROCESSING ({n_jobs} jobs)")
    print(f"   Processing {len(rat_sessions)} sessions in parallel...")
    
    start_time = time.time()
    
    try:
        results = analyze_rat_multi_session_parallel(
            rat_sessions=rat_sessions,
            roi_or_channels=roi_channels,
            freq_range=(3, 8),
            n_freqs=15,
            window_duration=1.0,
            n_cycles_factor=3.0,
            mapping_df=None,
            n_jobs=n_jobs,
            batch_size=batch_size
        )
        
        total_time = time.time() - start_time
        print(f"   ⏱️ Parallel total time: {total_time:.2f}s")
        print(f"   ✅ Successful sessions: {results['successful_sessions']}")
        
        return total_time, results
        
    except Exception as e:
        print(f"   ❌ Parallel processing failed: {e}")
        return float('inf'), {}


def test_multisession_scalability(data_path: str, rat_id: str = "531"):
    """Test multi-session parallelism with different configurations."""
    print("🧠 MULTI-SESSION PARALLELISM TEST")
    print("=" * 50)
    
    # Load data for rat
    rat_sessions = load_rat_sessions(data_path, rat_id, max_sessions=10)  # Limit for testing
    
    if len(rat_sessions) < 3:
        print(f"❌ Not enough sessions for rat {rat_id} (need at least 3)")
        return
    
    roi_channels = [0, 1, 2, 3, 4, 5]  # 6-channel ROI
    
    print(f"\nTesting with rat {rat_id}: {len(rat_sessions)} sessions, {len(roi_channels)} channels")
    
    # Test different session counts
    session_counts = [3, 6, len(rat_sessions)]
    
    for n_sessions in session_counts:
        if n_sessions > len(rat_sessions):
            continue
            
        print(f"\n{'='*60}")
        print(f"TESTING {n_sessions} SESSIONS")
        print(f"{'='*60}")
        
        test_sessions = rat_sessions[:n_sessions]
        
        # Sequential baseline
        seq_time, seq_results = time_sequential_multisession(test_sessions, roi_channels)
        
        if seq_time == float('inf') or len(seq_results) == 0:
            print(f"❌ Sequential failed, skipping parallel tests")
            continue
        
        # Parallel tests
        job_configs = [2, 4]
        
        for n_jobs in job_configs:
            par_time, par_results = time_parallel_multisession(
                test_sessions, roi_channels, n_jobs=n_jobs
            )
            
            if par_time != float('inf'):
                speedup = seq_time / par_time
                efficiency = speedup / n_jobs * 100
                
                print(f"\n📊 PERFORMANCE COMPARISON ({n_jobs} jobs)")
                print(f"   Sequential:    {seq_time:.2f}s")
                print(f"   Parallel:      {par_time:.2f}s")
                print(f"   Speedup:       {speedup:.2f}x {'✅' if speedup > 1.2 else '⚠️'}")
                print(f"   Efficiency:    {efficiency:.1f}% {'✅' if efficiency > 60 else '⚠️'}")
                
                # Estimate time for full analysis
                if rat_id in ["531", "532", "441", "442"]:  # Rats with many sessions
                    estimated_full_sessions = 30  # Typical session count
                    estimated_seq_time = seq_time * (estimated_full_sessions / n_sessions)
                    estimated_par_time = par_time * (estimated_full_sessions / n_sessions)
                    
                    print(f"\n⏰ ESTIMATED FULL RAT ANALYSIS (~{estimated_full_sessions} sessions)")
                    print(f"   Sequential:    {estimated_seq_time/60:.1f} minutes")
                    print(f"   Parallel:      {estimated_par_time/60:.1f} minutes") 
                    print(f"   Time saved:    {(estimated_seq_time-estimated_par_time)/60:.1f} minutes")


def test_different_roi_sizes(data_path: str, rat_id: str = "531"):
    """Test how parallelism scales with different ROI sizes."""
    print("\n🔬 ROI SIZE SCALING TEST")
    print("=" * 40)
    
    rat_sessions = load_rat_sessions(data_path, rat_id, max_sessions=4)  # Small test
    
    if len(rat_sessions) < 3:
        print(f"❌ Not enough sessions for rat {rat_id}")
        return
    
    test_sessions = rat_sessions[:3]  # Use 3 sessions
    
    # Test different ROI sizes
    roi_configs = [
        ([0, 1], "2-channel ROI"),
        ([0, 1, 2, 3], "4-channel ROI"), 
        ([0, 1, 2, 3, 4, 5, 6, 7], "8-channel ROI"),
        (list(range(16)), "16-channel ROI")
    ]
    
    results = []
    
    for roi_channels, roi_name in roi_configs:
        print(f"\n📊 Testing {roi_name} ({len(roi_channels)} channels)")
        
        # Sequential
        seq_time, seq_results = time_sequential_multisession(test_sessions, roi_channels)
        
        if seq_time == float('inf'):
            print(f"   ❌ Sequential failed")
            continue
        
        # Parallel with 4 jobs
        par_time, par_results = time_parallel_multisession(
            test_sessions, roi_channels, n_jobs=4
        )
        
        if par_time != float('inf'):
            speedup = seq_time / par_time
            results.append({
                'roi_name': roi_name,
                'channels': len(roi_channels),
                'seq_time': seq_time,
                'par_time': par_time,
                'speedup': speedup
            })
            
            print(f"   Sequential: {seq_time:.2f}s")
            print(f"   Parallel:   {par_time:.2f}s")
            print(f"   Speedup:    {speedup:.2f}x")
    
    # Summary
    print(f"\n📈 ROI SCALING SUMMARY")
    print("-" * 50)
    print(f"{'ROI Size':<20} {'Channels':<10} {'Speedup':<10} {'Recommendation'}")
    print("-" * 50)
    
    for result in results:
        recommendation = "✅ Use parallel" if result['speedup'] > 1.2 else "⚠️  Use sequential"
        print(f"{result['roi_name']:<20} {result['channels']:<10} "
              f"{result['speedup']:<10.2f} {recommendation}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-session parallelism')
    parser.add_argument('--data_path', type=str, 
                       default='data/processed/all_eeg_data.pkl',
                       help='Path to EEG data file')
    parser.add_argument('--rat_id', type=str, default='531',
                       help='Rat ID to test')
    parser.add_argument('--test_scaling', action='store_true',
                       help='Test ROI size scaling')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return 1
    
    try:
        # Main multi-session test
        test_multisession_scalability(args.data_path, args.rat_id)
        
        # ROI scaling test
        if args.test_scaling:
            test_different_roi_sizes(args.data_path, args.rat_id)
        
        print(f"\n{'='*60}")
        print("✅ MULTI-SESSION PARALLELISM TEST COMPLETE")
        print("💡 Use parallel processing for:")
        print("   - Multiple sessions (≥6 sessions)")
        print("   - Larger ROIs (≥8 channels)")
        print("   - Full rat analysis")
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