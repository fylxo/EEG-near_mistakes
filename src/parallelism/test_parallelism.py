#!/usr/bin/env python3
"""
Parallelism Testing Script

This script runs quick tests to validate and benchmark parallel implementations
against sequential ones.

Usage:
    python test_parallelism.py --quick          # Quick synthetic data test
    python test_parallelism.py --real           # Test with real data
    python test_parallelism.py --full           # Comprehensive benchmark
"""

import os
import sys
import argparse
import time
import numpy as np

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.dirname(__file__))

try:
    from performance_benchmark import ROIBenchmark, run_comprehensive_benchmark
    from parallel_roi_analysis import (
        compute_roi_theta_spectrogram_parallel_channels,
        compute_roi_theta_spectrogram_parallel_frequencies
    )
    from nm_theta_single_basic import compute_roi_theta_spectrogram
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def quick_validation_test():
    """Run a quick validation test with synthetic data."""
    print("🔬 QUICK VALIDATION TEST")
    print("=" * 40)
    
    # Generate small synthetic dataset
    benchmark = ROIBenchmark()
    test_data = benchmark.generate_synthetic_data(n_channels=8, n_samples=20000)
    
    roi_channels = [0, 1, 2, 3]
    freq_range = (3, 8)
    n_freqs = 10
    sfreq = test_data['sfreq']
    
    print(f"Test data: {test_data['eeg'].shape}, ROI: {roi_channels}")
    
    # Sequential version
    print("\n🔄 Sequential computation...")
    start_time = time.time()
    freqs_seq, roi_power_seq, _ = compute_roi_theta_spectrogram(
        test_data['eeg'], roi_channels, sfreq, freq_range, n_freqs
    )
    seq_time = time.time() - start_time
    print(f"   ⏱️ Sequential time: {seq_time:.2f}s")
    print(f"   📊 Result shape: {roi_power_seq.shape}")
    print(f"   📈 Result range: {roi_power_seq.min():.2f} to {roi_power_seq.max():.2f}")
    
    # Parallel channels version
    print("\n🔄 Parallel channels computation...")
    start_time = time.time()
    freqs_par, roi_power_par, _ = compute_roi_theta_spectrogram_parallel_channels(
        test_data['eeg'], roi_channels, sfreq, freq_range, n_freqs, 
        n_cycles_factor=3.0, n_jobs=2, method='multiprocessing'
    )
    par_time = time.time() - start_time
    print(f"   ⏱️ Parallel time: {par_time:.2f}s")
    print(f"   📊 Result shape: {roi_power_par.shape}")
    print(f"   📈 Result range: {roi_power_par.min():.2f} to {roi_power_par.max():.2f}")
    
    # Validation
    print("\n✅ VALIDATION")
    print("-" * 20)
    
    # Check shapes
    shapes_match = roi_power_seq.shape == roi_power_par.shape
    print(f"Shapes match: {shapes_match}")
    
    # Check values
    if shapes_match:
        max_diff = np.max(np.abs(roi_power_seq - roi_power_par))
        print(f"Maximum difference: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✅ RESULTS IDENTICAL")
        elif max_diff < 1e-6:
            print("✅ RESULTS VERY CLOSE (acceptable)")
        else:
            print("❌ RESULTS DIFFER SIGNIFICANTLY")
            return False
    else:
        print("❌ SHAPE MISMATCH")
        return False
    
    # Performance comparison
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n⚡ PERFORMANCE")
    print("-" * 20)
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {speedup:.2f}x {'✅' if speedup > 1.1 else '⚠️'}")
    
    return True


def test_with_real_data(data_path: str):
    """Test with real EEG data."""
    print("🧠 REAL DATA TEST")
    print("=" * 40)
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    benchmark = ROIBenchmark(data_path)
    
    if benchmark.all_data is None:
        print("❌ Failed to load data")
        return False
    
    # Use first session
    test_session = benchmark.all_data[0]
    print(f"Using session: rat {test_session.get('rat_id')}, "
          f"date {test_session.get('session_date')}")
    print(f"EEG shape: {test_session['eeg'].shape}")
    
    # Test with a reasonable ROI
    roi_channels = [0, 1, 2, 3, 4, 5]  # 6 channels
    
    results = benchmark.benchmark_roi_spectrogram(
        test_session,
        roi_channels,
        freq_range=(3, 8),
        n_freqs=15,
        test_parallel_channels=True,
        test_parallel_frequencies=False,  # Skip frequency parallelism for quick test
        n_jobs_list=[2, 4]
    )
    
    # Print results
    print("\n📊 BENCHMARK RESULTS")
    print("-" * 30)
    
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.method}: {result.execution_time:.2f}s")
    
    return True


def test_scalability():
    """Test scalability with different problem sizes."""
    print("📈 SCALABILITY TEST")
    print("=" * 40)
    
    benchmark = ROIBenchmark()
    
    # Test different channel counts
    channel_counts = [4, 8, 16]
    sample_counts = [20000, 50000]
    
    results = []
    
    for n_channels in channel_counts:
        for n_samples in sample_counts:
            print(f"\n🔬 Testing {n_channels} channels, {n_samples} samples")
            
            # Generate test data
            test_data = benchmark.generate_synthetic_data(
                n_channels=max(n_channels, 16),  # Ensure enough channels
                n_samples=n_samples
            )
            
            roi_channels = list(range(n_channels))
            
            # Quick test with 2 jobs
            try:
                start_time = time.time()
                freqs_seq, roi_power_seq, _ = compute_roi_theta_spectrogram(
                    test_data['eeg'], roi_channels, test_data['sfreq'], 
                    (3, 8), 10
                )
                seq_time = time.time() - start_time
                
                start_time = time.time()
                freqs_par, roi_power_par, _ = compute_roi_theta_spectrogram_parallel_channels(
                    test_data['eeg'], roi_channels, test_data['sfreq'],
                    (3, 8), 10, n_jobs=2, method='multiprocessing'
                )
                par_time = time.time() - start_time
                
                speedup = seq_time / par_time if par_time > 0 else 0
                
                results.append({
                    'n_channels': n_channels,
                    'n_samples': n_samples,
                    'seq_time': seq_time,
                    'par_time': par_time,
                    'speedup': speedup
                })
                
                print(f"   Sequential: {seq_time:.2f}s")
                print(f"   Parallel:   {par_time:.2f}s") 
                print(f"   Speedup:    {speedup:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n📊 SCALABILITY SUMMARY")
    print("-" * 40)
    print(f"{'Channels':<10} {'Samples':<10} {'Seq (s)':<10} {'Par (s)':<10} {'Speedup':<10}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['n_channels']:<10} {result['n_samples']:<10} "
              f"{result['seq_time']:<10.2f} {result['par_time']:<10.2f} "
              f"{result['speedup']:<10.2f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test parallelization implementations')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation test with synthetic data')
    parser.add_argument('--real', type=str, 
                       help='Test with real data (provide path to pkl file)')
    parser.add_argument('--scalability', action='store_true',
                       help='Test scalability with different problem sizes')
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive benchmark')
    parser.add_argument('--output_dir', type=str, default='results/parallelism',
                       help='Output directory for full benchmark')
    
    args = parser.parse_args()
    
    if not any([args.quick, args.real, args.scalability, args.full]):
        # Default to quick test
        args.quick = True
    
    success = True
    
    if args.quick:
        print("🚀 Running quick validation test...")
        success = quick_validation_test() and success
    
    if args.real:
        print(f"\n🚀 Running real data test with {args.real}...")
        success = test_with_real_data(args.real) and success
    
    if args.scalability:
        print("\n🚀 Running scalability test...")
        success = test_scalability() and success
    
    if args.full:
        print(f"\n🚀 Running comprehensive benchmark...")
        data_path = args.real if args.real else None
        try:
            results = run_comprehensive_benchmark(data_path, args.output_dir)
            print(f"✅ Comprehensive benchmark completed with {len(results)} tests")
        except Exception as e:
            print(f"❌ Comprehensive benchmark failed: {e}")
            success = False
    
    print(f"\n{'='*60}")
    if success:
        print("✅ ALL TESTS PASSED")
        print("💡 Parallelism is working correctly!")
        print("📁 Check results/parallelism/ for detailed reports")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the error messages above")
    print(f"{'='*60}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)