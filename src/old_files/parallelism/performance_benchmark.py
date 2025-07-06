#!/usr/bin/env python3
"""
Performance Benchmark Module

This module provides comprehensive benchmarking and validation tools to compare
sequential vs parallel implementations of EEG analysis functions.

Features:
- Performance timing comparisons
- Result validation (ensuring parallel gives same results as sequential)
- Memory usage monitoring
- Scalability analysis
- Detailed reporting
"""

import numpy as np
import time
import psutil
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Callable
import pickle
import matplotlib.pyplot as plt
from dataclasses import dataclass
from contextlib import contextmanager

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

try:
    from nm_theta_single_basic import compute_roi_theta_spectrogram, analyze_session_nm_theta_roi
    from nm_theta_multi_session import analyze_rat_multi_session
    from electrode_utils import get_channels, load_electrode_mappings
    from parallel_roi_analysis import (
        compute_roi_theta_spectrogram_parallel_channels,
        compute_roi_theta_spectrogram_parallel_frequencies,
        analyze_rat_multi_session_parallel
    )
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure you're running from the project root directory")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    function_name: str
    method: str  # 'sequential', 'parallel_channels', 'parallel_frequencies', etc.
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    result_shape: Tuple
    result_stats: Dict  # min, max, mean, std of the main result array
    parameters: Dict
    success: bool
    error_message: Optional[str] = None


@contextmanager
def monitor_resources():
    """Context manager to monitor CPU and memory usage."""
    process = psutil.Process(os.getpid())
    
    # Initial measurements
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_cpu = process.cpu_percent()
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        # Store results (you can access these in the calling function)
        global _last_benchmark_stats
        _last_benchmark_stats = {
            'execution_time': end_time - start_time,
            'memory_usage_mb': final_memory - initial_memory,
            'cpu_usage_percent': final_cpu
        }


def validate_results_equal(result1: np.ndarray, result2: np.ndarray, 
                          tolerance: float = 1e-10, name: str = "arrays") -> bool:
    """
    Validate that two arrays are approximately equal.
    
    Parameters:
    -----------
    result1, result2 : np.ndarray
        Arrays to compare
    tolerance : float
        Tolerance for numerical differences
    name : str
        Name for error reporting
    
    Returns:
    --------
    is_equal : bool
        True if arrays are approximately equal
    """
    try:
        if result1.shape != result2.shape:
            print(f"❌ Shape mismatch for {name}: {result1.shape} vs {result2.shape}")
            return False
        
        max_diff = np.max(np.abs(result1 - result2))
        if max_diff > tolerance:
            print(f"❌ Values differ for {name}: max difference = {max_diff:.2e} > {tolerance:.2e}")
            print(f"   Result1 range: {result1.min():.2e} to {result1.max():.2e}")
            print(f"   Result2 range: {result2.min():.2e} to {result2.max():.2e}")
            return False
        
        print(f"✅ {name} match within tolerance {tolerance:.2e} (max diff: {max_diff:.2e})")
        return True
        
    except Exception as e:
        print(f"❌ Error comparing {name}: {e}")
        return False


def compute_result_stats(array: np.ndarray) -> Dict:
    """Compute basic statistics for result validation."""
    return {
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'shape': array.shape
    }


class ROIBenchmark:
    """Benchmark class for ROI analysis functions."""
    
    def __init__(self, test_data_path: str = None):
        """
        Initialize benchmark with test data.
        
        Parameters:
        -----------
        test_data_path : str
            Path to pickle file containing test EEG data
        """
        self.test_data_path = test_data_path
        self.results = []
        
        if test_data_path and os.path.exists(test_data_path):
            self.load_test_data()
        else:
            print("⚠️  No test data loaded. Use load_test_data() or generate_synthetic_data()")
    
    def load_test_data(self):
        """Load test data from pickle file."""
        try:
            with open(self.test_data_path, 'rb') as f:
                self.all_data = pickle.load(f)
            print(f"✅ Loaded {len(self.all_data)} sessions from {self.test_data_path}")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.all_data = None
    
    def generate_synthetic_data(self, n_channels: int = 32, n_samples: int = 50000, 
                               sfreq: float = 200.0) -> Dict:
        """
        Generate synthetic EEG data for testing.
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        n_samples : int
            Number of time samples
        sfreq : float
            Sampling frequency
        
        Returns:
        --------
        synthetic_data : Dict
            Synthetic session data
        """
        print(f"🔧 Generating synthetic data: {n_channels} channels, {n_samples} samples")
        
        # Create realistic EEG-like data with multiple frequency components
        time_vector = np.arange(n_samples) / sfreq
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Add multiple frequency components with noise
            signal = (
                0.5 * np.sin(2 * np.pi * 4 * time_vector) +  # 4 Hz theta
                0.3 * np.sin(2 * np.pi * 8 * time_vector) +  # 8 Hz alpha
                0.2 * np.sin(2 * np.pi * 15 * time_vector) + # 15 Hz beta
                0.1 * np.random.randn(n_samples)              # Noise
            )
            eeg_data[ch, :] = signal
        
        # Generate some random NM events
        n_nm_events = 50
        nm_peak_times = np.sort(np.random.uniform(10, time_vector[-1] - 10, n_nm_events))
        nm_sizes = np.random.choice([1, 2, 3], n_nm_events)
        
        synthetic_data = {
            'rat_id': 'synthetic',
            'session_date': 'test',
            'eeg': eeg_data,
            'eeg_time': time_vector.reshape(1, -1),
            'nm_peak_times': nm_peak_times,
            'nm_sizes': nm_sizes,
            'sfreq': sfreq
        }
        
        return synthetic_data
    
    def benchmark_roi_spectrogram(self, 
                                 session_data: Dict,
                                 roi_channels: List[int],
                                 freq_range: Tuple[float, float] = (3, 8),
                                 n_freqs: int = 20,
                                 n_cycles_factor: float = 3.0,
                                 test_parallel_channels: bool = True,
                                 test_parallel_frequencies: bool = True,
                                 n_jobs_list: List[int] = None) -> List[BenchmarkResult]:
        """
        Benchmark ROI spectrogram computation with different methods.
        
        Parameters:
        -----------
        session_data : Dict
            EEG session data
        roi_channels : List[int]
            ROI channel indices
        freq_range : Tuple[float, float]
            Frequency range
        n_freqs : int
            Number of frequencies
        n_cycles_factor : float
            Cycles factor
        test_parallel_channels : bool
            Test parallel channel processing
        test_parallel_frequencies : bool
            Test parallel frequency processing
        n_jobs_list : List[int]
            List of job counts to test
        
        Returns:
        --------
        benchmark_results : List[BenchmarkResult]
            Results for each tested method
        """
        print(f"\n🏁 BENCHMARKING ROI SPECTROGRAM COMPUTATION")
        print(f"   ROI: {len(roi_channels)} channels")
        print(f"   Frequencies: {n_freqs} ({freq_range[0]}-{freq_range[1]} Hz)")
        
        eeg_data = session_data['eeg']
        sfreq = session_data.get('sfreq', 200.0)
        
        if n_jobs_list is None:
            n_jobs_list = [1, 2, 4, None]  # None = auto-detect
        
        results = []
        sequential_result = None
        
        # 1. Sequential (baseline)
        print(f"\n📊 Testing Sequential Implementation...")
        try:
            with monitor_resources():
                freqs_seq, roi_power_seq, channel_powers_seq = compute_roi_theta_spectrogram(
                    eeg_data, roi_channels, sfreq, freq_range, n_freqs, n_cycles_factor
                )
            
            sequential_result = roi_power_seq
            stats = _last_benchmark_stats
            
            result = BenchmarkResult(
                function_name="compute_roi_theta_spectrogram",
                method="sequential",
                execution_time=stats['execution_time'],
                memory_usage_mb=stats['memory_usage_mb'],
                cpu_usage_percent=stats['cpu_usage_percent'],
                result_shape=roi_power_seq.shape,
                result_stats=compute_result_stats(roi_power_seq),
                parameters={'roi_channels': len(roi_channels), 'n_freqs': n_freqs},
                success=True
            )
            results.append(result)
            
            print(f"   ⏱️ Time: {stats['execution_time']:.2f}s")
            print(f"   💾 Memory: {stats['memory_usage_mb']:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Sequential failed: {e}")
            result = BenchmarkResult(
                function_name="compute_roi_theta_spectrogram",
                method="sequential",
                execution_time=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                result_shape=(0,),
                result_stats={},
                parameters={},
                success=False,
                error_message=str(e)
            )
            results.append(result)
        
        # 2. Parallel Channels
        if test_parallel_channels and sequential_result is not None:
            for n_jobs in n_jobs_list:
                for method in ['multiprocessing', 'threading']:
                    print(f"\n📊 Testing Parallel Channels ({method}, {n_jobs} jobs)...")
                    try:
                        with monitor_resources():
                            freqs_par, roi_power_par, channel_powers_par = (
                                compute_roi_theta_spectrogram_parallel_channels(
                                    eeg_data, roi_channels, sfreq, freq_range, 
                                    n_freqs, n_cycles_factor, n_jobs, method
                                )
                            )
                        
                        stats = _last_benchmark_stats
                        
                        # Validate results
                        is_valid = validate_results_equal(
                            sequential_result, roi_power_par, 
                            tolerance=1e-10, name=f"parallel_channels_{method}_{n_jobs}"
                        )
                        
                        result = BenchmarkResult(
                            function_name="compute_roi_theta_spectrogram_parallel_channels",
                            method=f"parallel_channels_{method}",
                            execution_time=stats['execution_time'],
                            memory_usage_mb=stats['memory_usage_mb'],
                            cpu_usage_percent=stats['cpu_usage_percent'],
                            result_shape=roi_power_par.shape,
                            result_stats=compute_result_stats(roi_power_par),
                            parameters={'roi_channels': len(roi_channels), 'n_freqs': n_freqs, 
                                      'n_jobs': n_jobs, 'method': method},
                            success=is_valid
                        )
                        results.append(result)
                        
                        speedup = stats['execution_time'] / results[0].execution_time if results[0].execution_time > 0 else 0
                        print(f"   ⏱️ Time: {stats['execution_time']:.2f}s (speedup: {1/speedup:.2f}x)")
                        print(f"   💾 Memory: {stats['memory_usage_mb']:.1f} MB")
                        print(f"   ✅ Validation: {'PASS' if is_valid else 'FAIL'}")
                        
                    except Exception as e:
                        print(f"   ❌ Parallel channels failed: {e}")
                        result = BenchmarkResult(
                            function_name="compute_roi_theta_spectrogram_parallel_channels",
                            method=f"parallel_channels_{method}",
                            execution_time=0,
                            memory_usage_mb=0,
                            cpu_usage_percent=0,
                            result_shape=(0,),
                            result_stats={},
                            parameters={'n_jobs': n_jobs, 'method': method},
                            success=False,
                            error_message=str(e)
                        )
                        results.append(result)
        
        # 3. Parallel Frequencies
        if test_parallel_frequencies and sequential_result is not None:
            for batch_size in [2, 4, 8]:
                print(f"\n📊 Testing Parallel Frequencies (batch_size={batch_size})...")
                try:
                    with monitor_resources():
                        freqs_par, roi_power_par, channel_powers_par = (
                            compute_roi_theta_spectrogram_parallel_frequencies(
                                eeg_data, roi_channels, sfreq, freq_range, 
                                n_freqs, n_cycles_factor, n_jobs=None, batch_size=batch_size
                            )
                        )
                    
                    stats = _last_benchmark_stats
                    
                    # Validate results
                    is_valid = validate_results_equal(
                        sequential_result, roi_power_par, 
                        tolerance=1e-10, name=f"parallel_frequencies_{batch_size}"
                    )
                    
                    result = BenchmarkResult(
                        function_name="compute_roi_theta_spectrogram_parallel_frequencies",
                        method="parallel_frequencies",
                        execution_time=stats['execution_time'],
                        memory_usage_mb=stats['memory_usage_mb'],
                        cpu_usage_percent=stats['cpu_usage_percent'],
                        result_shape=roi_power_par.shape,
                        result_stats=compute_result_stats(roi_power_par),
                        parameters={'roi_channels': len(roi_channels), 'n_freqs': n_freqs, 
                                  'batch_size': batch_size},
                        success=is_valid
                    )
                    results.append(result)
                    
                    speedup = stats['execution_time'] / results[0].execution_time if results[0].execution_time > 0 else 0
                    print(f"   ⏱️ Time: {stats['execution_time']:.2f}s (speedup: {1/speedup:.2f}x)")
                    print(f"   💾 Memory: {stats['memory_usage_mb']:.1f} MB")
                    print(f"   ✅ Validation: {'PASS' if is_valid else 'FAIL'}")
                    
                except Exception as e:
                    print(f"   ❌ Parallel frequencies failed: {e}")
                    result = BenchmarkResult(
                        function_name="compute_roi_theta_spectrogram_parallel_frequencies",
                        method="parallel_frequencies",
                        execution_time=0,
                        memory_usage_mb=0,
                        cpu_usage_percent=0,
                        result_shape=(0,),
                        result_stats={},
                        parameters={'batch_size': batch_size},
                        success=False,
                        error_message=str(e)
                    )
                    results.append(result)
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult], 
                       save_path: str = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Parameters:
        -----------
        results : List[BenchmarkResult]
            Benchmark results
        save_path : str
            Path to save the report
        
        Returns:
        --------
        report : str
            Formatted report string
        """
        report_lines = []
        report_lines.append("🏁 PARALLELIZATION BENCHMARK REPORT")
        report_lines.append("=" * 60)
        
        # Summary table
        report_lines.append("\n📊 PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Method':<25} {'Time (s)':<10} {'Speedup':<10} {'Memory (MB)':<12} {'Valid':<6}")
        report_lines.append("-" * 40)
        
        baseline_time = None
        for result in results:
            if result.method == "sequential" and result.success:
                baseline_time = result.execution_time
                break
        
        for result in results:
            speedup = baseline_time / result.execution_time if baseline_time and result.execution_time > 0 else 0
            valid_str = "✅" if result.success else "❌"
            
            report_lines.append(
                f"{result.method:<25} "
                f"{result.execution_time:<10.2f} "
                f"{speedup:<10.2f} "
                f"{result.memory_usage_mb:<12.1f} "
                f"{valid_str:<6}"
            )
        
        # Best performers
        report_lines.append("\n🏆 BEST PERFORMERS")
        report_lines.append("-" * 30)
        
        valid_results = [r for r in results if r.success]
        if valid_results:
            fastest = min(valid_results, key=lambda x: x.execution_time)
            most_efficient = min(valid_results, key=lambda x: x.memory_usage_mb)
            
            report_lines.append(f"Fastest: {fastest.method} ({fastest.execution_time:.2f}s)")
            report_lines.append(f"Most Memory Efficient: {most_efficient.method} ({most_efficient.memory_usage_mb:.1f} MB)")
        
        # Recommendations
        report_lines.append("\n💡 RECOMMENDATIONS")
        report_lines.append("-" * 25)
        
        if baseline_time:
            good_speedups = [r for r in valid_results if baseline_time / r.execution_time > 1.2]
            if good_speedups:
                best_speedup = max(good_speedups, key=lambda x: baseline_time / x.execution_time)
                speedup_value = baseline_time / best_speedup.execution_time
                report_lines.append(f"✅ Best parallelization: {best_speedup.method} ({speedup_value:.2f}x speedup)")
            else:
                report_lines.append("⚠️  No significant speedup achieved. Consider:")
                report_lines.append("   - Larger datasets")
                report_lines.append("   - Different parallelization strategies")
                report_lines.append("   - Checking CPU utilization")
        
        # Detailed results
        report_lines.append("\n📋 DETAILED RESULTS")
        report_lines.append("-" * 30)
        
        for result in results:
            report_lines.append(f"\n{result.method}:")
            report_lines.append(f"  Function: {result.function_name}")
            report_lines.append(f"  Time: {result.execution_time:.2f}s")
            report_lines.append(f"  Memory: {result.memory_usage_mb:.1f} MB")
            report_lines.append(f"  CPU: {result.cpu_usage_percent:.1f}%")
            report_lines.append(f"  Result shape: {result.result_shape}")
            report_lines.append(f"  Success: {result.success}")
            if result.error_message:
                report_lines.append(f"  Error: {result.error_message}")
            
            if result.result_stats:
                stats = result.result_stats
                report_lines.append(f"  Result stats: min={stats['min']:.2e}, max={stats['max']:.2e}, "
                                  f"mean={stats['mean']:.2e}, std={stats['std']:.2e}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {save_path}")
        
        return report


def run_comprehensive_benchmark(data_path: str = None, 
                              output_dir: str = "results/parallelism"):
    """
    Run a comprehensive benchmark of all parallelization methods.
    
    Parameters:
    -----------
    data_path : str
        Path to EEG data file
    output_dir : str
        Directory to save results
    """
    print("🚀 Starting Comprehensive Parallelization Benchmark")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ROIBenchmark(data_path)
    
    # If no data provided, generate synthetic data
    if benchmark.all_data is None:
        print("📝 No real data provided, generating synthetic data...")
        test_session = benchmark.generate_synthetic_data(n_channels=32, n_samples=100000)
    else:
        # Use first session from real data
        test_session = benchmark.all_data[0]
    
    # Test different ROI sizes
    roi_configurations = [
        ([0, 1, 2, 3], "small_roi_4ch"),
        ([0, 1, 2, 3, 4, 5, 6, 7], "medium_roi_8ch"),
        (list(range(16)), "large_roi_16ch")
    ]
    
    all_results = []
    
    for roi_channels, roi_name in roi_configurations:
        print(f"\n{'='*60}")
        print(f"Testing {roi_name}: {len(roi_channels)} channels")
        print(f"{'='*60}")
        
        results = benchmark.benchmark_roi_spectrogram(
            test_session, 
            roi_channels,
            freq_range=(3, 8),
            n_freqs=20,
            n_cycles_factor=3.0,
            test_parallel_channels=True,
            test_parallel_frequencies=True,
            n_jobs_list=[1, 2, 4, None]
        )
        
        # Save individual results
        result_file = os.path.join(output_dir, f"benchmark_{roi_name}.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        all_results.extend(results)
        
        # Generate individual report
        report = benchmark.generate_report(
            results, 
            save_path=os.path.join(output_dir, f"report_{roi_name}.txt")
        )
        
        print("\n" + report)
    
    # Generate combined report
    combined_report = benchmark.generate_report(
        all_results,
        save_path=os.path.join(output_dir, "combined_benchmark_report.txt")
    )
    
    print(f"\n🎉 Benchmark complete! Results saved to {output_dir}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parallelization benchmarks')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to EEG data pickle file')
    parser.add_argument('--output_dir', type=str, default='results/parallelism',
                       help='Output directory for results')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of real data')
    
    args = parser.parse_args()
    
    if args.synthetic:
        args.data_path = None
    
    results = run_comprehensive_benchmark(args.data_path, args.output_dir)
    
    print(f"\nCompleted {len(results)} benchmark tests")