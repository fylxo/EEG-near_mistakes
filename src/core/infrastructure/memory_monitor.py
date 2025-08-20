#!/usr/bin/env python3
"""
Memory monitoring utilities for EEG analysis.

This module provides functions to monitor memory usage during analysis
and identify memory-intensive operations.
"""

import psutil
import os
import gc
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

class MemoryMonitor:
    """
    Context manager and utility class for monitoring memory usage
    during EEG analysis operations.
    """
    
    def __init__(self, log_file: str = None, log_interval: float = 1.0):
        """
        Initialize the memory monitor.
        
        Parameters:
        -----------
        log_file : str, optional
            Path to file for logging memory usage
        log_interval : float
            Interval in seconds between memory measurements (default: 1.0)
        """
        self.log_file = log_file
        self.log_interval = log_interval
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.peak_memory = 0
        self.start_memory = 0
        
    def get_memory_info(self) -> Dict:
        """
        Get current memory information.
        
        Returns:
        --------
        Dict
            Dictionary with memory information
        """
        try:
            # Process memory info
            mem_info = self.process.memory_info()
            
            # System memory info
            system_mem = psutil.virtual_memory()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'process_memory_mb': round(mem_info.rss / 1024 / 1024, 2),
                'process_memory_peak_mb': round(self.peak_memory / 1024 / 1024, 2),
                'system_memory_total_gb': round(system_mem.total / 1024 / 1024 / 1024, 2),
                'system_memory_available_gb': round(system_mem.available / 1024 / 1024 / 1024, 2),
                'system_memory_percent': system_mem.percent,
                'system_memory_free_gb': round(system_mem.free / 1024 / 1024 / 1024, 2)
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                mem_info = self.process.memory_info()
                current_memory = mem_info.rss
                
                # Update peak memory
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Store in history
                self.memory_history.append({
                    'time': time.time(),
                    'memory_mb': round(current_memory / 1024 / 1024, 2)
                })
                
                # Log to file if specified
                if self.log_file:
                    self._log_to_file()
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                break
    
    def _log_to_file(self):
        """Log current memory state to file."""
        try:
            mem_info = self.get_memory_info()
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(mem_info) + '\n')
        except Exception as e:
            print(f"Memory logging error: {e}")
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.start_memory = self.process.memory_info().rss
            self.peak_memory = self.start_memory
            self.memory_history = []
            
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """
        Stop monitoring and return summary.
        
        Returns:
        --------
        Dict
            Memory usage summary
        """
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
        
        final_memory = self.process.memory_info().rss
        
        return {
            'start_memory_mb': round(self.start_memory / 1024 / 1024, 2),
            'final_memory_mb': round(final_memory / 1024 / 1024, 2),
            'peak_memory_mb': round(self.peak_memory / 1024 / 1024, 2),
            'memory_increase_mb': round((final_memory - self.start_memory) / 1024 / 1024, 2),
            'peak_increase_mb': round((self.peak_memory - self.start_memory) / 1024 / 1024, 2),
            'duration_seconds': len(self.memory_history) * self.log_interval if self.memory_history else 0
        }
    
    def __enter__(self):
        """Enter context manager."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.stop_monitoring()


@contextmanager
def monitor_memory_usage(operation_name: str, log_file: str = None, verbose: bool = True):
    """
    Context manager for monitoring memory usage of a specific operation.
    
    Parameters:
    -----------
    operation_name : str
        Name of the operation being monitored
    log_file : str, optional
        Path to log file for detailed logging
    verbose : bool
        Whether to print memory usage summary
        
    Usage:
    ------
    with monitor_memory_usage("spectrogram_computation", verbose=True) as monitor:
        # Your memory-intensive operation here
        result = compute_spectrogram(data)
    """
    monitor = MemoryMonitor(log_file=log_file)
    
    if verbose:
        print(f"Starting memory monitoring for: {operation_name}")
    
    try:
        monitor.start_monitoring()
        yield monitor
        
    finally:
        summary = monitor.stop_monitoring()
        
        if verbose:
            print(f"Memory summary for {operation_name}:")
            print(f"  Start: {summary['start_memory_mb']:.1f} MB")
            print(f"  Peak: {summary['peak_memory_mb']:.1f} MB")
            print(f"  Final: {summary['final_memory_mb']:.1f} MB")
            print(f"  Peak increase: {summary['peak_increase_mb']:.1f} MB")
            
            # Warning for high memory usage
            if summary['peak_memory_mb'] > 8000:  # > 8 GB
                print(f"  ⚠️  High memory usage detected!")
            elif summary['peak_memory_mb'] > 4000:  # > 4 GB
                print(f"  📈 Moderate memory usage")


def force_garbage_collection(verbose: bool = False) -> Dict:
    """
    Force garbage collection and return memory statistics.
    
    Parameters:
    -----------
    verbose : bool
        Whether to print garbage collection stats
        
    Returns:
    --------
    Dict
        Memory statistics before and after garbage collection
    """
    process = psutil.Process(os.getpid())
    
    # Get memory before GC
    mem_before = process.memory_info().rss
    
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory after GC
    mem_after = process.memory_info().rss
    
    stats = {
        'memory_before_mb': round(mem_before / 1024 / 1024, 2),
        'memory_after_mb': round(mem_after / 1024 / 1024, 2),
        'memory_freed_mb': round((mem_before - mem_after) / 1024 / 1024, 2),
        'objects_collected': collected
    }
    
    if verbose:
        print(f"🧹 Garbage collection stats:")
        print(f"  Before: {stats['memory_before_mb']:.1f} MB")
        print(f"  After: {stats['memory_after_mb']:.1f} MB")
        print(f"  Freed: {stats['memory_freed_mb']:.1f} MB")
        print(f"  Objects collected: {stats['objects_collected']}")
    
    return stats


def check_memory_requirements(n_freqs: int, n_time_points: int, n_channels: int = 1, 
                             safety_factor: float = 3.0) -> Dict:
    """
    Estimate memory requirements for spectrogram computation.
    
    Parameters:
    -----------
    n_freqs : int
        Number of frequencies
    n_time_points : int
        Number of time points
    n_channels : int
        Number of channels (default: 1)
    safety_factor : float
        Safety factor for memory estimation (default: 3.0)
        
    Returns:
    --------
    Dict
        Memory requirement estimates
    """
    # Estimate array sizes
    complex_array_bytes = n_freqs * n_time_points * 16  # complex128 = 16 bytes
    float_array_bytes = n_freqs * n_time_points * 8    # float64 = 8 bytes
    
    # Total per channel (intermediate arrays, final arrays, etc.)
    per_channel_bytes = complex_array_bytes * safety_factor
    total_bytes = per_channel_bytes * n_channels
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    system_memory = psutil.virtual_memory()
    
    estimates = {
        'estimated_requirement_gb': round(total_bytes / 1024 / 1024 / 1024, 2),
        'estimated_requirement_mb': round(total_bytes / 1024 / 1024, 2),
        'per_channel_mb': round(per_channel_bytes / 1024 / 1024, 2),
        'current_memory_mb': round(current_memory / 1024 / 1024, 2),
        'available_memory_gb': round(system_memory.available / 1024 / 1024 / 1024, 2),
        'sufficient_memory': total_bytes < system_memory.available * 0.8,  # Use 80% of available
        'array_shape': (n_freqs, n_time_points),
        'n_channels': n_channels,
        'safety_factor': safety_factor
    }
    
    return estimates


def log_memory_warning(operation: str, estimates: Dict, verbose: bool = True):
    """
    Log memory warning if requirements are high.
    
    Parameters:
    -----------
    operation : str
        Name of the operation
    estimates : Dict
        Memory estimates from check_memory_requirements
    verbose : bool
        Whether to print warnings
    """
    if not estimates['sufficient_memory']:
        warning_msg = (
            f"⚠️  Memory warning for {operation}:\n"
            f"   Estimated requirement: {estimates['estimated_requirement_gb']:.1f} GB\n"
            f"   Available memory: {estimates['available_memory_gb']:.1f} GB\n"
            f"   Array shape: {estimates['array_shape']}\n"
            f"   Consider using HPC with more memory!"
        )
        
        if verbose:
            print(warning_msg)
        
        return warning_msg
    
    elif estimates['estimated_requirement_gb'] > 2.0:
        info_msg = (
            f"Processing Memory info for {operation}:\n"
            f"   Estimated requirement: {estimates['estimated_requirement_gb']:.1f} GB\n"
            f"   Available memory: {estimates['available_memory_gb']:.1f} GB\n"
            f"   Status: Should be sufficient"
        )
        
        if verbose:
            print(info_msg)
        
        return info_msg
    
    return None


if __name__ == "__main__":
    # Test the memory monitoring functionality
    import numpy as np
    
    print("Testing memory monitoring...")
    
    with monitor_memory_usage("test_operation") as monitor:
        # Simulate memory-intensive operation
        large_array = np.random.random((1000, 1000, 100))
        time.sleep(2)
        del large_array
        
        # Force garbage collection
        force_garbage_collection(verbose=True)
    
    # Test memory requirements estimation
    print("\nTesting memory requirements estimation...")
    estimates = check_memory_requirements(240, 474506, n_channels=3)
    log_memory_warning("test_spectrogram", estimates)