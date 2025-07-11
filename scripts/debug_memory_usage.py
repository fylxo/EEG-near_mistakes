#!/usr/bin/env python3
"""
Debug Memory Usage and Process Limits

This script helps debug why the aggregation is getting "Killed"
"""

import os
import sys
import glob
import pickle
import psutil
import resource
from typing import Dict, List

def print_system_info():
    """Print system resource information"""
    print("🖥️  SYSTEM RESOURCE INFO")
    print("=" * 50)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    
    # Process limits
    try:
        mem_limit = resource.getrlimit(resource.RLIMIT_AS)
        if mem_limit[0] == resource.RLIM_INFINITY:
            print(f"Memory limit: Unlimited")
        else:
            print(f"Memory limit: {mem_limit[0] / (1024**3):.1f} GB")
    except:
        print(f"Memory limit: Unable to determine")
    
    # Current process info
    process = psutil.Process()
    print(f"Current process memory: {process.memory_info().rss / (1024**3):.1f} GB")
    
    print()

def estimate_memory_usage(rat_dir: str) -> Dict:
    """Estimate memory usage for processing this rat"""
    print("📊 MEMORY ESTIMATION")
    print("=" * 50)
    
    session_files = glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl"))
    
    if not session_files:
        print("No session files found")
        return {}
    
    print(f"Found {len(session_files)} session files")
    
    # Sample first session to estimate size
    try:
        with open(session_files[0], 'rb') as f:
            sample_result = pickle.load(f)
        
        if 'nm_windows' not in sample_result:
            print("Sample session missing 'nm_windows'")
            return {}
        
        # Estimate sizes
        total_spectrogram_elements = 0
        nm_sizes = len(sample_result['nm_windows'])
        
        for nm_size, data in sample_result['nm_windows'].items():
            spectrogram = data['avg_spectrogram']
            elements = spectrogram.size
            total_spectrogram_elements += elements
            print(f"  NM size {nm_size}: {spectrogram.shape} = {elements:,} elements")
        
        # Estimate memory (assuming float64 = 8 bytes per element)
        bytes_per_session = total_spectrogram_elements * 8
        total_bytes = bytes_per_session * len(session_files)
        
        print(f"\nMemory estimates:")
        print(f"  Per session: {bytes_per_session / (1024**2):.1f} MB")
        print(f"  All sessions: {total_bytes / (1024**3):.1f} GB")
        print(f"  Plus overhead (~2x): {total_bytes * 2 / (1024**3):.1f} GB")
        
        return {
            'n_sessions': len(session_files),
            'nm_sizes': nm_sizes,
            'per_session_mb': bytes_per_session / (1024**2),
            'total_gb': total_bytes / (1024**3),
            'estimated_peak_gb': total_bytes * 2 / (1024**3)
        }
        
    except Exception as e:
        print(f"Error sampling session: {e}")
        return {}

def test_load_sessions(rat_dir: str, max_sessions: int = 5):
    """Test loading a few sessions to see where it fails"""
    print("🧪 SESSION LOADING TEST")
    print("=" * 50)
    
    session_files = glob.glob(os.path.join(rat_dir, "session_*", "nm_roi_theta_analysis_results.pkl"))
    
    if not session_files:
        print("No session files found")
        return
    
    test_files = session_files[:max_sessions]
    print(f"Testing first {len(test_files)} sessions...")
    
    for i, session_file in enumerate(test_files):
        session_name = os.path.basename(os.path.dirname(session_file))
        
        try:
            print(f"  Loading {session_name}...", end="")
            
            # Check memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**2)
            
            with open(session_file, 'rb') as f:
                result = pickle.load(f)
            
            # Check memory after
            mem_after = process.memory_info().rss / (1024**2)
            mem_delta = mem_after - mem_before
            
            if 'nm_windows' in result:
                nm_sizes = len(result['nm_windows'])
                print(f" ✓ ({nm_sizes} NM sizes, +{mem_delta:.1f} MB)")
            else:
                print(f" ⚠️  Missing 'nm_windows'")
            
        except Exception as e:
            print(f" ❌ {e}")
            break

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug memory usage and system limits')
    parser.add_argument('--rat_dir', required=True, help='Path to rat directory')
    
    args = parser.parse_args()
    
    print("🔍 Memory Usage Debugger")
    print("=" * 60)
    print(f"Rat directory: {args.rat_dir}")
    print("=" * 60)
    print()
    
    # System info
    print_system_info()
    
    # Memory estimation
    estimates = estimate_memory_usage(args.rat_dir)
    print()
    
    # Test loading
    test_load_sessions(args.rat_dir)
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("=" * 50)
    
    if estimates:
        peak_gb = estimates.get('estimated_peak_gb', 0)
        if peak_gb > 50:
            print(f"⚠️  Estimated peak memory usage: {peak_gb:.1f} GB")
            print("   Consider using memory-efficient processing")
        elif peak_gb > 0:
            print(f"✓ Estimated peak memory usage: {peak_gb:.1f} GB (should be fine)")
    
    print("\nPossible causes of 'Killed':")
    print("1. Memory limit (ulimit -v)")
    print("2. Swap space exhausted") 
    print("3. SLURM job memory limit")
    print("4. Node out of memory")
    print("5. File system issues")
    
    print("\nTo investigate:")
    print("- Check: ulimit -a")
    print("- Check: free -h") 
    print("- Check: df -h")
    print("- Run with: python -u script.py (unbuffered output)")

if __name__ == "__main__":
    main()