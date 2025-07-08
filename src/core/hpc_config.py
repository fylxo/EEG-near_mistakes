#!/usr/bin/env python3
"""
HPC Configuration for Cross-Rats NM Theta Analysis

This script provides different configuration presets for running the analysis
on HPC systems with different memory and time constraints.
"""

import os
from typing import Dict, List, Optional

# Base configuration - modify these paths for your HPC system
HPC_BASE_CONFIG = {
    'data_path': '/scratch/your_username/data/all_eeg_data.pkl',
    'save_path': '/mnt/d/nm_theta_results',  # Use D: drive (471 GB available)
    'python_path': '/path/to/your/eeg-near_mistakes/src',
    'email': 'your_email@domain.com'
}

# Analysis presets
ANALYSIS_PRESETS = {
    'quick_test': {
        'roi': '1,2,3',
        'freq_min': 3.0,
        'freq_max': 8.0,
        'n_freqs': 30,
        'window_duration': 1.0,
        'method': 'mne',
        'rat_ids': ['9151', '9152'],  # Just 2 rats for testing
        'memory_gb': 64,  # Increased for safety
        'time_hours': 4,
        'cleanup_intermediate_files': True
    },
    
    'full_analysis': {
        'roi': '1,2,3',
        'freq_min': 1.0,
        'freq_max': 12.0,
        'n_freqs': 256,
        'window_duration': 2.0,
        'method': 'mne',
        'rat_ids': None,  # All rats
        'memory_gb': 256,  # Increased for memory-intensive sessions
        'time_hours': 48,  # Longer time for large analyses
        'cleanup_intermediate_files': True
    },
    
    'high_freq_analysis': {
        'roi': '1,2,3',
        'freq_min': 1.0,
        'freq_max': 40.0,
        'n_freqs': 240,
        'window_duration': 2.0,
        'method': 'mne',
        'rat_ids': None,  # All rats
        'memory_gb': 384,  # High memory for problematic sessions (1.7GB+ per array)
        'time_hours': 72,  # Extended time for high-frequency analysis
        'cleanup_intermediate_files': True
    },
    
    'single_channel': {
        'roi': '1',
        'freq_min': 1.0,
        'freq_max': 12.0,
        'n_freqs': 256,
        'window_duration': 2.0,
        'method': 'mne',
        'rat_ids': None,
        'memory_gb': 128,  # Increased for safety
        'time_hours': 12,
        'cleanup_intermediate_files': True
    },
    
    'frontal_roi': {
        'roi': 'frontal',
        'freq_min': 1.0,
        'freq_max': 12.0,
        'n_freqs': 256,
        'window_duration': 2.0,
        'method': 'mne',
        'rat_ids': None,
        'memory_gb': 192,  # Increased for ROI analysis
        'time_hours': 24,
        'cleanup_intermediate_files': True
    }
}

def generate_slurm_script(preset_name: str, output_file: str = None) -> str:
    """
    Generate a SLURM batch script for the specified analysis preset.
    
    Parameters:
    -----------
    preset_name : str
        Name of the analysis preset to use
    output_file : str, optional
        Path to save the script file
        
    Returns:
    --------
    str
        The generated SLURM script content
    """
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(ANALYSIS_PRESETS.keys())}")
    
    preset = ANALYSIS_PRESETS[preset_name]
    
    # Build command arguments
    args = [
        f"--roi \"{preset['roi']}\"",
        f"--freq_min {preset['freq_min']}",
        f"--freq_max {preset['freq_max']}",
        f"--n_freqs {preset['n_freqs']}",
        f"--window_duration {preset['window_duration']}",
        f"--method {preset['method']}",
        f"--save_path {HPC_BASE_CONFIG['save_path']}_{preset_name}",
        f"--pkl_path {HPC_BASE_CONFIG['data_path']}",
        "--verbose"
    ]
    
    if preset['rat_ids']:
        args.append(f"--rat_ids \"{','.join(preset['rat_ids'])}\"")
    
    command = " \\\n    ".join(args)
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=nm_theta_{preset_name}
#SBATCH --partition=cpu  # Adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem={preset['memory_gb']}G
#SBATCH --time={preset['time_hours']:02d}:00:00
#SBATCH --output=nm_theta_{preset_name}_%j.out
#SBATCH --error=nm_theta_{preset_name}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={HPC_BASE_CONFIG['email']}

# Load required modules (adjust for your cluster)
module purge
module load python/3.9

# Set environment variables for memory efficiency
export PYTHONPATH="${{PYTHONPATH}}:{HPC_BASE_CONFIG['python_path']}"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1  # Force unbuffered output for real-time logging
export MKL_NUM_THREADS=16  # Control Intel MKL threading

# Memory monitoring setup
MEMORY_LOG="{HPC_BASE_CONFIG['save_path']}_{preset_name}/memory_usage.log"
CHECKPOINT_FILE="{HPC_BASE_CONFIG['save_path']}_{preset_name}/checkpoint.pkl"

# Function to monitor memory usage
monitor_memory() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Memory monitoring started" >> $MEMORY_LOG
    while true; do
        if [ -f "$CHECKPOINT_FILE" ]; then
            PEAK_MEM=$(ps -o pid,ppid,pmem,pcpu,comm -p $$ | tail -1)
            FREE_MEM=$(free -h | grep '^Mem:' | awk '{{print $7}}')
            echo "$(date '+%Y-%m-%d %H:%M:%S'): Peak: $PEAK_MEM, Free: $FREE_MEM" >> $MEMORY_LOG
        fi
        sleep 300  # Log every 5 minutes
    done &
    MONITOR_PID=$!
}}

# Cleanup function
cleanup() {{
    echo "Analysis interrupted or completed at $(date)"
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
    fi
    # Log final memory state
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Analysis ended" >> $MEMORY_LOG
    free -h >> $MEMORY_LOG
}}

# Set up signal handling
trap cleanup EXIT INT TERM

# Create output directory
mkdir -p {HPC_BASE_CONFIG['save_path']}_{preset_name}

# Change to working directory
cd {HPC_BASE_CONFIG['python_path']}/core

# System information logging
echo "============================================"
echo "Starting {preset_name} analysis at $(date)"
echo "Node: $(hostname)"
echo "Memory allocated: {preset['memory_gb']}GB"
echo "Time limit: {preset['time_hours']} hours"
echo "============================================"

# Log system resources
echo "System Memory Info:"
free -h
echo ""
echo "CPU Info:"
lscpu | grep -E "(Model name|CPU\\(s\\)|Thread)"
echo ""

# Start memory monitoring
monitor_memory

# Create checkpoint marker
touch "$CHECKPOINT_FILE"

# Run the analysis with memory error handling
echo "Starting analysis execution..."
python nm_theta_cross_rats.py \\
    {command} 2>&1 | tee {HPC_BASE_CONFIG['save_path']}_{preset_name}/analysis.log

# Check exit status
EXIT_STATUS=${{PIPESTATUS[0]}}

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Analysis completed successfully at $(date)"
    echo "Results saved to: {HPC_BASE_CONFIG['save_path']}_{preset_name}"
    
    # Final memory usage summary
    echo "Final memory usage:"
    free -h
    
    # Optional: Copy results to permanent storage
    # cp -r {HPC_BASE_CONFIG['save_path']}_{preset_name} /home/your_username/results/
    
else
    echo "❌ Analysis failed with exit code $EXIT_STATUS at $(date)"
    echo "Check logs for details:"
    echo "  - Analysis log: {HPC_BASE_CONFIG['save_path']}_{preset_name}/analysis.log"
    echo "  - Memory log: {HPC_BASE_CONFIG['save_path']}_{preset_name}/memory_usage.log"
    echo "  - SLURM output: nm_theta_{preset_name}_%j.out"
    echo "  - SLURM error: nm_theta_{preset_name}_%j.err"
fi

# Remove checkpoint marker
rm -f "$CHECKPOINT_FILE"

echo "Job finished at $(date) with exit status $EXIT_STATUS"
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(script_content)
        print(f"SLURM script saved to: {output_file}")
    
    return script_content

def run_local_analysis(preset_name: str, **kwargs):
    """
    Run the analysis locally using a preset configuration.
    
    Parameters:
    -----------
    preset_name : str
        Name of the analysis preset to use
    **kwargs
        Additional arguments to override preset values
    """
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(ANALYSIS_PRESETS.keys())}")
    
    from nm_theta_cross_rats import run_cross_rats_analysis
    
    preset = ANALYSIS_PRESETS[preset_name].copy()
    preset.update(kwargs)  # Override with any provided kwargs
    
    # Remove HPC-specific keys
    preset.pop('memory_gb', None)
    preset.pop('time_hours', None)
    
    print(f"Running {preset_name} analysis locally...")
    print(f"Configuration: {preset}")
    
    return run_cross_rats_analysis(**preset)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HPC scripts for cross-rats analysis')
    parser.add_argument('--preset', required=True, choices=list(ANALYSIS_PRESETS.keys()),
                       help='Analysis preset to use')
    parser.add_argument('--output', help='Output file for SLURM script')
    parser.add_argument('--run-local', action='store_true', help='Run locally instead of generating script')
    
    args = parser.parse_args()
    
    if args.run_local:
        run_local_analysis(args.preset)
    else:
        output_file = args.output or f"run_{args.preset}.sh"
        generate_slurm_script(args.preset, output_file)