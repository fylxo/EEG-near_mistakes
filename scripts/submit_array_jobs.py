#!/usr/bin/env python3
"""
SLURM Array Job Submission Script

This script automates the submission of SLURM array jobs for cross-rats analysis.
It discovers available rats, configures the array job, and submits it to SLURM.
"""

import os
import sys
import json
import subprocess
import argparse
from typing import Dict, List

def update_slurm_script(script_path: str, config: Dict, cluster_config: Dict) -> str:
    """
    Update the SLURM script with the correct array range and paths.
    
    Parameters:
    -----------
    script_path : str
        Path to the SLURM script template
    config : Dict
        Rat configuration from discover_rats.py
    cluster_config : Dict
        Cluster-specific configuration
        
    Returns:
    --------
    str
        Path to the updated SLURM script
    """
    # Read the template
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Update array range
    array_line = f"#SBATCH --array={config['array_range']}"
    
    # Find the line with array placeholder and replace it
    lines = script_content.split('\n')
    updated_lines = []
    
    for line in lines:
        if line.strip().startswith('# #SBATCH --array='):
            updated_lines.append(array_line)
        elif line.startswith('DATA_PATH='):
            updated_lines.append(f'DATA_PATH="{cluster_config["data_path"]}"')
        elif line.startswith('FREQ_FILE_PATH='):
            updated_lines.append(f'FREQ_FILE_PATH="{cluster_config["freq_file_path"]}"')
        elif line.startswith('SAVE_PATH='):
            updated_lines.append(f'SAVE_PATH="{cluster_config["save_path"]}"')
        elif line.startswith('PROJECT_DIR='):
            updated_lines.append(f'PROJECT_DIR="{cluster_config["project_dir"]}"')
        elif line.startswith('#SBATCH --mail-user='):
            updated_lines.append(f'#SBATCH --mail-user={cluster_config["email"]}')
        elif line.startswith('#SBATCH --mem='):
            updated_lines.append(f'#SBATCH --mem={cluster_config["memory_per_job"]}')
        elif line.startswith('#SBATCH --time='):
            updated_lines.append(f'#SBATCH --time={cluster_config["time_limit"]}')
        elif line.startswith('#SBATCH --partition='):
            updated_lines.append(f'#SBATCH --partition={cluster_config["partition"]}')
        elif line.startswith('ROI='):
            updated_lines.append(f'ROI="{cluster_config["roi"]}"')
        elif line.startswith('FREQ_MIN='):
            updated_lines.append(f'FREQ_MIN={cluster_config["freq_min"]}')
        elif line.startswith('FREQ_MAX='):
            updated_lines.append(f'FREQ_MAX={cluster_config["freq_max"]}')
        elif line.startswith('WINDOW_DURATION='):
            updated_lines.append(f'WINDOW_DURATION={cluster_config["window_duration"]}')
        elif line.startswith('N_CYCLES_FACTOR='):
            updated_lines.append(f'N_CYCLES_FACTOR={cluster_config["n_cycles_factor"]}')
        else:
            updated_lines.append(line)
    
    # Write updated script
    updated_script_path = script_path.replace('.sh', '_configured.sh')
    with open(updated_script_path, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    # Make executable
    os.chmod(updated_script_path, 0o755)
    
    return updated_script_path

def submit_job(script_path: str, dry_run: bool = False) -> str:
    """
    Submit the SLURM job.
    
    Parameters:
    -----------
    script_path : str
        Path to the SLURM script
    dry_run : bool
        If True, only print the command without submitting
        
    Returns:
    --------
    str
        Job ID or command that would be executed
    """
    cmd = ['sbatch', script_path]
    
    if dry_run:
        print(f"Would execute: {' '.join(cmd)}")
        return ' '.join(cmd)
    else:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return None

def create_monitoring_script(config: Dict, cluster_config: Dict) -> str:
    """
    Create a script to monitor job progress.
    
    Parameters:
    -----------
    config : Dict
        Rat configuration
    cluster_config : Dict
        Cluster configuration
        
    Returns:
    --------
    str
        Path to the monitoring script
    """
    monitoring_script = f"""#!/bin/bash
# Job Monitoring Script for NM Theta Array Jobs

echo "========================================"
echo "NM Theta Array Job Monitoring"
echo "========================================"
echo "Total rats: {config['n_rats']}"
echo "Array range: {config['array_range']}"
echo "Results path: {cluster_config['save_path']}"
echo "========================================"

# Check SLURM queue status
echo "Current job status:"
squeue -u $USER --name=nm_theta_array --format="%.8i %.8j %.8T %.10M %.6D %R"

echo ""
echo "Job states summary:"
squeue -u $USER --name=nm_theta_array --format="%.8T" --noheader | sort | uniq -c

echo ""
echo "Completed rats (with COMPLETED marker):"
find {cluster_config['save_path']} -name "COMPLETED" -type f | wc -l

echo ""
echo "Failed rats (with FAILED marker):"
find {cluster_config['save_path']} -name "FAILED" -type f | wc -l

echo ""
echo "Results directories:"
ls -la {cluster_config['save_path']}/ | grep "^d" | wc -l

echo ""
echo "Recent log files:"
ls -lt logs/analysis_rat_*.log | head -5

echo ""
echo "========================================"
echo "To check individual rat status:"
echo "  ls {cluster_config['save_path']}/rat_*/COMPLETED"
echo "  ls {cluster_config['save_path']}/rat_*/FAILED"
echo ""
echo "To view logs:"
echo "  tail -f logs/analysis_rat_<RAT_ID>.log"
echo "  tail -f logs/nm_theta_rat_<JOB_ID>_<ARRAY_ID>.out"
echo "========================================"
"""
    
    monitor_script_path = 'monitor_jobs.sh'
    with open(monitor_script_path, 'w') as f:
        f.write(monitoring_script)
    
    os.chmod(monitor_script_path, 0o755)
    return monitor_script_path

def main():
    parser = argparse.ArgumentParser(description='Submit SLURM array jobs for cross-rats analysis')
    
    # Required arguments
    parser.add_argument('--data_path', required=True, help='Path to the main EEG data file')
    parser.add_argument('--project_dir', required=True, help='Path to the project directory')
    parser.add_argument('--save_path', required=True, help='Path to save results')
    
    # Optional configuration
    parser.add_argument('--freq_file_path', help='Path to frequencies file (default: data/config/frequencies_128.txt)')
    parser.add_argument('--email', default='your_email@domain.com', help='Email for job notifications')
    parser.add_argument('--partition', default='cpu', help='SLURM partition')
    parser.add_argument('--memory_per_job', default='128G', help='Memory per job')
    parser.add_argument('--time_limit', default='24:00:00', help='Time limit per job')
    
    # Analysis parameters
    parser.add_argument('--roi', default='1,2,3', help='ROI specification')
    parser.add_argument('--freq_min', type=float, default=1.0, help='Minimum frequency')
    parser.add_argument('--freq_max', type=float, default=45.0, help='Maximum frequency')
    parser.add_argument('--window_duration', type=float, default=2.0, help='Window duration')
    parser.add_argument('--n_cycles_factor', type=float, default=3.0, help='N cycles factor')
    
    # Options
    parser.add_argument('--exclude_9442', action='store_true', help='Exclude rat 9442')
    parser.add_argument('--dry_run', action='store_true', help='Show commands without executing')
    parser.add_argument('--force_rediscover', action='store_true', help='Force rediscovery of rats')
    
    args = parser.parse_args()
    
    # Set default freq_file_path if not provided
    if args.freq_file_path is None:
        args.freq_file_path = os.path.join(args.project_dir, 'data', 'config', 'frequencies.txt')
    
    print("========================================")
    print("SLURM Array Job Submission")
    print("========================================")
    print(f"Data path: {args.data_path}")
    print(f"Project dir: {args.project_dir}")
    print(f"Save path: {args.save_path}")
    print(f"Frequency file: {args.freq_file_path}")
    print(f"Analysis ROI: {args.roi}")
    print(f"Frequency range: {args.freq_min}-{args.freq_max} Hz")
    print("========================================")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    
    # Step 1: Discover rats
    rat_config_file = 'rat_config.json'
    if args.force_rediscover or not os.path.exists(rat_config_file):
        print("Discovering available rats...")
        discover_cmd = [
            'python', 'scripts/discover_rats.py',
            '--pkl_path', args.data_path,
            '--output', rat_config_file
        ]
        
        if args.exclude_9442:
            discover_cmd.append('--exclude_9442')
        
        if args.dry_run:
            print(f"Would execute: {' '.join(discover_cmd)}")
        else:
            subprocess.run(discover_cmd, check=True)
    
    # Load rat configuration
    with open(rat_config_file, 'r') as f:
        rat_config = json.load(f)
    
    print(f"Found {rat_config['n_rats']} rats to process")
    print(f"Array range: {rat_config['array_range']}")
    
    # Step 2: Configure cluster settings
    cluster_config = {
        'data_path': args.data_path,
        'freq_file_path': args.freq_file_path,
        'save_path': args.save_path,
        'project_dir': args.project_dir,
        'email': args.email,
        'partition': args.partition,
        'memory_per_job': args.memory_per_job,
        'time_limit': args.time_limit,
        'roi': args.roi,
        'freq_min': args.freq_min,
        'freq_max': args.freq_max,
        'window_duration': args.window_duration,
        'n_cycles_factor': args.n_cycles_factor
    }
    
    # Step 3: Update SLURM script
    print("Configuring SLURM script...")
    original_script = 'scripts/slurm_array_job.sh'
    configured_script = update_slurm_script(original_script, rat_config, cluster_config)
    print(f"Updated script: {configured_script}")
    
    # Step 4: Create monitoring script
    monitor_script = create_monitoring_script(rat_config, cluster_config)
    print(f"Created monitoring script: {monitor_script}")
    
    # Step 5: Submit job
    print("Submitting array job...")
    job_id = submit_job(configured_script, args.dry_run)
    
    if job_id and not args.dry_run:
        print(f"✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        print(f"Array jobs: {rat_config['n_rats']} rats")
        print(f"Results will be saved to: {args.save_path}")
        print("")
        print("Monitoring commands:")
        print(f"  squeue -u $USER --name=nm_theta_array")
        print(f"  ./{monitor_script}")
        print("")
        print("To cancel all jobs:")
        print(f"  scancel {job_id}")
        
    elif args.dry_run:
        print("✅ Dry run completed - no jobs submitted")
        print("Review the configuration above and run without --dry_run to submit")
    else:
        print("❌ Job submission failed")
        sys.exit(1)

if __name__ == "__main__":
    main()