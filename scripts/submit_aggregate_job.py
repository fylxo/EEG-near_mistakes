#!/usr/bin/env python3
"""
Submit Session Aggregation SLURM Job

This script submits a SLURM job to aggregate partial session results
with plenty of memory allocated.

Usage:
    python scripts/submit_aggregate_job.py --results_dir /path/to/results
"""

import os
import sys
import subprocess
import argparse

def update_slurm_script(script_path: str, project_dir: str, memory: str = "256G", time_limit: str = "4:00:00"):
    """Update the SLURM script with correct paths and resources."""
    
    # Read the template
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update the project directory path
    content = content.replace(
        'cd /path/to/your/eeg-near_mistakes  # UPDATE THIS PATH',
        f'cd {project_dir}'
    )
    
    # Update memory if requested
    if memory != "256G":
        content = content.replace('#SBATCH --mem=256G', f'#SBATCH --mem={memory}')
    
    # Update time limit if requested
    if time_limit != "4:00:00":
        content = content.replace('#SBATCH --time=4:00:00', f'#SBATCH --time={time_limit}')
    
    # Write updated script
    updated_script_path = script_path.replace('.sh', '_configured.sh')
    with open(updated_script_path, 'w') as f:
        f.write(content)
    
    return updated_script_path

def submit_job(script_path: str, results_dir: str, verbose: bool = True) -> str:
    """Submit the SLURM job."""
    
    cmd = [
        'sbatch',
        f'--export=RESULTS_DIR={results_dir}',
        script_path
    ]
    
    if verbose:
        print(f"Submitting job with command:")
        print(f"  {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract job ID from output (format: "Submitted batch job 12345")
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            return job_id
        else:
            return output
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Job submission failed:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Submit SLURM job for session aggregation')
    parser.add_argument('--results_dir', required=True, help='Path to results directory')
    parser.add_argument('--memory', default='256G', help='Memory to request (default: 256G)')
    parser.add_argument('--time_limit', default='4:00:00', help='Time limit (default: 4:00:00)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🚀 SLURM Session Aggregation Job Submission")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Memory request: {args.memory}")
    print(f"Time limit: {args.time_limit}")
    print("=" * 60)
    
    # Get current project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_dir = os.path.dirname(__file__)
    
    print(f"Project directory: {project_dir}")
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Update SLURM script
    original_script = os.path.join(script_dir, 'slurm_aggregate_sessions.sh')
    
    if not os.path.exists(original_script):
        print(f"❌ SLURM script not found: {original_script}")
        sys.exit(1)
    
    try:
        configured_script = update_slurm_script(
            original_script, 
            project_dir, 
            args.memory, 
            args.time_limit
        )
        
        print(f"✓ Updated SLURM script: {configured_script}")
        
    except Exception as e:
        print(f"❌ Failed to update SLURM script: {e}")
        sys.exit(1)
    
    # Ensure logs directory exists
    logs_dir = os.path.join(project_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Submit job
    print(f"\n🚀 Submitting SLURM job...")
    job_id = submit_job(configured_script, args.results_dir, verbose=args.verbose)
    
    if job_id:
        print(f"✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")
        
        print(f"\n📊 Monitor job progress:")
        print(f"  squeue -j {job_id}")
        print(f"  tail -f logs/aggregate_sessions_{job_id}.out")
        
        print(f"\n📁 Log files:")
        print(f"  Output: logs/aggregate_sessions_{job_id}.out")
        print(f"  Error:  logs/aggregate_sessions_{job_id}.err")
        
        print(f"\n⏰ Estimated completion:")
        print(f"  The job should complete within {args.time_limit}")
        print(f"  It will process session aggregation AND cross-rat analysis")
        
    else:
        print(f"❌ Job submission failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()