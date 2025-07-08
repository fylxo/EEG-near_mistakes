#!/usr/bin/env python3
"""
Batch Processing Strategies for HPC Deployment

This script provides different strategies for running cross-rats analysis
on HPC systems, optimized for different scenarios and resource constraints.
"""

import os
from typing import List, Dict, Tuple
from hpc_config import ANALYSIS_PRESETS, generate_slurm_script

def create_single_large_job_strategy(preset_name: str = 'high_freq_analysis', 
                                   output_dir: str = 'hpc_scripts') -> str:
    """
    Strategy 1: Single large job processing all rats sequentially.
    
    Pros:
    - Simple to manage
    - Automatic checkpoint/resume
    - Efficient resource usage
    
    Cons:
    - All rats fail if job fails
    - Long queue times for large memory requests
    
    Parameters:
    -----------
    preset_name : str
        Analysis preset to use
    output_dir : str
        Directory to save scripts
        
    Returns:
    --------
    str
        Path to generated script
    """
    os.makedirs(output_dir, exist_ok=True)
    script_file = os.path.join(output_dir, f'single_job_{preset_name}.sh')
    
    generate_slurm_script(preset_name, script_file)
    
    return script_file

def create_parallel_jobs_by_rat_strategy(rat_ids: List[str], 
                                        preset_name: str = 'high_freq_analysis',
                                        memory_per_job_gb: int = 128,
                                        output_dir: str = 'hpc_scripts') -> List[str]:
    """
    Strategy 2: Parallel jobs, one per rat.
    
    Pros:
    - Fault tolerant (one rat failure doesn't affect others)
    - Faster parallel processing
    - Lower memory per job
    
    Cons:
    - More complex job management
    - Higher scheduler overhead
    - Need to aggregate results manually
    
    Parameters:
    -----------
    rat_ids : List[str]
        List of rat IDs to process
    preset_name : str
        Base analysis preset
    memory_per_job_gb : int
        Memory allocation per job
    output_dir : str
        Directory to save scripts
        
    Returns:
    --------
    List[str]
        List of paths to generated scripts
    """
    os.makedirs(output_dir, exist_ok=True)
    script_files = []
    
    # Get base preset
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    base_preset = ANALYSIS_PRESETS[preset_name].copy()
    
    for rat_id in rat_ids:
        # Create per-rat preset
        rat_preset = base_preset.copy()
        rat_preset['rat_ids'] = [rat_id]
        rat_preset['memory_gb'] = memory_per_job_gb
        rat_preset['time_hours'] = min(base_preset['time_hours'], 24)  # Shorter time per rat
        
        # Generate script
        rat_preset_name = f"{preset_name}_rat_{rat_id}"
        script_file = os.path.join(output_dir, f'rat_{rat_id}_{preset_name}.sh')
        
        # Temporarily add to presets
        from hpc_config import ANALYSIS_PRESETS
        ANALYSIS_PRESETS[rat_preset_name] = rat_preset
        
        generate_slurm_script(rat_preset_name, script_file)
        script_files.append(script_file)
        
        # Clean up temporary preset
        del ANALYSIS_PRESETS[rat_preset_name]
    
    # Create job submission script
    submit_script = os.path.join(output_dir, 'submit_all_rats.sh')
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all rat processing jobs\n\n")
        
        for script_file in script_files:
            job_name = os.path.basename(script_file).replace('.sh', '')
            f.write(f"echo 'Submitting {job_name}'\n")
            f.write(f"sbatch {script_file}\n")
        
        f.write("\necho 'All jobs submitted!'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")
    
    os.chmod(submit_script, 0o755)
    script_files.append(submit_script)
    
    return script_files

def create_staged_processing_strategy(rat_ids: List[str],
                                    preset_name: str = 'high_freq_analysis',
                                    rats_per_batch: int = 3,
                                    output_dir: str = 'hpc_scripts') -> List[str]:
    """
    Strategy 3: Staged processing in batches.
    
    Pros:
    - Balanced between single job and parallel jobs
    - Manageable resource requests
    - Some fault tolerance
    
    Cons:
    - Sequential batch dependency
    - Complex scheduling
    
    Parameters:
    -----------
    rat_ids : List[str]
        List of rat IDs to process
    preset_name : str
        Base analysis preset
    rats_per_batch : int
        Number of rats per batch
    output_dir : str
        Directory to save scripts
        
    Returns:
    --------
    List[str]
        List of paths to generated scripts
    """
    os.makedirs(output_dir, exist_ok=True)
    script_files = []
    
    # Get base preset
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    base_preset = ANALYSIS_PRESETS[preset_name].copy()
    
    # Split rats into batches
    batches = [rat_ids[i:i + rats_per_batch] for i in range(0, len(rat_ids), rats_per_batch)]
    
    for batch_idx, batch_rats in enumerate(batches):
        # Create batch preset
        batch_preset = base_preset.copy()
        batch_preset['rat_ids'] = batch_rats
        batch_preset['memory_gb'] = min(base_preset['memory_gb'], 
                                       max(128, len(batch_rats) * 64))  # Scale memory with batch size
        batch_preset['time_hours'] = min(base_preset['time_hours'], 
                                        max(12, len(batch_rats) * 8))  # Scale time with batch size
        
        # Generate script
        batch_preset_name = f"{preset_name}_batch_{batch_idx + 1}"
        script_file = os.path.join(output_dir, f'batch_{batch_idx + 1:02d}_{preset_name}.sh')
        
        # Temporarily add to presets
        from hpc_config import ANALYSIS_PRESETS
        ANALYSIS_PRESETS[batch_preset_name] = batch_preset
        
        generate_slurm_script(batch_preset_name, script_file)
        script_files.append(script_file)
        
        # Clean up temporary preset
        del ANALYSIS_PRESETS[batch_preset_name]
    
    # Create sequential submission script
    submit_script = os.path.join(output_dir, 'submit_staged.sh')
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit staged batch processing jobs\n\n")
        
        previous_job_id = None
        for i, script_file in enumerate(script_files):
            job_name = os.path.basename(script_file).replace('.sh', '')
            
            if previous_job_id is None:
                f.write(f"echo 'Submitting batch {i+1}'\n")
                f.write(f"JOB_{i+1}=$(sbatch --parsable {script_file})\n")
                f.write(f"echo 'Batch {i+1} job ID: $JOB_{i+1}'\n\n")
                previous_job_id = f"$JOB_{i+1}"
            else:
                f.write(f"echo 'Submitting batch {i+1} (depends on batch {i})'\n")
                f.write(f"JOB_{i+1}=$(sbatch --parsable --dependency=afterok:{previous_job_id} {script_file})\n")
                f.write(f"echo 'Batch {i+1} job ID: $JOB_{i+1}'\n\n")
                previous_job_id = f"$JOB_{i+1}"
        
        f.write("echo 'All staged jobs submitted!'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")
    
    os.chmod(submit_script, 0o755)
    script_files.append(submit_script)
    
    return script_files

def create_memory_adaptive_strategy(rat_ids: List[str],
                                  preset_name: str = 'high_freq_analysis',
                                  output_dir: str = 'hpc_scripts') -> Dict[str, List[str]]:
    """
    Strategy 4: Memory-adaptive processing based on estimated requirements.
    
    Categorizes rats into high/medium/low memory requirements and creates
    separate job scripts for each category.
    
    Parameters:
    -----------
    rat_ids : List[str]
        List of rat IDs to process
    preset_name : str
        Base analysis preset
    output_dir : str
        Directory to save scripts
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping memory category to script files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize rats by memory requirements (simplified heuristic)
    # In reality, you might analyze session lengths or other factors
    high_memory_rats = rat_ids[:len(rat_ids)//3]  # First third - assume high memory
    medium_memory_rats = rat_ids[len(rat_ids)//3:2*len(rat_ids)//3]  # Middle third
    low_memory_rats = rat_ids[2*len(rat_ids)//3:]  # Last third
    
    categories = {
        'high_memory': {
            'rats': high_memory_rats,
            'memory_gb': 384,
            'time_hours': 48,
            'description': 'High memory rats (long sessions, many frequencies)'
        },
        'medium_memory': {
            'rats': medium_memory_rats,
            'memory_gb': 192,
            'time_hours': 24,
            'description': 'Medium memory rats (standard sessions)'
        },
        'low_memory': {
            'rats': low_memory_rats,
            'memory_gb': 128,
            'time_hours': 12,
            'description': 'Low memory rats (short sessions, fewer frequencies)'
        }
    }
    
    script_files = {}
    
    # Get base preset
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    base_preset = ANALYSIS_PRESETS[preset_name].copy()
    
    for category, config in categories.items():
        if not config['rats']:  # Skip empty categories
            continue
            
        # Create category preset
        cat_preset = base_preset.copy()
        cat_preset['rat_ids'] = config['rats']
        cat_preset['memory_gb'] = config['memory_gb']
        cat_preset['time_hours'] = config['time_hours']
        
        # Generate script
        cat_preset_name = f"{preset_name}_{category}"
        script_file = os.path.join(output_dir, f'{category}_{preset_name}.sh')
        
        # Temporarily add to presets
        from hpc_config import ANALYSIS_PRESETS
        ANALYSIS_PRESETS[cat_preset_name] = cat_preset
        
        generate_slurm_script(cat_preset_name, script_file)
        script_files[category] = [script_file]
        
        # Clean up temporary preset
        del ANALYSIS_PRESETS[cat_preset_name]
    
    # Create submission script for all categories
    submit_script = os.path.join(output_dir, 'submit_memory_adaptive.sh')
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit memory-adaptive processing jobs\n\n")
        
        for category, config in categories.items():
            if category in script_files:
                f.write(f"echo 'Submitting {category} ({len(config[\"rats\"])} rats)'\n")
                f.write(f"echo '  {config[\"description\"]}'\n")
                f.write(f"sbatch {script_files[category][0]}\n\n")
        
        f.write("echo 'All memory-adaptive jobs submitted!'\n")
        f.write("echo 'Monitor with: squeue -u $USER'\n")
    
    os.chmod(submit_script, 0o755)
    script_files['submit_all'] = [submit_script]
    
    return script_files

def generate_strategy_comparison_report(rat_ids: List[str], 
                                      preset_name: str = 'high_freq_analysis',
                                      output_file: str = 'strategy_comparison.md') -> str:
    """
    Generate a comparison report of different batch processing strategies.
    
    Parameters:
    -----------
    rat_ids : List[str]
        List of rat IDs to process
    preset_name : str
        Analysis preset to use for comparison
    output_file : str
        Output file for the report
        
    Returns:
    --------
    str
        Path to the generated report
    """
    base_preset = ANALYSIS_PRESETS.get(preset_name, {})
    n_rats = len(rat_ids)
    
    report = f"""# HPC Batch Processing Strategy Comparison

## Analysis Configuration
- **Preset**: {preset_name}
- **Number of rats**: {n_rats}
- **Memory per rat (estimated)**: {base_preset.get('memory_gb', 256) // max(1, n_rats)} GB
- **Total estimated time**: {base_preset.get('time_hours', 48)} hours

## Strategy Comparison

### 1. Single Large Job
**Command**: `python batch_processing_strategies.py --strategy single --preset {preset_name}`

- **Pros**: Simple management, automatic checkpoints, efficient resource usage
- **Cons**: All rats fail if job fails, long queue times
- **Memory**: {base_preset.get('memory_gb', 256)} GB
- **Time**: {base_preset.get('time_hours', 48)} hours
- **Jobs**: 1
- **Best for**: Small number of rats, reliable cluster

### 2. Parallel Jobs by Rat
**Command**: `python batch_processing_strategies.py --strategy parallel --preset {preset_name}`

- **Pros**: Fault tolerant, fast parallel processing, lower memory per job
- **Cons**: Complex management, higher scheduler overhead
- **Memory**: {base_preset.get('memory_gb', 256) // max(1, n_rats)} GB per job
- **Time**: {base_preset.get('time_hours', 48) // max(1, n_rats)} hours per job
- **Jobs**: {n_rats}
- **Best for**: Large number of rats, unreliable nodes

### 3. Staged Batches (3 rats per batch)
**Command**: `python batch_processing_strategies.py --strategy staged --preset {preset_name} --batch-size 3`

- **Pros**: Balanced approach, manageable resources, some fault tolerance
- **Cons**: Sequential dependencies, complex scheduling
- **Memory**: 192-256 GB per batch
- **Time**: 12-24 hours per batch
- **Jobs**: {(n_rats + 2) // 3}
- **Best for**: Medium number of rats, mixed reliability

### 4. Memory-Adaptive
**Command**: `python batch_processing_strategies.py --strategy adaptive --preset {preset_name}`

- **Pros**: Optimized resource usage, faster queue times for small jobs
- **Cons**: Complex setup, requires memory estimation
- **Memory**: 128-384 GB (adaptive)
- **Time**: 12-48 hours (adaptive)
- **Jobs**: 2-3 (by memory category)
- **Best for**: Mixed rat complexity, resource-constrained cluster

## Recommendations

- **For your analysis** ({n_rats} rats, {preset_name}): 
  - If cluster is reliable: **Single Large Job**
  - If cluster is unreliable: **Parallel Jobs by Rat**
  - If unsure: **Staged Batches**

## Resource Requirements Summary

| Strategy | Total Memory | Peak Jobs | Queue Time | Fault Tolerance |
|----------|--------------|-----------|------------|-----------------|
| Single | {base_preset.get('memory_gb', 256)} GB | 1 | Long | Low |
| Parallel | {base_preset.get('memory_gb', 256)} GB | {n_rats} | Short | High |
| Staged | {base_preset.get('memory_gb', 256)} GB | 1 | Medium | Medium |
| Adaptive | {base_preset.get('memory_gb', 256)} GB | 2-3 | Short-Medium | Medium |

"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HPC batch processing strategies')
    parser.add_argument('--strategy', choices=['single', 'parallel', 'staged', 'adaptive', 'report'], 
                       required=True, help='Batch processing strategy')
    parser.add_argument('--preset', default='high_freq_analysis', help='Analysis preset')
    parser.add_argument('--rat-ids', nargs='+', default=['9151', '9152', '9153', '9154', '9155'],
                       help='Rat IDs to process')
    parser.add_argument('--output-dir', default='hpc_scripts', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size for staged strategy')
    
    args = parser.parse_args()
    
    print(f"Generating {args.strategy} strategy for {len(args.rat_ids)} rats...")
    
    if args.strategy == 'single':
        script_file = create_single_large_job_strategy(args.preset, args.output_dir)
        print(f"Generated: {script_file}")
        
    elif args.strategy == 'parallel':
        script_files = create_parallel_jobs_by_rat_strategy(args.rat_ids, args.preset, 
                                                           output_dir=args.output_dir)
        print(f"Generated {len(script_files)} files in {args.output_dir}")
        
    elif args.strategy == 'staged':
        script_files = create_staged_processing_strategy(args.rat_ids, args.preset, 
                                                        args.batch_size, args.output_dir)
        print(f"Generated {len(script_files)} batch files in {args.output_dir}")
        
    elif args.strategy == 'adaptive':
        script_files = create_memory_adaptive_strategy(args.rat_ids, args.preset, args.output_dir)
        total_files = sum(len(files) for files in script_files.values())
        print(f"Generated {total_files} files across {len(script_files)} categories in {args.output_dir}")
        
    elif args.strategy == 'report':
        report_file = generate_strategy_comparison_report(args.rat_ids, args.preset)
        print(f"Generated comparison report: {report_file}")
    
    print("\nNext steps:")
    print("1. Review generated scripts")
    print("2. Adjust memory/time requirements for your cluster")
    print("3. Submit jobs: sbatch <script_file>")
    print("4. Monitor: squeue -u $USER")