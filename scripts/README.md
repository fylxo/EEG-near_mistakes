# SLURM Array Jobs for Cross-Rats Analysis

This directory contains scripts to run the NM Theta cross-rats analysis on SLURM clusters using array jobs. Each array job processes one rat, providing fault tolerance and parallel processing.

## Files Overview

- `discover_rats.py` - Discovers available rat IDs from the data file
- `slurm_array_job.sh` - SLURM array job template (one rat per job)
- `submit_array_jobs.py` - Main submission script that orchestrates everything
- `README.md` - This file

## Quick Start

1. **Basic submission** (recommended):
   ```bash
   python scripts/submit_array_jobs.py \
     --data_path /path/to/your/data/all_eeg_data.pkl \
     --project_dir /path/to/your/eeg-near_mistakes \
     --save_path /path/to/your/results
   ```

2. **With custom parameters**:
   ```bash
   python scripts/submit_array_jobs.py \
     --data_path /path/to/your/data/all_eeg_data.pkl \
     --project_dir /path/to/your/eeg-near_mistakes \
     --save_path /path/to/your/results \
     --roi "1,2,3" \
     --freq_min 1.0 \
     --freq_max 45.0 \
     --memory_per_job 256G \
     --time_limit 48:00:00 \
     --email your_email@domain.com
   ```

3. **Test before submitting**:
   ```bash
   python scripts/submit_array_jobs.py \
     --data_path /path/to/your/data/all_eeg_data.pkl \
     --project_dir /path/to/your/eeg-near_mistakes \
     --save_path /path/to/your/results \
     --dry_run
   ```

## Step-by-Step Process

### 1. Rat Discovery
The script automatically discovers available rats:
```bash
python scripts/discover_rats.py \
  --pkl_path /path/to/your/data/all_eeg_data.pkl \
  --exclude_9442  # Optional: exclude rat 9442 due to compatibility issues
```

This creates `rat_config.json` with the rat ID mapping for array jobs.

### 2. Array Job Configuration
The submission script automatically configures the SLURM script with:
- Array range based on number of rats
- Cluster-specific paths
- Analysis parameters
- Resource requirements

### 3. Job Submission
Each array job processes one rat:
- Array task ID 1 → first rat
- Array task ID 2 → second rat
- etc.

### 4. Monitoring
After submission, use the generated monitoring script:
```bash
./monitor_jobs.sh
```

Or use standard SLURM commands:
```bash
squeue -u $USER --name=nm_theta_array
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,MaxRSS
```

## Resource Requirements

### Per-Rat Job Requirements:
- **Memory**: 128G (adjust based on your data size)
- **Time**: 24 hours (adjust based on your analysis complexity)
- **CPUs**: 8 cores
- **Storage**: ~5-10GB per rat for results

### Cluster Considerations:
- Total jobs = number of rats
- Each job is independent (fault tolerant)
- Failed jobs can be resubmitted individually
- Results are saved per rat in separate directories

## File Structure After Submission

```
project_dir/
├── logs/
│   ├── analysis_rat_<RAT_ID>.log
│   ├── memory_rat_<RAT_ID>.log
│   └── nm_theta_rat_<JOB_ID>_<ARRAY_ID>.out
├── results/
│   ├── rat_<RAT_ID>/
│   │   ├── cross_rats_aggregated_results.pkl
│   │   ├── cross_rats_spectrograms.png
│   │   └── COMPLETED (success marker)
│   └── rat_<RAT_ID>/
│       ├── analysis_files...
│       └── FAILED (failure marker)
├── rat_config.json
└── monitor_jobs.sh
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--roi` | ROI specification | "1,2,3" |
| `--freq_min` | Minimum frequency (Hz) | 1.0 |
| `--freq_max` | Maximum frequency (Hz) | 45.0 |
| `--window_duration` | Event window duration (s) | 2.0 |
| `--memory_per_job` | Memory per job | 128G |
| `--time_limit` | Time limit per job | 24:00:00 |
| `--partition` | SLURM partition | cpu |

## Troubleshooting

### Common Issues:

1. **Memory errors**: Increase `--memory_per_job`
2. **Time limit exceeded**: Increase `--time_limit`
3. **Missing data file**: Check `--data_path` and file permissions
4. **Module loading issues**: Adjust module commands in `slurm_array_job.sh`

### Checking Job Status:
```bash
# Overall status
squeue -u $USER

# Specific job details
scontrol show job <JOB_ID>

# Check individual rat results
ls results/rat_*/COMPLETED
ls results/rat_*/FAILED
```

### Resubmitting Failed Jobs:
```bash
# Find failed array tasks
sacct -j <JOB_ID> --format=JobID,State | grep FAILED

# Resubmit specific array tasks
sbatch --array=<FAILED_TASK_IDS> scripts/slurm_array_job_configured.sh
```

## Migration from Old Scripts

The old `batch_processing_strategies.py` and `hpc_config.py` scripts have been replaced with this simpler system. The new approach:

- ✅ Works with current `nm_theta_cross_rats.py` 
- ✅ Simpler configuration
- ✅ Better fault tolerance
- ✅ Easier monitoring
- ✅ Automatic rat discovery

## Advanced Usage

### Processing Subset of Rats:
```bash
# Edit rat_config.json manually to include only desired rats
# Then submit normally
```

### Custom Analysis Parameters:
```bash
python scripts/submit_array_jobs.py \
  --data_path /path/to/data.pkl \
  --project_dir /path/to/project \
  --save_path /path/to/results \
  --roi "frontal" \
  --freq_min 4.0 \
  --freq_max 8.0 \
  --window_duration 1.0
```

### Different Memory Requirements:
```bash
# For high-frequency analysis
--memory_per_job 384G --time_limit 72:00:00

# For quick testing
--memory_per_job 64G --time_limit 12:00:00
```