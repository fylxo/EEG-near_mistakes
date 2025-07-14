# EEG Near-Mistakes Analysis Scripts

This directory contains a comprehensive suite of scripts for EEG theta analysis in near-mistakes paradigms using SLURM cluster computing.

## 📋 Overview

The workflow processes EEG data from multiple rats to identify theta activity patterns during near-mistake events. The analysis uses wavelets, cross-rat aggregation, and statistical visualization to understand neural dynamics across subjects.

---

## 🔧 Core Processing Scripts

### **Data Preparation**
- **`create_eeg_data_file.py`** - Converts raw MAT files to processed EEG data format
- **`discover_rats.py`** - Automatically discovers available rat subjects for batch processing

### **SLURM Cluster Processing** 
- **`submit_array_jobs.py`** - Main orchestration script for SLURM array job submission
- **`slurm_array_job.sh`** - SLURM template for individual rat processing (high-memory, parallel)

### **Session Aggregation**
- **`aggregate_sessions_to_multi.py`** - Aggregates individual session results into multi-session summaries
- **`aggregate_sessions_batch.py`** - Batch processes multiple rats for session aggregation
- **`slurm_aggregate_sessions.sh`** - SLURM script for memory-intensive session aggregation
- **`submit_aggregate_job.py`** - Submits SLURM jobs for session aggregation

### **Final Analysis**
- **`aggregate_results.py`** - Cross-rat aggregation with spectrograms and statistical analysis

---

## 🔄 Recovery & Maintenance Scripts

### **Results Recovery**
- **`results_recovery_workflow.py`** - Complete workflow for recovering from partial SLURM failures

### **Utilities**
- **`cleanup_results.py`** - Manages disk space by cleaning analysis result folders
- **`reorganize_hpc_results.py`** - Reorganizes HPC result directory structures

### **Debugging**
- **`debug_memory_usage.py`** - Diagnoses memory usage and system limits
- **`debug_session_structure.py`** - Examines structure of session result files

---

## 🚀 Quick Start

### **Standard Workflow:**
```bash
# 1. Submit array jobs for all rats
python scripts/submit_array_jobs.py \
  --data_path /path/to/all_eeg_data.pkl \
  --project_dir /path/to/eeg-near_mistakes \
  --save_path /path/to/results

# 2. After jobs complete, aggregate results
python scripts/aggregate_results.py \
  --results_path /path/to/results \
  --roi 1,2,3 \
  --freq_min 3.0 \
  --freq_max 8.0
```

### **Recovery from Partial Failures:**
```bash
# Recover partial SLURM results
python scripts/results_recovery_workflow.py \
  --results_dir /path/to/results \
  --verbose
```

---

## 📊 Analysis Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--roi` | Brain regions of interest | "1,2,3" | Corresponds to specific electrode channels |
| `--freq_min/max` | Frequency range (Hz) | 1.0-45.0 | Theta: typically 3-8 Hz |
| `--n_freqs` | Number of frequency bins | 256 | High-resolution spectral analysis |
| `--n_cycles` | Morlet wavelet cycles | 5 | Fixed cycles across frequencies |
| `--window_duration` | Event window (s) | 2.0 | Time around near-mistake events |

---

## 💾 Resource Requirements

### **SLURM Job Resources:**
- **Memory**: 128-256G per rat (adjustable based on frequency resolution)
- **Time**: 24-48 hours per rat
- **Storage**: ~5-10GB per rat for results
- **CPUs**: 4-8 cores per job

### **Total Project Scale:**
- **Rats**: ~14 subjects
- **Sessions**: 300+ total across all rats  
- **Frequencies**: 264 frequency bins (1.01-45.00 Hz)
- **Final output**: Cross-rat spectrograms and statistical aggregations

---

## 📁 Output Structure

```
results/
├── rat_1055/
│   ├── multi_session_results.pkl    # Aggregated sessions for this rat
│   ├── session_*/                   # Individual session results
│   └── COMPLETED                    # Success marker
├── rat_9591/
│   └── ...
└── cross_rats_aggregated/
    ├── cross_rats_aggregated_results.pkl
    ├── cross_rats_spectrograms.png
    └── analysis_metadata.json
```

---

## 🔍 Monitoring & Troubleshooting

### **Job Monitoring:**
```bash
# Check job status
squeue -u $USER --name=nm_theta_array

# Monitor specific job
scontrol show job <JOB_ID>

# Check results
ls results/rat_*/COMPLETED
ls results/rat_*/FAILED
```

### **Common Issues:**
- **Memory errors**: Increase `--memory_per_job` to 384G
- **Time limits**: Extend `--time_limit` to 72:00:00  
- **ROI compatibility**: Some rats may have different channel mappings
- **Partial failures**: Use recovery scripts to salvage successful sessions

---

## 🧬 Analysis Details

### **Workflow Steps:**
1. **Preprocessing**: Morlet wavelet transform on EEG signals
2. **Event detection**: Identify near-mistake behavioral events
3. **Spectral analysis**: Compute time-frequency representations
4. **Session aggregation**: Average across sessions within each rat
5. **Cross-rat analysis**: Statistical aggregation across all subjects
6. **Visualization**: Generate spectrograms and statistical plots

### **Scientific Output:**
- **Individual rat results**: Session-averaged spectrograms per rat
- **Cross-rat spectrograms**: Population-level theta activity patterns
- **Statistical analysis**: Mean ± SEM across subjects
- **Colormap range**: Optimized for theta power visualization ([-0.42, 0.22])

---

## 🔬 Technical Notes

- **Frequency resolution**: 256 frequencies for high-resolution analysis
- **Wavelet parameters**: 5 cycles fixed across all frequencies for consistent temporal resolution
- **Memory optimization**: Results processed in chunks to handle 17GB+ datasets
- **Fault tolerance**: Each rat processed independently; failed jobs can be resubmitted
- **Compatibility**: Handles different channel mappings across rat subjects

---

## 📚 Dependencies

- **Core**: `numpy`, `scipy`, `matplotlib`, `mne`, `pickle`
- **Cluster**: SLURM workload manager
- **Storage**: High-capacity storage for intermediate and final results
- **Environment**: Python 3.8+ with scientific computing stack