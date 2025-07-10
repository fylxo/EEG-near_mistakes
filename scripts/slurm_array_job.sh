#!/bin/bash
#SBATCH --job-name=nm_theta_array
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nm_theta_rat_%A_%a.out
#SBATCH --error=logs/nm_theta_rat_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com

# Array job parameters (will be set by submission script)
# #SBATCH --array=1-N

# ============================================================================
# SLURM Array Job for NM Theta Cross-Rats Analysis
# 
# This script processes one rat per job using SLURM array functionality.
# Each job processes a different rat based on SLURM_ARRAY_TASK_ID.
# ============================================================================

# Configuration
DATA_PATH="/path/to/your/data/all_eeg_data.pkl"
FREQ_FILE_PATH="/path/to/your/data/config/frequencies.txt"
SAVE_PATH="/path/to/your/results"
PROJECT_DIR="/path/to/your/eeg-near_mistakes"
RAT_CONFIG_FILE="rat_config.json"

# Analysis parameters
ROI="1,2,3"
FREQ_MIN=1.0
FREQ_MAX=45.0
WINDOW_DURATION=2.0
N_CYCLES_FACTOR=3.0

# Load modules (adjust for your cluster)
module purge
module load python/3.9

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}/src"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=8

# Get job information
echo "============================================"
echo "SLURM Array Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Memory allocated: ${SLURM_MEM_PER_NODE}M"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "============================================"

# Navigate to project directory
cd ${PROJECT_DIR}

# Check if rat configuration file exists
if [ ! -f "$RAT_CONFIG_FILE" ]; then
    echo "ERROR: Rat configuration file not found: $RAT_CONFIG_FILE"
    echo "Please run: python scripts/discover_rats.py --pkl_path $DATA_PATH"
    exit 1
fi

# Extract rat ID for this array task
RAT_ID=$(python3 -c "
import json
with open('$RAT_CONFIG_FILE', 'r') as f:
    config = json.load(f)
rat_id = config['rat_id_map']['$SLURM_ARRAY_TASK_ID']
print(rat_id)
")

if [ -z "$RAT_ID" ]; then
    echo "ERROR: Could not determine rat ID for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing rat: $RAT_ID"
echo "============================================"

# Create logs directory
mkdir -p logs

# Create rat-specific save path
RAT_SAVE_PATH="${SAVE_PATH}/rat_${RAT_ID}"
mkdir -p "$RAT_SAVE_PATH"

# Memory monitoring function
monitor_memory() {
    MEMORY_LOG="logs/memory_rat_${RAT_ID}.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Starting memory monitoring for rat $RAT_ID" > $MEMORY_LOG
    
    while true; do
        MEM_USAGE=$(ps -o pid,ppid,pmem,pcpu,comm -p $$ | tail -1)
        FREE_MEM=$(free -h | grep '^Mem:' | awk '{print $7}')
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Usage: $MEM_USAGE, Free: $FREE_MEM" >> $MEMORY_LOG
        sleep 300  # Log every 5 minutes
    done &
    MONITOR_PID=$!
}

# Cleanup function
cleanup() {
    echo "Analysis interrupted or completed at $(date)"
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S'): Analysis ended for rat $RAT_ID" >> logs/memory_rat_${RAT_ID}.log
    free -h >> logs/memory_rat_${RAT_ID}.log
}

# Set up signal handling
trap cleanup EXIT INT TERM

# Start memory monitoring
monitor_memory

# System info
echo "System Information:"
echo "Memory:"
free -h
echo ""
echo "CPU:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread)" | head -3
echo ""

# Run the analysis
echo "Starting analysis for rat $RAT_ID at $(date)"
echo "Command: python src/core/nm_theta_cross_rats.py --rat_ids $RAT_ID --roi \"$ROI\" --freq_min $FREQ_MIN --freq_max $FREQ_MAX --freq_file_path \"$FREQ_FILE_PATH\" --window_duration $WINDOW_DURATION --n_cycles_factor $N_CYCLES_FACTOR --save_path \"$RAT_SAVE_PATH\" --pkl_path \"$DATA_PATH\" --verbose"

python src/core/nm_theta_cross_rats.py \
    --rat_ids "$RAT_ID" \
    --roi "$ROI" \
    --freq_min $FREQ_MIN \
    --freq_max $FREQ_MAX \
    --freq_file_path "$FREQ_FILE_PATH" \
    --window_duration $WINDOW_DURATION \
    --n_cycles_factor $N_CYCLES_FACTOR \
    --save_path "$RAT_SAVE_PATH" \
    --pkl_path "$DATA_PATH" \
    --verbose 2>&1 | tee logs/analysis_rat_${RAT_ID}.log

# Check exit status
EXIT_STATUS=${PIPESTATUS[0]}

if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Analysis completed successfully for rat $RAT_ID at $(date)"
    echo "Results saved to: $RAT_SAVE_PATH"
    
    # Create completion marker
    echo "$(date): Analysis completed successfully" > "${RAT_SAVE_PATH}/COMPLETED"
    
    # Log final memory usage
    echo "Final memory usage:"
    free -h
    
else
    echo "❌ Analysis failed for rat $RAT_ID with exit code $EXIT_STATUS at $(date)"
    echo "Check logs:"
    echo "  - Analysis log: logs/analysis_rat_${RAT_ID}.log"
    echo "  - Memory log: logs/memory_rat_${RAT_ID}.log"
    echo "  - SLURM output: logs/nm_theta_rat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    echo "  - SLURM error: logs/nm_theta_rat_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    
    # Create failure marker
    echo "$(date): Analysis failed with exit code $EXIT_STATUS" > "${RAT_SAVE_PATH}/FAILED"
fi

echo "Job finished at $(date) with exit status $EXIT_STATUS"
echo "============================================"