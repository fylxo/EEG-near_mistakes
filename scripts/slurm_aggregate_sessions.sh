#!/bin/bash
#SBATCH --job-name=aggregate_sessions
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/aggregate_sessions_%j.out
#SBATCH --error=logs/aggregate_sessions_%j.err

# Script to aggregate partial session results into multi_session_results.pkl files
# Usage: sbatch --export=RESULTS_DIR=/path/to/results scripts/slurm_aggregate_sessions.sh

echo "==============================================="
echo "Session Aggregation Job Started"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Results directory: $RESULTS_DIR"
echo "==============================================="

# Load required modules (adjust for your cluster)
# module load python/3.8
# module load scipy-stack

# Activate virtual environment (same as array jobs)
source ${PROJECT_DIR}/eeg_analysis_env/bin/activate

# Ensure logs directory exists
mkdir -p logs

# Set up environment
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}/src"
export PYTHONUNBUFFERED=1

# Validate results directory
if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR environment variable not set"
    echo "Usage: sbatch --export=RESULTS_DIR=/path/to/results scripts/slurm_aggregate_sessions.sh"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Get project directory from script location
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Changing to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "Available memory:"
free -h

echo "Python version:"
python3 --version

echo "Starting session aggregation..."
echo "==============================================="

# Run the batch aggregation
python3 scripts/simple_batch_aggregate.py \
    --results_dir "$RESULTS_DIR" \
    --verbose

AGGREGATION_EXIT_CODE=$?

echo "==============================================="
echo "Session aggregation completed with exit code: $AGGREGATION_EXIT_CODE"

if [ $AGGREGATION_EXIT_CODE -eq 0 ]; then
    echo "✅ Session aggregation successful!"
    
    echo "Running cross-rat analysis..."
    python3 scripts/aggregate_results.py \
        --results_path "$RESULTS_DIR" \
        --roi "1,2,3" \
        --freq_min 3.0 \
        --freq_max 8.0 \
        --verbose
    
    CROSS_RAT_EXIT_CODE=$?
    
    if [ $CROSS_RAT_EXIT_CODE -eq 0 ]; then
        echo "✅ Cross-rat analysis successful!"
        echo "Final results available in: $RESULTS_DIR/cross_rats_aggregated/"
    else
        echo "❌ Cross-rat analysis failed with exit code: $CROSS_RAT_EXIT_CODE"
    fi
    
else
    echo "❌ Session aggregation failed with exit code: $AGGREGATION_EXIT_CODE"
fi

echo "==============================================="
echo "Job completed at: $(date)"
echo "==============================================="