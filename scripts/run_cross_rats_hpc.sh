#!/bin/bash
#SBATCH --job-name=nm_theta_cross_rats
#SBATCH --partition=gpu  # or cpu, depending on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Adjust based on your needs
#SBATCH --mem=64G  # Memory - adjust based on your data size
#SBATCH --time=24:00:00  # Max runtime
#SBATCH --output=nm_theta_cross_rats_%j.out
#SBATCH --error=nm_theta_cross_rats_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@domain.com

# Load required modules (adjust for your cluster)
module purge
module load python/3.9
module load cuda/11.8  # if using GPU

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/path/to/your/eeg-near_mistakes/src"
export OMP_NUM_THREADS=16

# Create output directory
mkdir -p /scratch/your_username/nm_theta_results

# Change to working directory
cd /path/to/your/eeg-near_mistakes/src/core

# Run the analysis
echo "Starting cross-rats analysis at $(date)"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi  # Remove if not using GPU

# Run with specific parameters
python nm_theta_cross_rats.py \
    --roi "1,2,3" \
    --freq_min 1.0 \
    --freq_max 12.0 \
    --n_freqs 256 \
    --window_duration 2.0 \
    --method mne \
    --save_path /scratch/your_username/nm_theta_results \
    --pkl_path /path/to/your/data/all_eeg_data.pkl \
    --verbose

echo "Analysis completed at $(date)"

# Optional: Copy results to permanent storage
# cp -r /scratch/your_username/nm_theta_results /home/your_username/results/