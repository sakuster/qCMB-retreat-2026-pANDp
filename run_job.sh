#!/usr/bin/env bash
# =============================================================================
# run_job.sh — SLURM batch job script for Alpine HPC
# =============================================================================
# Submit with:
#   sbatch run_job.sh
#
# Monitor with:
#   squeue -u $USER
#
# View output:
#   cat logs/prion_pipeline.out
# =============================================================================

#SBATCH --job-name=prion_pipeline
#SBATCH --output=logs/prion_pipeline.out
#SBATCH --error=logs/prion_pipeline.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # CPU workers for data loading
#SBATCH --mem=32G                     # Memory — increase if you have many large images
#SBATCH --time=08:00:00               # Max runtime (HH:MM:SS). 8 hours is generous.
#SBATCH --account=YOUR_ACCOUNT_HERE   # Replace with your Alpine allocation account

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# Load the same modules used during setup
module purge
module load python/3.11.4
module load cuda/11.8.0
module load gcc/11.2.0

# Activate the virtual environment created by setup_hpc.sh
source venv/bin/activate

# Run the pipeline
python run.py --config config.yaml

echo "Job finished: $(date)"
