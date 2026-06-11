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
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # CPU workers for data loading
#SBATCH --mem=32G                     # Memory — increase if you have many large images
#SBATCH --time=04:00:00               # Max runtime (HH:MM:SS).
#SBATCH --account=csu-general

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"

# Load the same modules used during setup
source /curc/sw/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate prion_pipeline

# Run the pipeline
python run.py --config config_alpine.yaml

echo "Job finished: $(date)"
