#!/usr/bin/env bash
# =============================================================================
# setup_hpc.sh — Create conda environment and install all dependencies
# =============================================================================
# Run once from the project directory:
#   bash setup_hpc.sh
# =============================================================================

set -e

echo "Creating conda environment 'prion_pipeline' ..."
conda create -n prion_pipeline python=3.11 -y

echo "Activating environment ..."
source /curc/sw/anaconda3/2020.11/etc/profile.d/conda.sh
conda activate prion_pipeline

echo "Installing PyTorch with CUDA support ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing remaining dependencies ..."
pip install -r requirements.txt

echo ""
echo "================================================================"
echo "Setup complete. Submit the job with:"
echo "  sbatch run_job.sh"
echo "================================================================"
