#!/usr/bin/env bash
# =============================================================================
# setup_hpc.sh — Install pipeline dependencies into the base conda environment
# =============================================================================
# Run once from the project directory:
#   bash setup_hpc.sh
# =============================================================================

set -e

echo "Installing PyTorch with CUDA 11.8 support ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing remaining dependencies ..."
pip install \
    scikit-learn \
    pandas \
    numpy \
    Pillow \
    tqdm \
    pyyaml \
    umap-learn \
    matplotlib \
    seaborn \
    grad-cam \
    scipy

echo ""
echo "================================================================"
echo "Setup complete. Submit the job with:"
echo "  sbatch run_job.sh"
echo "================================================================"
