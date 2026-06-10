#!/usr/bin/env bash
# =============================================================================
# setup_hpc.sh — Set up the pipeline environment on Alpine HPC
# =============================================================================
# Run this once after logging into Alpine, from the project directory:
#   bash setup_hpc.sh
#
# It loads the required modules, creates a virtual environment in ./venv,
# and installs all Python dependencies including GPU-enabled PyTorch.
# =============================================================================

set -e

# --- Load Alpine modules ------------------------------------------------------
# These make Python, CUDA, and the compiler available on the compute node.
# If a module version listed here is unavailable, run:
#   module spider python
#   module spider cuda
# to see what versions are currently installed on Alpine.

module purge
module load python/3.11.4
module load cuda/11.8.0
module load gcc/11.2.0

echo "Loaded modules:"
module list 2>&1

# --- Create virtual environment -----------------------------------------------
echo ""
echo "Creating virtual environment in ./venv ..."
python -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

# --- Install PyTorch with CUDA support ----------------------------------------
# This installs the CUDA 11.8 build of PyTorch.
# If Alpine has a different CUDA version (check with: nvcc --version),
# update the --index-url below. Find the correct URL at: https://pytorch.org/get-started/locally/
echo "Installing PyTorch (GPU build for CUDA 11.8) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

# --- Install remaining dependencies -------------------------------------------
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
    scipy \
    --quiet

echo ""
echo "================================================================"
echo "Setup complete."
echo ""
echo "To run the pipeline interactively (for testing):"
echo "  source venv/bin/activate && python run.py"
echo ""
echo "To submit as a batch job:"
echo "  sbatch run_job.sh"
echo "================================================================"
