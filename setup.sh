#!/usr/bin/env bash
# =============================================================================
# setup.sh — Create a Python virtual environment and install all dependencies
# =============================================================================
# Run this once before using the pipeline:
#   bash setup.sh
#
# After it finishes, open VSCode and select the interpreter:
#   Cmd+Shift+P  →  "Python: Select Interpreter"  →  choose  ./venv/bin/python
# =============================================================================

set -e  # Stop immediately if any command fails

echo "Creating virtual environment in ./venv ..."
python3 -m venv venv

echo "Activating environment and installing packages ..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo "================================================================"
echo "Setup complete."
echo ""
echo "To activate the environment manually:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  source venv/bin/activate && python run.py"
echo "================================================================"
