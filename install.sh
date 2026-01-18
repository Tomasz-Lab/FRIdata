#!/bin/bash
# Installation script for toolbox environment
# Usage: ./install.sh <GROUP_DIR> [--cpu]
#   --cpu: Install CPU-only PyTorch (for CI or non-GPU systems)
#   Default: Install GPU-enabled PyTorch

set -e

GROUP_DIR=$1
CPU_ONLY=false

# Parse optional flags
shift
for arg in "$@"; do
    case $arg in
        --cpu)
            CPU_ONLY=true
            ;;
    esac
done

if [ -z "$GROUP_DIR" ]; then
    echo "Usage: ./install.sh <GROUP_DIR> [--cpu]"
    exit 1
fi

CONDA_DIR="$GROUP_DIR/.conda"
conda config --add pkgs_dirs "$CONDA_DIR"

# Create environment from base YAML (without PyTorch)
conda env create --prefix $ENV_PATH --file "toolbox_env_conda.yml"

conda config --set auto_activate_base false

source activate $ENV_PATH

# Install PyTorch based on mode
if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only PyTorch..."
    conda install -y pytorch cpuonly -c pytorch -c conda-forge
else
    echo "Installing GPU-enabled PyTorch..."
    conda install -y pytorch-gpu -c conda-forge
fi

# Install ESM (requires PyTorch to be installed first)
echo "Installing ESM..."
pip install esm

echo "Installation complete!"
