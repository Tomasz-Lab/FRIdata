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

# Check if DEEPFRI_PATH is set, if not then throw an error
if [[ ! -v DEEPFRI_PATH ]]; then
    echo "Error: DEEPFRI_PATH environment variable is not set."
    exit 1
fi

# Check if CONDA_ENV_PATH is set, if not then set a default path
if [[ ! -v CONDA_ENV_PATH ]]; then
    CONDA_ENV_PATH="$DEEPFRI_PATH/conda_dev"
fi

CONDA_DIR="$GROUP_DIR/.conda"

module load miniconda3
conda config --add pkgs_dirs "$CONDA_DIR"

# Create environment from base YAML (without PyTorch)
conda env create --prefix $CONDA_ENV_PATH --file "$DEEPFRI_PATH/FRIdata/toolbox_env_conda.yml"

conda config --set auto_activate_base false

source activate $CONDA_ENV_PATH

# Install PyTorch based on mode
if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only PyTorch..."
    conda install -y pytorch cpuonly -c pytorch
else
    echo "Installing GPU-enabled PyTorch..."
    conda install -y pytorch-gpu
fi

# Install ESM (requires PyTorch to be installed first)
echo "Installing ESM..."
pip install esm

echo "Installation complete!"
