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

# Try loading a Conda/Miniconda module in a robust way (handle varied names)
LOADED_MODULE=false
if command -v module >/dev/null 2>&1; then
    MODULE_CANDIDATES=(miniconda3 Miniconda3 miniconda Anaconda3 anaconda3)
    for MOD in "${MODULE_CANDIDATES[@]}"; do
        if module load "$MOD" >/dev/null 2>&1; then
            echo "Loaded module: $MOD"
            LOADED_MODULE=true
            break
        fi
    done
fi

if [ "$LOADED_MODULE" = false ]; then
    echo "Error: Could not load a Conda module."
    exit 1
fi

conda config --add pkgs_dirs "$CONDA_DIR"

# Create environment from base YAML (without PyTorch or pip-only deps)
ENV_WORKDIR="$(mktemp -d)"
trap 'rm -rf "$ENV_WORKDIR"' EXIT
cp "$DEEPFRI_PATH/FRIdata/fridata_env_conda.yml" "$ENV_WORKDIR/fridata_env_conda.yml"
conda env create --prefix $CONDA_ENV_PATH --file "$ENV_WORKDIR/fridata_env_conda.yml"

conda config --set auto_activate_base false

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_PATH

echo "Installing pip requirements..."
pip install -r "$DEEPFRI_PATH/FRIdata/requirements-fridata.txt"

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
