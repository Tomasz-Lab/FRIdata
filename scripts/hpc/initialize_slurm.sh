#!/bin/bash
# Installation script for the FRIdata HPC environment (venv + pip).
# Usage: ./initialize_slurm.sh [--cpu]
#   --cpu: Install CPU-only PyTorch (for CI or non-GPU systems)
#   Default: Install GPU-enabled PyTorch matched to the node's CUDA driver
#
# Environment variables:
#   DEEPFRI_PATH  (required) parent directory of the FRIdata repo
#   VENV_PATH     (optional) virtualenv location, defaults to $DEEPFRI_PATH/.venv

set -e

CPU_ONLY=false
for arg in "$@"; do
    case $arg in
        --cpu)
            CPU_ONLY=true
            ;;
    esac
done

# Check if DEEPFRI_PATH is set, if not then throw an error
if [[ ! -v DEEPFRI_PATH ]]; then
    echo "Error: DEEPFRI_PATH environment variable is not set."
    exit 1
fi

# Virtualenv location (mirrors common_slurm.sh).
if [[ ! -v VENV_PATH ]]; then
    VENV_PATH="$DEEPFRI_PATH/.venv"
fi

# Try loading GCC and a Python module in a robust way (handle varied names).
LOADED_GCC=false
LOADED_PYTHON=false
if command -v module >/dev/null 2>&1; then
    GCC_CANDIDATES=(gcc GCC)
    for MOD in "${GCC_CANDIDATES[@]}"; do
        if module load "$MOD" >/dev/null 2>&1; then
            echo "Loaded module: $MOD"
            LOADED_GCC=true
            break
        fi
    done

    # Cluster-specific: adjust these names to match `module avail python`.
    PYTHON_CANDIDATES=(python Python python3 Python3)
    for MOD in "${PYTHON_CANDIDATES[@]}"; do
        if module load "$MOD" >/dev/null 2>&1; then
            echo "Loaded module: $MOD"
            LOADED_PYTHON=true
            break
        fi
    done
fi

if [ "$LOADED_PYTHON" = false ]; then
    echo "Error: Could not load a Python module."
    exit 1
fi

# Create the virtualenv.
echo "Creating virtualenv at '$VENV_PATH'..."
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip

echo "Installing FRIdata core dependencies..."
pip install -e "$DEEPFRI_PATH/FRIdata"

# Install PyTorch/ESM matched to the node (GPU by default, CPU with --cpu).
PYTORCH_ARGS=()
if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only PyTorch..."
    PYTORCH_ARGS+=(--cpu)
else
    echo "Installing GPU-enabled PyTorch..."
fi
PYTHON="$VENV_PATH/bin/python" bash "$DEEPFRI_PATH/FRIdata/scripts/install_pytorch.sh" "${PYTORCH_ARGS[@]}"

echo "Installation complete!"
