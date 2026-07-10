#!/bin/bash
# Create a FRIdata virtual environment and install its dependencies with pip.
#
# This creates a plain Python venv, installs FRIdata (editable) with its dev
# extras, and then installs a PyTorch build matched to your GPU driver via
# install_pytorch.sh. No conda/mamba required.
#
# Usage: ./scripts/setup_env.sh [options]
#   -p, --path PATH   Virtualenv directory (default: .venv)
#   --cpu             Install CPU-only PyTorch
#   --skip-pytorch    Skip PyTorch/ESM installation (core install only)
#   -h, --help        Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
CPU_ONLY=false
SKIP_PYTORCH=false

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '/^$/d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--path)
            VENV_PATH="$2"
            shift 2
            ;;
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --skip-pytorch)
            SKIP_PYTORCH=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Pick a Python interpreter (prefer python3).
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: no python3/python found on PATH." >&2
    exit 1
fi

echo "Creating virtual environment at '$VENV_PATH'..."
"$PYTHON" -m venv "$VENV_PATH"

# Use the venv's interpreter directly so this works without 'source activate'.
VENV_PY="$VENV_PATH/bin/python"
if [ ! -x "$VENV_PY" ]; then
    # Windows Git Bash layout.
    VENV_PY="$VENV_PATH/Scripts/python.exe"
fi

echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

if [ "$SKIP_PYTORCH" = true ]; then
    echo "Installing FRIdata with dev + test extras (no embeddings/PyTorch)..."
    "$VENV_PY" -m pip install -e "$REPO_ROOT[test]"
else
    echo "Installing FRIdata with dev extras..."
    "$VENV_PY" -m pip install -e "$REPO_ROOT[dev]"

    # install_pytorch.sh selects a torch wheel matching the machine's CUDA driver
    # and reinstalls torch/esm accordingly. It operates on the interpreter it is
    # given via the PYTHON env var.
    PYTORCH_ARGS=()
    if [ "$CPU_ONLY" = true ]; then
        PYTORCH_ARGS+=(--cpu)
    fi
    PYTHON="$VENV_PY" bash "$SCRIPT_DIR/install_pytorch.sh" "${PYTORCH_ARGS[@]}"
fi

cat <<EOF

Setup complete.

Activate the environment with:
  source "$VENV_PATH/bin/activate"

EOF
