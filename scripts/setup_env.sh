#!/bin/bash
# Create a FRIdata conda/mamba environment and install Python + PyTorch dependencies.
#
# Pip packages are installed from requirements-fridata.txt after the conda solve so
# installs work even when this repository checkout is not writable (libmamba writes
# temporary pip requirement files next to the environment YAML).
#
# Usage: ./scripts/setup_env.sh [options]
#   -n, --name NAME   Environment name (default: fridata_env)
#   --cpu             Install CPU-only PyTorch
#   --skip-pytorch    Skip PyTorch/ESM installation
#   -h, --help        Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="fridata_env"
CPU_ONLY=false
SKIP_PYTORCH=false

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '/^$/d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            ENV_NAME="$2"
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

if command -v mamba >/dev/null 2>&1; then
    MAMBA=mamba
elif command -v micromamba >/dev/null 2>&1; then
    MAMBA=micromamba
else
    echo "Error: mamba or micromamba is required but not found on PATH." >&2
    exit 1
fi

YAML_SRC="$REPO_ROOT/fridata_env_conda.yml"
REQS="$REPO_ROOT/requirements-fridata.txt"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

if [ ! -f "$YAML_SRC" ]; then
    echo "Error: environment file not found: $YAML_SRC" >&2
    exit 1
fi

if [ ! -f "$REQS" ]; then
    echo "Error: requirements file not found: $REQS" >&2
    exit 1
fi

# Copy the YAML to a writable directory. This avoids libmamba permission errors when
# the repository checkout itself is read-only and keeps older mamba versions working.
cp "$YAML_SRC" "$WORKDIR/fridata_env_conda.yml"

echo "Creating conda environment '$ENV_NAME'..."
if "$MAMBA" env create -f "$WORKDIR/fridata_env_conda.yml" -n "$ENV_NAME" -y; then
    :
elif "$MAMBA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists; updating conda packages..."
    "$MAMBA" env update -n "$ENV_NAME" -f "$WORKDIR/fridata_env_conda.yml" -y
else
    echo "Error: failed to create environment '$ENV_NAME'." >&2
    exit 1
fi

echo "Installing pip requirements from requirements-fridata.txt..."
export TMPDIR="${TMPDIR:-/tmp}"
"$MAMBA" run -n "$ENV_NAME" python -m pip install -r "$REQS"

if [ "$SKIP_PYTORCH" = false ]; then
    PYTORCH_ARGS=()
    if [ "$CPU_ONLY" = true ]; then
        PYTORCH_ARGS+=(--cpu)
    fi

    # install_pytorch.sh expects an activated environment.
    eval "$("$MAMBA" shell hook --shell bash)"
    mamba activate "$ENV_NAME"
    bash "$SCRIPT_DIR/install_pytorch.sh" "${PYTORCH_ARGS[@]}"
fi

cat <<EOF

Setup complete.

Activate the environment with:
  eval "\$($MAMBA shell hook --shell bash)"
  mamba activate $ENV_NAME

EOF
