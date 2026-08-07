#!/bin/bash
# Create a FRIdata virtual environment and install its dependencies with pip.
#
# This creates a plain Python venv, installs FRIdata (editable) with its dev
# extras, and then installs a PyTorch build matched to your GPU driver via
# install_pytorch.sh. No conda/mamba required.
#
# FRIdata needs Python >= 3.11, so the script uses the first interpreter it
# finds that is new enough. An interpreter that is too old (for example a conda
# base environment on PATH) is skipped instead of failing later in pip.
#
# Usage: ./scripts/setup_env.sh [options]
#   -p, --path PATH   Virtualenv directory (default: .venv)
#   --python EXEC     Interpreter to build the venv with (default: auto-detect)
#   --cpu             Install CPU-only PyTorch
#   --skip-pytorch    Skip PyTorch/ESM installation (core install only)
#   -h, --help        Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
CPU_ONLY=false
SKIP_PYTORCH=false
# Keep in sync with requires-python in pyproject.toml.
MIN_MAJOR=3
MIN_MINOR=11
PYTHON="${PYTHON:-}"

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '/^$/d'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--path)
            VENV_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
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

# True if the given interpreter exists and is at least MIN_MAJOR.MIN_MINOR.
python_is_new_enough() {
    command -v "$1" >/dev/null 2>&1 || return 1
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (int(sys.argv[1]), int(sys.argv[2])) else 1)' \
        "$MIN_MAJOR" "$MIN_MINOR" >/dev/null 2>&1
}

python_version() {
    "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown"
}

# Pick a Python interpreter. An explicit --python (or PYTHON=...) must be new
# enough; otherwise try the newest named interpreters before plain python3,
# so a too-old python3 on PATH does not win.
if [ -n "$PYTHON" ]; then
    if ! python_is_new_enough "$PYTHON"; then
        echo "Error: '$PYTHON' is Python $(python_version "$PYTHON"), but FRIdata needs >= $MIN_MAJOR.$MIN_MINOR." >&2
        exit 1
    fi
else
    for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
        if python_is_new_enough "$candidate"; then
            PYTHON="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: no Python >= $MIN_MAJOR.$MIN_MINOR found on PATH." >&2
        if command -v python3 >/dev/null 2>&1; then
            echo "       'python3' is Python $(python_version python3)." >&2
        fi
        echo "       Install a newer Python, or pass one with --python /path/to/python3.13." >&2
        exit 1
    fi
fi

echo "Using $PYTHON (Python $(python_version "$PYTHON"))."

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
