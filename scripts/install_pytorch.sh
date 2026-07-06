#!/bin/bash
# Install a PyTorch build matching the machine's CUDA driver, then install ESM.
#
# Fixes errors such as:
#   UserWarning: CUDA initialization: The NVIDIA driver on your system is too
#   old (found version 12040). ...
# which happen when the installed PyTorch was compiled against a newer CUDA
# toolkit than the installed NVIDIA driver supports.
#
# The script reads the maximum CUDA version the driver supports from
# `nvidia-smi`, selects the highest PyTorch wheel (cuXXX) that is not newer
# than that, and installs it into the *currently active* environment via pip.
# If no NVIDIA GPU/driver is found it installs the CPU-only build.
#
# Usage: ./install_pytorch.sh [--cpu] [--dry-run]
#   --cpu      Force the CPU-only PyTorch build (skip GPU detection)
#   --dry-run  Print what would be installed without installing anything

set -euo pipefail

# PyTorch wheel CUDA versions, ascending. Update as new wheels are released.
# https://pytorch.org/get-started/locally/ and https://download.pytorch.org/whl/
SUPPORTED_CUDA=(11.8 12.1 12.4 12.6 12.8)

CPU_ONLY=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --cpu)     CPU_ONLY=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: ./install_pytorch.sh [--cpu] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# --- Verify an environment is active ------------------------------------------
if ! command -v python >/dev/null 2>&1; then
    echo "Error: no 'python' found on PATH. Activate your conda/venv first." >&2
    exit 1
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Target environment: $CONDA_PREFIX"
    if [ "$(basename "$CONDA_PREFIX")" = "base" ]; then
        echo "Warning: the conda 'base' environment is active. It is strongly" >&2
        echo "         recommended to activate a dedicated environment first." >&2
    fi
else
    echo "Warning: CONDA_PREFIX is not set; installing into '$(command -v python)'." >&2
fi

# Turn a X.Y version into a comparable integer, e.g. 12.4 -> 1204.
cuda_to_int() {
    local major minor
    major="${1%%.*}"
    minor="${1#*.}"
    printf '%d' "$((major * 100 + minor))"
}

# --- Detect the driver's maximum supported CUDA version -----------------------
CUDA_TAG=""
if [ "$CPU_ONLY" = true ]; then
    echo "Forcing CPU-only PyTorch (--cpu)."
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No 'nvidia-smi' found; assuming no NVIDIA GPU -> CPU-only PyTorch."
    CPU_ONLY=true
else
    # The 'CUDA Version' field in nvidia-smi reports the highest CUDA runtime
    # the installed driver can support.
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null \
        | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' \
        | grep -oE '[0-9]+\.[0-9]+' \
        | head -n1 || true)"

    if [ -z "$DRIVER_CUDA" ]; then
        echo "Could not read a CUDA version from nvidia-smi; falling back to CPU-only." >&2
        CPU_ONLY=true
    else
        echo "Driver supports up to CUDA $DRIVER_CUDA."
        driver_int="$(cuda_to_int "$DRIVER_CUDA")"

        # Pick the highest supported wheel that is <= the driver's max CUDA.
        chosen=""
        for v in "${SUPPORTED_CUDA[@]}"; do
            if [ "$(cuda_to_int "$v")" -le "$driver_int" ]; then
                chosen="$v"
            fi
        done

        if [ -z "$chosen" ]; then
            echo "Driver CUDA $DRIVER_CUDA is older than the oldest available" >&2
            echo "PyTorch CUDA wheel (${SUPPORTED_CUDA[0]}); falling back to CPU-only." >&2
            CPU_ONLY=true
        else
            CUDA_TAG="cu${chosen//./}"
            echo "Selected PyTorch CUDA build: $CUDA_TAG (for CUDA $chosen)."
        fi
    fi
fi

# --- Build and run the install command ----------------------------------------
if [ "$CPU_ONLY" = true ]; then
    INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
fi

PIP_CMD=(python -m pip install torch torchvision torchaudio --index-url "$INDEX_URL")

echo
echo "Will run: ${PIP_CMD[*]}"
echo "Then:     python -m pip install esm"

if [ "$DRY_RUN" = true ]; then
    echo "(--dry-run: nothing installed)"
    exit 0
fi

echo
echo "Installing PyTorch..."
"${PIP_CMD[@]}"

# ESM requires PyTorch to be installed first.
echo "Installing ESM..."
python -m pip install esm

# --- Verify the installation --------------------------------------------------
echo "Verifying PyTorch installation..."
python - <<'PY'
import torch
print(f"torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA available: True (torch built for CUDA {torch.version.cuda})")
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA available: False (CPU-only build or no usable GPU)")
PY

echo "Installation complete!"
