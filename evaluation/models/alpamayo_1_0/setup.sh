#!/bin/bash
# =====================================================================
# Alpamayo 1.0 — model setup script
# Clones the repo and downloads model weights from HuggingFace.
#
# This script is called at container build time (or manually).
# Set HF_TOKEN in the environment before running if not already logged in.
# =====================================================================
set -euo pipefail

MODEL_DIR="/workspace/model"
WEIGHTS_DIR="/workspace/weights"
REPO_URL="https://github.com/NVlabs/alpamayo"
MODEL_REPO="nvidia/Alpamayo-R1-10B"

echo "====================================================================="
echo "  Alpamayo 1.0 setup"
echo "  Model dir : $MODEL_DIR"
echo "  Weights   : $WEIGHTS_DIR"
echo "====================================================================="

mkdir -p "$MODEL_DIR" "$WEIGHTS_DIR"

# Step 1: Clone the Alpamayo repository
if [ ! -d "$MODEL_DIR/.git" ]; then
    git clone --depth 1 "$REPO_URL" "$MODEL_DIR"
else
    echo "[setup] Repo already cloned, skipping."
fi

cd "$MODEL_DIR"

# Step 2: Install Python 3.12 and all dependencies via uv
uv python install 3.12
uv sync

# Step 3: Download model weights from HuggingFace (~22 GB)
# Requires access to the gated repo: https://huggingface.co/nvidia/Alpamayo-R1-10B
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[setup] WARNING: HF_TOKEN is not set. Skipping weight download."
    echo "         Set HF_TOKEN and re-run, or let the model auto-download on first inference."
else
    uv run huggingface-cli login --token "$HF_TOKEN"
    uv run huggingface-cli download "$MODEL_REPO" --local-dir "$WEIGHTS_DIR"
    echo "[setup] Weights downloaded to $WEIGHTS_DIR"
fi

echo "====================================================================="
echo "  Setup complete."
echo "  Run inference: uv run python src/alpamayo_r1/test_inference.py"
echo "====================================================================="
