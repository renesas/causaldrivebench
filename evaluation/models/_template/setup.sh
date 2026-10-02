#!/bin/bash
# =====================================================================
# MODEL SETUP SCRIPT — replace placeholders for your model
# =====================================================================
set -euo pipefail

MODEL_DIR="/workspace/model"
WEIGHTS_DIR="/workspace/weights"

echo "=====================================================================+"
echo "  Setting up model in $MODEL_DIR"
echo "  Weights will be saved to $WEIGHTS_DIR"
echo "====================================================================="

mkdir -p "$MODEL_DIR" "$WEIGHTS_DIR"

# Step 1: Clone model repository
# Replace <MODEL_REPO_URL> with the actual GitHub URL for your model.
git clone --depth 1 <MODEL_REPO_URL> "$MODEL_DIR"
cd "$MODEL_DIR"

# Step 2 (optional): Install additional dependencies from within the repo
# pip install -e .

# Step 3: Download model weights — choose ONE of the options below:

# ---- Option A — HuggingFace Hub ----------------------------------------
# Requires: huggingface_hub (included in requirements.in)
# Set HF_TOKEN env var for gated models.
#
# huggingface-cli login --token "${HF_TOKEN:-}"
# huggingface-cli download <ORG/REPO> --local-dir "$WEIGHTS_DIR"

# ---- Option B — Google Drive (via gdown) --------------------------------
# Requires: pip install gdown
#
# pip install gdown
# gdown <GDRIVE_FILE_ID> -O "$WEIGHTS_DIR/weights.pt"

# ---- Option C — AWS S3 --------------------------------------------------
# Requires: awscli
#
# aws s3 cp s3://<BUCKET>/<KEY> "$WEIGHTS_DIR/"

# ---- Option D — Direct URL download ------------------------------------
# wget -q -O "$WEIGHTS_DIR/weights.bin" https://<HOST>/<PATH>/weights.bin

echo "====================================================================="
echo "  Setup complete."
echo "====================================================================="
