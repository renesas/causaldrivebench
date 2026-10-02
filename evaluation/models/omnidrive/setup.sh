#!/usr/bin/env bash
# setup.sh — prepare the OmniDrive evaluation environment on the host.
#
# Prerequisites:
#   - Miniconda at /root/miniconda3  (run /workspace/setup_omnidrive.sh first)
#   - OmniDrive repo at /workspace/OmniDrive (cloned from NVlabs/OmniDrive)
#   - Model checkpoint at $MODEL_CHECKPOINT (see dev.env.example)
#   - TinyLlama weights at $OMNIDRIVE_ROOT/ckpts/tiny_llama/
#
# Usage:
#   cp dev.env.example dev.env && $EDITOR dev.env
#   bash setup.sh          # verify env and paths
#   bash run_inference.sh  # run inference (see commands.md)

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-/root/miniconda3}"
ENV_NAME="omnidrive"
OMNIDRIVE_ROOT="${OMNIDRIVE_ROOT:-/workspace/OmniDrive}"
PYTHON="${CONDA_DIR}/envs/${ENV_NAME}/bin/python"

echo "=== OmniDrive evaluation setup check ==="

# --- Conda env ---
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: conda env '${ENV_NAME}' not found at ${PYTHON}"
    echo "  Run: bash /workspace/setup_omnidrive.sh"
    exit 1
fi
echo "[OK] Python: $($PYTHON --version 2>&1) at ${PYTHON}"

# --- OmniDrive repo ---
if [ ! -d "$OMNIDRIVE_ROOT/projects" ]; then
    echo "ERROR: OmniDrive repo not found at ${OMNIDRIVE_ROOT}"
    echo "  Clone from: https://github.com/NVlabs/OmniDrive"
    exit 1
fi
echo "[OK] OmniDrive repo: ${OMNIDRIVE_ROOT}"

# --- Model config ---
MODEL_CONFIG="${MODEL_CONFIG:-${OMNIDRIVE_ROOT}/projects/configs/OmniDrive/eva_base_tinyllama.py}"
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "ERROR: model config not found: ${MODEL_CONFIG}"
    exit 1
fi
echo "[OK] Model config: ${MODEL_CONFIG}"

# --- Model checkpoint ---
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${OMNIDRIVE_ROOT}/work_dirs/omnidrive.pth}"
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "WARN: checkpoint not found: ${MODEL_CHECKPOINT}"
    echo "  Download from: https://huggingface.co/exiawsh/OmniDrive"
else
    echo "[OK] Checkpoint: ${MODEL_CHECKPOINT}"
fi

# --- TinyLlama weights ---
TINYLLAMA_DIR="${OMNIDRIVE_ROOT}/ckpts/tiny_llama"
if [ ! -f "${TINYLLAMA_DIR}/config.json" ]; then
    echo "WARN: TinyLlama weights not found at ${TINYLLAMA_DIR}"
    echo "  Download with:"
    echo "    huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\"
    echo "      --local-dir ${TINYLLAMA_DIR}"
else
    echo "[OK] TinyLlama: ${TINYLLAMA_DIR}"
fi

echo ""
echo "=== To run inference ==="
echo "  source dev.env  # or export vars manually"
echo "  ${PYTHON} inference.py"
echo ""
echo "=== Quick one-liner ==="
echo "  env \$(grep -v '^#' dev.env | xargs) ${PYTHON} inference.py"
