#!/bin/bash
# =====================================================================
# setup.sh — Build and start the Gemma4 llama.cpp inference server
#
# This script replaces the Docker model runner daemon (which caps
# n_parallel ≈ 4) with a directly-configurable llama.cpp instance for
# high-throughput full-dataset evaluation.
#
# What it does:
#   1. Builds the llamacpp-server:gemma4 Docker image (idempotent)
#   2. Verifies GGUF weights exist at MODEL_DIR
#   3. Starts the server with --parallel PARALLEL_SLOTS
#
# GPU memory at default settings (Gemma4 E4B, A100 80 GB):
#   Model weights:    3002 MiB
#   Non-SWA KV:      10240 MiB  (4 global-attn layers × 64 seqs × 10240 ctx)
#   SWA KV:           2560 MiB  (20 sliding-window layers × 64 seqs × 1024 window)
#   CLIP projector:    944 MiB
#   Compute buffer:    570 MiB
#   Total:           ~18 GB      ~63 GB free on A100 80 GB
#
# Scaling PARALLEL_SLOTS:
#   --parallel 64  → ~18 GB (default)
#   --parallel 128 → ~31 GB
#   --parallel 256 → ~56 GB
#   (Gemma4 SWA arch is VRAM-efficient vs. dense-attention models)
#
# Usage:
#   bash setup.sh                       # start with defaults (--parallel 64)
#   PARALLEL_SLOTS=128 bash setup.sh    # override parallelism
#   MODEL_DIR=/other/path bash setup.sh # override weights directory
#
#   docker stop llamacpp-gemma4         # stop
#   docker logs -f llamacpp-gemma4      # follow logs
#   curl http://localhost:8080/v1/models # verify server
#
# After starting, run inference with:
#   GEMMA4_API_BASE_URL=http://localhost:8080/v1
#   GEMMA4_API_MODEL_NAME=model.gguf      (llama.cpp reports the filename)
#   CONCURRENT_REQUESTS=<PARALLEL_SLOTS>  (must match --parallel N)
#
# Configuration (override via env or source from dev.env):
#   MODEL_DIR       — host dir with GGUF + mmproj files
#   GGUF_FILE       — GGUF filename inside MODEL_DIR
#   MMPROJ_FILE     — vision projector filename inside MODEL_DIR
#   PARALLEL_SLOTS  — concurrent inference slots (default: 64)
#   CTX_PER_SLOT    — context tokens per slot (default: 10240)
#   PORT            — host port to expose (default: 8080)
# =====================================================================

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/datadrive/models/gemma4}"
GGUF_FILE="${GGUF_FILE:-gemma-4-e4b-it-Q4_K_M.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-gemma-4-e4b-it.mmproj}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-64}"
CTX_PER_SLOT="${CTX_PER_SLOT:-10240}"
PORT="${PORT:-8080}"

IMAGE="llamacpp-server:gemma4"
CONTAINER_NAME="llamacpp-gemma4"
CTX_SIZE=$(( PARALLEL_SLOTS * CTX_PER_SLOT ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 1. Build image (skip if already present) --------------------------------
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[setup] Image $IMAGE already present — skipping build"
else
    echo "[setup] Building $IMAGE from server/Dockerfile..."
    docker build -t "$IMAGE" "$SCRIPT_DIR/server/"
    echo "[setup] Image built."
fi

# ---- 2. Verify weights -------------------------------------------------------
if [ ! -f "$MODEL_DIR/$GGUF_FILE" ]; then
    echo ""
    echo "ERROR: GGUF not found: $MODEL_DIR/$GGUF_FILE"
    echo ""
    echo "  Option A — Extract from an existing Docker model runner volume:"
    echo "    VOL=\$(docker volume ls -q | grep model-runner | head -1)"
    echo "    mkdir -p $MODEL_DIR"
    echo "    docker run --rm -v \"\$VOL:/src:ro\" -v \"$MODEL_DIR:/dst\" alpine \\"
    echo "      sh -c 'find /src -name \"*.gguf\" | head -1 | xargs -I{} cp {} /dst/$GGUF_FILE'"
    echo ""
    echo "  Option B — Download from HuggingFace (requires huggingface_hub):"
    echo "    pip install huggingface_hub"
    echo "    huggingface-cli download unsloth/gemma-4-e4b-it-GGUF \\"
    echo "      gemma-4-e4b-it-Q4_K_M.gguf --local-dir $MODEL_DIR"
    echo ""
    exit 1
fi

if [ ! -f "$MODEL_DIR/$MMPROJ_FILE" ]; then
    echo ""
    echo "ERROR: Mmproj not found: $MODEL_DIR/$MMPROJ_FILE"
    echo ""
    echo "  Download from HuggingFace:"
    echo "    huggingface-cli download unsloth/gemma-4-e4b-it-GGUF \\"
    echo "      mmproj-gemma-4-e4b-it-f16.gguf --local-dir $MODEL_DIR"
    echo "    mv $MODEL_DIR/mmproj-gemma-4-e4b-it-f16.gguf $MODEL_DIR/$MMPROJ_FILE"
    echo ""
    exit 1
fi

# ---- 3. Start server (idempotent) --------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    STATUS=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME")
    if [ "$STATUS" = "running" ]; then
        echo "[setup] Server already running (container: $CONTAINER_NAME, port: $PORT)"
        docker ps --filter "name=${CONTAINER_NAME}" --format "  ID: {{.ID}}  Status: {{.Status}}"
        echo ""
        echo "  API endpoint:      http://localhost:${PORT}/v1"
        echo "  Model name:        model.gguf"
        echo "  CONCURRENT_REQUESTS=${PARALLEL_SLOTS}"
        exit 0
    fi
    echo "[setup] Removing stopped container $CONTAINER_NAME..."
    docker rm "$CONTAINER_NAME"
fi

echo "[setup] Starting llama.cpp server..."
echo "  Container:  $CONTAINER_NAME"
echo "  Image:      $IMAGE"
echo "  Model:      $MODEL_DIR/$GGUF_FILE"
echo "  Mmproj:     $MODEL_DIR/$MMPROJ_FILE"
echo "  Slots:      $PARALLEL_SLOTS × $CTX_PER_SLOT tokens/slot = ctx $CTX_SIZE"
echo "  Port:       $PORT"
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    -v "${MODEL_DIR}:/models:ro" \
    -p "${PORT}:${PORT}" \
    "$IMAGE" \
    -m "/models/${GGUF_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --parallel "$PARALLEL_SLOTS" \
    --cont-batching \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 \
    -fa on

echo ""
echo "Container started. Waiting for model load (~40s)..."
for i in $(seq 1 12); do
    sleep 5
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo ""
        echo "[setup] Server ready."
        echo "  API endpoint:      http://localhost:${PORT}/v1"
        echo "  Model name:        model.gguf"
        echo "  CONCURRENT_REQUESTS=${PARALLEL_SLOTS}"
        echo ""
        echo "  Follow logs:  docker logs -f $CONTAINER_NAME"
        echo "  Stop:         docker stop $CONTAINER_NAME"
        exit 0
    fi
    printf "  ...%ds elapsed\n" "$((i * 5))"
done
echo ""
echo "[setup] Server may still be loading (large model). Check:"
echo "  docker logs -f $CONTAINER_NAME"
echo "  curl http://localhost:${PORT}/health"
