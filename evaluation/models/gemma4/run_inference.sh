#!/bin/bash
# =====================================================================
# run_inference.sh — Launch or resume Gemma4 full-dataset inference
#
# All configuration is read from environment variables (or a dev.env
# file).  No hardcoded paths — copy this script to any eval session
# directory and point it at the right dev.env.
#
# Usage:
#   bash evaluation/models/gemma4/run_inference.sh              # fresh run
#   bash evaluation/models/gemma4/run_inference.sh --resume     # resume latest run in CDB_OUTPUTS_DIR
#   bash evaluation/models/gemma4/run_inference.sh --run-dir /path/to/run  # resume specific run
#   ENV_FILE=/my/dev.env bash evaluation/models/gemma4/run_inference.sh
#
# Quickstart:
#   cp evaluation/models/gemma4/dev.env.example dev.env   # fill in paths
#   bash evaluation/models/gemma4/setup.sh                # start server
#   bash evaluation/models/gemma4/run_inference.sh        # run inference
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Parse CLI args ────────────────────────────────────────────────────────────
MODE="fresh"
RUN_DIR_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)       MODE="resume";  shift ;;
        --run-dir)      MODE="resume";  RUN_DIR_ARG="$2"; shift 2 ;;
        --run-dir=*)    MODE="resume";  RUN_DIR_ARG="${1#*=}"; shift ;;
        --env-file)     ENV_FILE="$2";  shift 2 ;;
        --env-file=*)   ENV_FILE="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Source env file ────────────────────────────────────────────────────────────
ENV_FILE="${ENV_FILE:-}"
if [ -z "$ENV_FILE" ] && [ -f "$SCRIPT_DIR/dev.env" ]; then
    ENV_FILE="$SCRIPT_DIR/dev.env"
fi
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    echo "[run_inference] Loading env from $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
fi

# ── Required variables (must be set in env or dev.env) ────────────────────────
CAUSAL_BENCH_DIR="${CAUSAL_BENCH_DIR:?Set CAUSAL_BENCH_DIR (path to bench dataset parent, e.g. /data/causal_drive_bench_YYYYMMDD)}"
CDB_OUTPUTS_DIR="${CDB_OUTPUTS_DIR:?Set CDB_OUTPUTS_DIR (path to outputs root, e.g. /data/cdb_outputs_YYYYMMDD)}"

# ── Optional variables with defaults ─────────────────────────────────────────
GEMMA4_MODEL_TAG="${GEMMA4_MODEL_TAG:-e4b}"
GEMMA4_API_BASE_URL="${GEMMA4_API_BASE_URL:-http://localhost:8080/v1}"
GEMMA4_API_MODEL_NAME="${GEMMA4_API_MODEL_NAME:-model.gguf}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-64}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-6}"
TRAJECTORY_EVAL="${TRAJECTORY_EVAL:-true}"
PRELOAD_IMAGES="${PRELOAD_IMAGES:-true}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-64}"
PROMPT_GEN_WORKERS="${PROMPT_GEN_WORKERS:-64}"
DATASET="${DATASET:-causal_nuscenes}"
PYTHON="${PYTHON:-python3}"

# Config: use eval_config.yaml in repo root's evaluation/ dir if not set
CONFIG="${CONFIG:-$REPO_ROOT/evaluation/eval_config.yaml}"

# Bench dir: append dataset name if it's the parent directory
BENCH_DIR="${BENCH_DIR:-$CAUSAL_BENCH_DIR/$DATASET}"
if [ ! -d "$BENCH_DIR" ] && [ -d "$CAUSAL_BENCH_DIR" ]; then
    BENCH_DIR="$CAUSAL_BENCH_DIR"
fi

# ── Resolve run dir ────────────────────────────────────────────────────────────
MODEL_ID="gemma4_${GEMMA4_MODEL_TAG//-/_}"
if [ "$MODE" = "resume" ]; then
    if [ -n "$RUN_DIR_ARG" ]; then
        RUN_DIR="$RUN_DIR_ARG"
    else
        RUN_DIR=$(find "$CDB_OUTPUTS_DIR" -maxdepth 1 -type d -name "${MODEL_ID}_*" 2>/dev/null | sort | tail -1)
        if [ -z "$RUN_DIR" ]; then
            echo "[run_inference] ERROR: No existing run found in $CDB_OUTPUTS_DIR for model $MODEL_ID" >&2
            exit 1
        fi
    fi
    RESUME_FLAG="--run-dir $RUN_DIR"
    echo "[run_inference] Resuming: $RUN_DIR"
else
    RESUME_FLAG=""
    RUN_DIR=""
    echo "[run_inference] Starting fresh run in $CDB_OUTPUTS_DIR"
fi

# Log goes into the run dir (if resuming) or a temp location (if fresh)
if [ -n "$RUN_DIR" ]; then
    LOG="$RUN_DIR/inference_nohup.log"
else
    mkdir -p "$CDB_OUTPUTS_DIR"
    LOG="$CDB_OUTPUTS_DIR/${MODEL_ID}_inference_nohup.log"
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "[run_inference] ──────────────────────────────────────"
echo "  Server:   $GEMMA4_API_BASE_URL  (model: $GEMMA4_API_MODEL_NAME)"
echo "  Bench:    $BENCH_DIR"
echo "  Outputs:  $CDB_OUTPUTS_DIR"
echo "  Slots:    $CONCURRENT_REQUESTS parallel  (SC=$SAMPLE_CONCURRENCY)"
echo "  Traj:     $TRAJECTORY_EVAL"
echo "  Preload:  $PRELOAD_IMAGES"
echo "  Log:      $LOG"
echo "[run_inference] ──────────────────────────────────────"
echo ""

> "$LOG"  # truncate / create

nohup env \
    GEMMA4_API_BASE_URL="$GEMMA4_API_BASE_URL" \
    GEMMA4_API_MODEL_NAME="$GEMMA4_API_MODEL_NAME" \
    GEMMA4_MODEL_TAG="$GEMMA4_MODEL_TAG" \
    MAX_TOKENS="$MAX_TOKENS" \
    IMAGE_SIZE="$IMAGE_SIZE" \
    CONCURRENT_REQUESTS="$CONCURRENT_REQUESTS" \
    SAMPLE_CONCURRENCY="$SAMPLE_CONCURRENCY" \
    TRAJECTORY_EVAL="$TRAJECTORY_EVAL" \
    PRELOAD_IMAGES="$PRELOAD_IMAGES" \
    PRELOAD_WORKERS="$PRELOAD_WORKERS" \
    PROMPT_GEN_WORKERS="$PROMPT_GEN_WORKERS" \
    "$PYTHON" \
        "$REPO_ROOT/evaluation/models/gemma4/inference.py" \
        --config "$CONFIG" \
        --bench-dir "$BENCH_DIR" \
        --output-base "$CDB_OUTPUTS_DIR" \
        --api-base-url "$GEMMA4_API_BASE_URL" \
        --model-name "$GEMMA4_API_MODEL_NAME" \
        --mode full \
        --max-tokens "$MAX_TOKENS" \
        --image-size "$IMAGE_SIZE" \
        ${TRAJECTORY_EVAL:+--trajectory-eval} \
        $RESUME_FLAG \
    >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""
echo "Follow log:  tail -f $LOG"
echo "Monitor:     bash $SCRIPT_DIR/monitor.sh"
echo "Find run:    grep 'Run directory:' $LOG | tail -1"
