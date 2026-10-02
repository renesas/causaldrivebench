#!/bin/bash
# =====================================================================
# run_postprocess.sh — VQA + trajectory postprocess for a Gemma4 run
#
# All configuration is read from environment variables (or a dev.env
# file). Automatically finds the latest run dir if not specified.
#
# Usage:
#   bash evaluation/models/gemma4/run_postprocess.sh              # auto-find latest run
#   bash evaluation/models/gemma4/run_postprocess.sh --run-dir /path/to/run
#   bash evaluation/models/gemma4/run_postprocess.sh --no-judge   # skip LLM judge
#   ENV_FILE=/my/dev.env bash evaluation/models/gemma4/run_postprocess.sh
#
# Required env:
#   CAUSAL_BENCH_DIR   — path to bench dataset parent (e.g. /data/causal_drive_bench_YYYYMMDD)
#   CDB_OUTPUTS_DIR    — path to outputs root (e.g. /data/cdb_outputs_YYYYMMDD)
#
# Optional env (for LLM judge):
#   ANTHROPIC_API_KEY or provider-specific keys (see eval_config.yaml)
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Parse CLI args ────────────────────────────────────────────────────────────
RUN_DIR_ARG=""
USE_JUDGE=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)      RUN_DIR_ARG="$2"; shift 2 ;;
        --run-dir=*)    RUN_DIR_ARG="${1#*=}"; shift ;;
        --no-judge)     USE_JUDGE=false; shift ;;
        --env-file)     ENV_FILE="$2"; shift 2 ;;
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
    echo "[run_postprocess] Loading env from $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
fi

# ── Required variables ────────────────────────────────────────────────────────
CAUSAL_BENCH_DIR="${CAUSAL_BENCH_DIR:?Set CAUSAL_BENCH_DIR (path to bench dataset parent)}"
CDB_OUTPUTS_DIR="${CDB_OUTPUTS_DIR:?Set CDB_OUTPUTS_DIR (path to outputs root)}"

# ── Optional variables with defaults ─────────────────────────────────────────
GEMMA4_MODEL_TAG="${GEMMA4_MODEL_TAG:-e4b}"
DATASET="${DATASET:-causal_nuscenes}"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-$REPO_ROOT/evaluation/eval_config.yaml}"

# Bench dir: append dataset name if parent directory is given
BENCH_DIR="${BENCH_DIR:-$CAUSAL_BENCH_DIR/$DATASET}"
if [ ! -d "$BENCH_DIR" ] && [ -d "$CAUSAL_BENCH_DIR" ]; then
    BENCH_DIR="$CAUSAL_BENCH_DIR"
fi

# Log dir: next to this script (or override with LOG_DIR env var)
LOG_DIR="${LOG_DIR:-$CDB_OUTPUTS_DIR}"

# ── Resolve run dir ────────────────────────────────────────────────────────────
MODEL_ID="gemma4_${GEMMA4_MODEL_TAG//-/_}"
if [ -n "$RUN_DIR_ARG" ]; then
    RUN_DIR="$RUN_DIR_ARG"
else
    RUN_DIR=$(find "$CDB_OUTPUTS_DIR" -maxdepth 1 -type d -name "${MODEL_ID}_*" 2>/dev/null | sort | tail -1)
    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "[run_postprocess] ERROR: No ${MODEL_ID} run dir found in $CDB_OUTPUTS_DIR" >&2
        exit 1
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POSTPROCESS="$REPO_ROOT/evaluation/models/gemma4/postprocess.py"

echo "========================================"
echo " Gemma4 ${GEMMA4_MODEL_TAG} Postprocess"
echo " Run dir: $RUN_DIR"
echo " Bench:   $BENCH_DIR"
echo " Judge:   $USE_JUDGE"
echo " $(date)"
echo "========================================"

echo ""
echo "--- Step 1: VQA Postprocess$([ "$USE_JUDGE" = true ] && echo " (with LLM judge)" || echo "") ---"
JUDGE_FLAG=""
[ "$USE_JUDGE" = true ] && JUDGE_FLAG="--use-llm-judge"

"$PYTHON" "$POSTPROCESS" \
    --config "$CONFIG" \
    --bench-dir "$BENCH_DIR" \
    --run-dir "$RUN_DIR" \
    --mode full \
    $JUDGE_FLAG \
    2>&1 | tee "$LOG_DIR/postprocess_vqa_${TIMESTAMP}.log"

echo ""
echo "--- Step 2: Trajectory Postprocess ---"
"$PYTHON" "$POSTPROCESS" \
    --config "$CONFIG" \
    --bench-dir "$BENCH_DIR" \
    --run-dir "$RUN_DIR" \
    --mode full \
    --trajectory-eval \
    2>&1 | tee "$LOG_DIR/postprocess_traj_${TIMESTAMP}.log"

echo ""
echo "========================================"
echo " Postprocess complete."
echo " Run dir:  $RUN_DIR"
echo " Logs:     $LOG_DIR/postprocess_*_${TIMESTAMP}.log"
echo "========================================"
