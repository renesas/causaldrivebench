#!/bin/bash
# =====================================================================
# milestone_watch.sh — Background 5% milestone watcher for Gemma4
#
# Logs each 5% VQA completion milestone with rate, GPU, and error counts.
# Run in the background via nohup or disown.
#
# Usage:
#   nohup bash evaluation/models/gemma4/milestone_watch.sh >> milestone_watch.log 2>&1 &
#   bash evaluation/models/gemma4/milestone_watch.sh --run-dir /path/to/run
#   ENV_FILE=/my/dev.env bash evaluation/models/gemma4/milestone_watch.sh
#
# Required env:
#   CDB_OUTPUTS_DIR    — root outputs directory
#   TOTAL_SAMPLES      — expected number of samples (or set CAUSAL_BENCH_DIR to auto-detect)
#
# Optional env:
#   GEMMA4_MODEL_TAG   — default: e4b
#   CHECK_INTERVAL     — seconds between checks (default: 120)
#   PYTHON             — python interpreter (default: python3)
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse CLI args ────────────────────────────────────────────────────────────
RUN_DIR_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)      RUN_DIR_ARG="$2"; shift 2 ;;
        --run-dir=*)    RUN_DIR_ARG="${1#*=}"; shift ;;
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
    set -a; source "$ENV_FILE"; set +a
fi

# ── Required variables ────────────────────────────────────────────────────────
CDB_OUTPUTS_DIR="${CDB_OUTPUTS_DIR:?Set CDB_OUTPUTS_DIR (path to outputs root)}"

# ── Optional variables with defaults ─────────────────────────────────────────
GEMMA4_MODEL_TAG="${GEMMA4_MODEL_TAG:-e4b}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
PYTHON="${PYTHON:-python3}"

# ── Resolve run dir ────────────────────────────────────────────────────────────
MODEL_ID="gemma4_${GEMMA4_MODEL_TAG//-/_}"
if [ -n "$RUN_DIR_ARG" ]; then
    RUN_DIR="$RUN_DIR_ARG"
else
    RUN_DIR=$(find "$CDB_OUTPUTS_DIR" -maxdepth 1 -type d -name "${MODEL_ID}_*" 2>/dev/null | sort | tail -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: No run dir found in $CDB_OUTPUTS_DIR for model $MODEL_ID" >&2
        exit 1
    fi
fi

RUN_NAME="$(basename "$RUN_DIR")"

# ── Find dataset dir ──────────────────────────────────────────────────────────
DATASET_DIR=$(find "$RUN_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | head -1)
if [ -z "$DATASET_DIR" ]; then
    echo "ERROR: No dataset directory found in $RUN_DIR" >&2
    exit 1
fi
DATASET_NAME="$(basename "$DATASET_DIR")"

# ── Resolve TOTAL_SAMPLES ─────────────────────────────────────────────────────
TOTAL="${TOTAL_SAMPLES:-0}"
if [ "$TOTAL" -eq 0 ] && [ -n "${CAUSAL_BENCH_DIR:-}" ]; then
    BENCH_DATASET="${CAUSAL_BENCH_DIR}/${DATASET_NAME}"
    [ ! -d "$BENCH_DATASET" ] && BENCH_DATASET="$CAUSAL_BENCH_DIR"
    if [ -f "$BENCH_DATASET/manifest.json" ]; then
        TOTAL=$("$PYTHON" -c "import json; m=json.load(open('$BENCH_DATASET/manifest.json')); print(m.get('n_samples_with_qa', m.get('n_samples', 0)))" 2>/dev/null || echo 0)
    fi
fi
if [ "$TOTAL" -eq 0 ]; then
    echo "WARNING: TOTAL_SAMPLES=0 — set TOTAL_SAMPLES or CAUSAL_BENCH_DIR for percentage reporting" >&2
    TOTAL=1  # avoid division by zero; percentages will be meaningless
fi

echo "[milestone_watch] Started at $(date)"
echo "[milestone_watch] Run dir:   $RUN_DIR"
echo "[milestone_watch] Dataset:   $DATASET_NAME"
echo "[milestone_watch] Total:     $TOTAL samples"
echo "[milestone_watch] Interval:  ${CHECK_INTERVAL}s"

LAST_MILESTONE=0

while true; do
    PID=$(pgrep -f "inference.py.*${RUN_NAME}" 2>/dev/null || true)

    DONE_VQA=$(find "$DATASET_DIR" -name "outputs.jsonl" -size +0c 2>/dev/null | wc -l)
    DONE_TRAJ=$(find "$DATASET_DIR" -name "trajectory_outputs.jsonl" -size +0c 2>/dev/null | wc -l)
    PCT=$(( DONE_VQA * 100 / TOTAL ))
    MILESTONE=$(( (PCT / 5) * 5 ))

    if [ "$MILESTONE" -gt "$LAST_MILESTONE" ]; then
        RATE=$("$PYTHON" - <<PYEOF 2>/dev/null
import json, pathlib
from datetime import datetime, timezone, timedelta
ds = pathlib.Path("$DATASET_DIR")
cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
recent = []
for f in ds.rglob("outputs.jsonl"):
    try:
        for line in open(f):
            rec = json.loads(line.strip()) if line.strip() else None
            if rec:
                ts = rec.get("timestamp","")
                if ts:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if dt >= cutoff:
                        recent.append(dt)
    except: pass
if len(recent) > 1:
    recent.sort()
    span = (recent[-1]-recent[0]).total_seconds()
    print(f"{len(recent)/span:.3f}" if span > 0 else "n/a")
else:
    print("n/a")
PYEOF
        )
        GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
        ERR=$(find "$DATASET_DIR" -name "outputs.jsonl" -size +0c 2>/dev/null | xargs grep -l '"text": "ERROR:' 2>/dev/null | wc -l)
        echo ""
        echo "=== MILESTONE ${MILESTONE}% reached at $(date) ==="
        echo "    VQA:  $DONE_VQA / $TOTAL samples"
        echo "    Traj: $DONE_TRAJ / $TOTAL samples"
        echo "    Rate: ${RATE} q/s (last 10 min)"
        echo "    GPU:  $GPU"
        echo "    Errors: $ERR samples with VQA errors"
        LAST_MILESTONE=$MILESTONE
    fi

    if [ "$DONE_VQA" -ge "$TOTAL" ]; then
        echo ""
        echo "=== 100% VQA COMPLETE at $(date) ==="
        echo "    Traj: $DONE_TRAJ / $TOTAL"
        echo "    Run postprocess: bash $SCRIPT_DIR/run_postprocess.sh"
        exit 0
    fi

    if [ -z "$PID" ] && [ "$DONE_VQA" -lt "$TOTAL" ]; then
        echo "[$(date)] WARNING: inference process not running! VQA=$DONE_VQA/$TOTAL"
    fi

    sleep "$CHECK_INTERVAL"
done
