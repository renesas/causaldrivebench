#!/bin/bash
# =====================================================================
# monitor.sh — Snapshot progress of a Gemma4 inference run
#
# All configuration is read from environment variables (or a dev.env
# file). Prints VQA/trajectory progress, throughput, errors, and GPU.
#
# Usage:
#   bash evaluation/models/gemma4/monitor.sh              # auto-find latest run
#   bash evaluation/models/gemma4/monitor.sh --run-dir /path/to/run
#   bash evaluation/models/gemma4/monitor.sh --tail 20    # show last N log lines
#   ENV_FILE=/my/dev.env bash evaluation/models/gemma4/monitor.sh
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Parse CLI args ────────────────────────────────────────────────────────────
RUN_DIR_ARG=""
TAIL_N=15
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)      RUN_DIR_ARG="$2"; shift 2 ;;
        --run-dir=*)    RUN_DIR_ARG="${1#*=}"; shift ;;
        --tail)         TAIL_N="$2"; shift 2 ;;
        --tail=*)       TAIL_N="${1#*=}"; shift ;;
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
LOG_FILE="$RUN_DIR/inference_nohup.log"

# ── Find dataset dir (first subdirectory in run dir) ─────────────────────────
DATASET_DIR=$(find "$RUN_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | head -1)
if [ -z "$DATASET_DIR" ]; then
    echo "ERROR: No dataset directory found in $RUN_DIR" >&2
    exit 1
fi
DATASET_NAME="$(basename "$DATASET_DIR")"

# Try to read TOTAL_SAMPLES from the benchmark manifest
BENCH_DIR=""
if [ -n "${CAUSAL_BENCH_DIR:-}" ]; then
    BENCH_DIR="${BENCH_DIR:-$CAUSAL_BENCH_DIR/$DATASET_NAME}"
    if [ ! -d "$BENCH_DIR" ] && [ -d "$CAUSAL_BENCH_DIR" ]; then
        BENCH_DIR="$CAUSAL_BENCH_DIR"
    fi
fi
TOTAL_SAMPLES="${TOTAL_SAMPLES:-0}"
if [ "$TOTAL_SAMPLES" -eq 0 ] && [ -f "$BENCH_DIR/manifest.json" ]; then
    TOTAL_SAMPLES=$("$PYTHON" -c "import json; m=json.load(open('$BENCH_DIR/manifest.json')); print(m.get('n_samples_with_qa', m.get('n_samples', 0)))" 2>/dev/null || echo 0)
fi

echo "========================================"
echo " Gemma4 ${GEMMA4_MODEL_TAG} — Monitor"
echo " $(date)"
echo "========================================"
echo " Run: $RUN_NAME"
echo " Dir: $RUN_DIR"
echo " DS:  $DATASET_NAME"
echo "========================================"
echo ""

# ── Process status ────────────────────────────────────────────────────────────
PID=$(pgrep -f "inference.py.*${RUN_NAME}" 2>/dev/null || true)
[ -n "$PID" ] && echo "Status:  RUNNING (PID $PID)" || echo "Status:  STOPPED"

# ── VQA / traj sample counts ──────────────────────────────────────────────────
DONE_VQA=$(find "$DATASET_DIR" -name "outputs.jsonl" -size +0c 2>/dev/null | wc -l)
DONE_TRAJ=$(find "$DATASET_DIR" -name "trajectory_outputs.jsonl" -size +0c 2>/dev/null | wc -l)

if [ "$TOTAL_SAMPLES" -gt 0 ]; then
    PCT_VQA=$(( DONE_VQA * 100 / TOTAL_SAMPLES ))
    PCT_TRAJ=$(( DONE_TRAJ * 100 / TOTAL_SAMPLES ))
    MILESTONE=$(( (PCT_VQA / 5) * 5 ))
    echo "VQA:     $DONE_VQA / $TOTAL_SAMPLES  (${PCT_VQA}%)  [milestone: ${MILESTONE}%]"
    echo "Traj:    $DONE_TRAJ / $TOTAL_SAMPLES  (${PCT_TRAJ}%)"
else
    echo "VQA:     $DONE_VQA samples with outputs"
    echo "Traj:    $DONE_TRAJ samples with traj outputs"
fi

# ── Question-level counts ─────────────────────────────────────────────────────
VQA_Q=$(find "$DATASET_DIR" -name "outputs.jsonl" -size +0c 2>/dev/null \
        | xargs grep -c '' 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
TRAJ_Q=$(find "$DATASET_DIR" -name "trajectory_outputs.jsonl" -size +0c 2>/dev/null \
         | xargs grep -c '' 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
echo "VQA q:   $VQA_Q done"
echo "Traj q:  $TRAJ_Q done"

# ── Throughput (last 5 min from timestamps) ───────────────────────────────────
RATE=$("$PYTHON" - <<PYEOF 2>/dev/null
import json, pathlib
from datetime import datetime, timezone, timedelta
ds = pathlib.Path("$DATASET_DIR")
cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
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
    print(f"{len(recent)/span:.3f} q/s ({len(recent)} q in {span:.0f}s)" if span > 0 else "n/a")
else:
    print("n/a (< 2 completions in last 5 min)")
PYEOF
)
echo "Rate:    $RATE (last 5 min)"

# ── Errors ────────────────────────────────────────────────────────────────────
ERR_VQA=$(find "$DATASET_DIR" -name "outputs.jsonl" -size +0c 2>/dev/null \
          | xargs grep -l '"text": "ERROR:' 2>/dev/null | wc -l)
ERR_TRAJ=$(find "$DATASET_DIR" -name "trajectory_outputs.jsonl" -size +0c 2>/dev/null \
           | xargs grep -l '"text": "ERROR:' 2>/dev/null | wc -l)
echo "Errors:  VQA=$ERR_VQA  Traj=$ERR_TRAJ"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "n/a")
echo "GPU:     $GPU"

# ── Last scene from log ───────────────────────────────────────────────────────
if [ -f "$LOG_FILE" ]; then
    LAST=$(grep "\[main\] ──" "$LOG_FILE" 2>/dev/null | tail -1 || true)
    echo "Last:    ${LAST:-n/a}"
else
    echo "Log:     not found ($LOG_FILE)"
fi
echo ""

# ── Log tail ──────────────────────────────────────────────────────────────────
if [ -f "$LOG_FILE" ]; then
    echo "--- Last $TAIL_N log lines ---"
    tail -"$TAIL_N" "$LOG_FILE" 2>/dev/null | grep -v '^\s*$' || true
fi

echo "========================================"
echo "Run dir: $RUN_DIR"
echo "Log:     $LOG_FILE"
