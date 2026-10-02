#!/bin/bash
#
# Standalone graph verification pipeline using Claude Haiku.
#
# For each scene that has a *_pruned.json in the graphs/outputs directory,
# sends T=0 images + 4 temporal BEVs + condensed causal graph to Haiku and
# asks it to verify node classifications and edge justifications.
#
# Output (flat in graphs dir, alongside *_pruned.json):
#   {graphs_dir}/{scene_id}_verification.json   — per scene
#
# Usage:
#   ./verify_graphs.sh nuscenes
#   ./verify_graphs.sh nuscenes --graphs /path/to/outputs/nuscenes_full
#   ./verify_graphs.sh nuscenes --max-samples 5
#   ./verify_graphs.sh nuscenes --dry-run
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATASET=""
GRAPHS_DIR_OVERRIDE=""
RECORDS_DIR_OVERRIDE=""
BATCH_DIR_OVERRIDE=""
MAX_SAMPLES_ARG=""
DRY_RUN=false
FORCE=""
PROVIDER=""

# ─── Helpers ──────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 DATASET [OPTIONS]"
    echo ""
    echo "DATASET: nuscenes | openscene | argoverse2"
    echo ""
    echo "OPTIONS:"
    echo "  --graphs DIR       Outputs/graphs directory (default: from config.yaml)"
    echo "                     Verification JSONs are written here alongside *_pruned.json"
    echo "  --records DIR      Records directory (default: from config.yaml)"
    echo "  --batch-dir DIR    Batch JSONL directory (default: from config.yaml)"
    echo "  --provider NAME    LLM provider: openai | claude | gemini (default: from config.yaml)"
    echo "  --max-samples N     Limit to first N samples"
    echo "  --dry-run          Prepare batch JSONL but do not submit"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 nuscenes"
    echo "  $0 nuscenes --graphs /spiral_hdd_3/workspace/navvrat/Neurips/outputs/nuscenes_full"
    echo "  $0 nuscenes --max-samples 5 --dry-run"
    exit 0
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "ERROR: $1" >&2
    exit 1
}

# ─── Read config values ───────────────────────────────────────────────────────
_cfg_path() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', {}).get('$2', ''))
" 2>/dev/null
}

_cfg_top() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', ''))
" 2>/dev/null
}

_cfg_batch() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('batch', {}).get('$1', ''))
" 2>/dev/null
}

# ─── Parse arguments ──────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    usage
fi

DATASET=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --graphs)      GRAPHS_DIR_OVERRIDE=$2;  shift 2 ;;
        --records)     RECORDS_DIR_OVERRIDE=$2; shift 2 ;;
        --batch-dir)   BATCH_DIR_OVERRIDE=$2;   shift 2 ;;
        --provider)    PROVIDER=$2;             shift 2 ;;
        --max-samples)  MAX_SAMPLES_ARG="--max-samples $2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --force)       FORCE="--force"; shift ;;
        -h|--help)     usage ;;
        *) error "Unknown option: $1" ;;
    esac
done

if [[ ! "$DATASET" =~ ^(nuscenes|openscene|argoverse2)$ ]]; then
    error "Invalid dataset: $DATASET"
fi

[[ -z "$PROVIDER" ]] && PROVIDER="$(_cfg_top provider)"

# ─── Resolve paths ────────────────────────────────────────────────────────────
RECORDS_DIR="${RECORDS_DIR_OVERRIDE:-$(_cfg_path "$DATASET" records)}"
GRAPHS_DIR="${GRAPHS_DIR_OVERRIDE:-$(_cfg_path "$DATASET" outputs)}"

if [[ -n "$BATCH_DIR_OVERRIDE" ]]; then
    BATCH_DIR="$BATCH_DIR_OVERRIDE"
    RESULTS_DIR="$BATCH_DIR_OVERRIDE/batch_results"
else
    BATCH_DIR="$(_cfg_batch batch_dir)"
    RESULTS_DIR="$(_cfg_batch results_dir)"
fi

[[ -z "$RECORDS_DIR" ]] && error "Records dir not set — use --records or set config.yaml ${DATASET}.records"
[[ -z "$GRAPHS_DIR"  ]] && error "Graphs dir not set  — use --graphs  or set config.yaml ${DATASET}.outputs"
[[ -z "$BATCH_DIR"   ]] && BATCH_DIR="batch_jobs"
[[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="batch_results"

mkdir -p "$BATCH_DIR" "$RESULTS_DIR"

# ─── Submit + wait + download + process (with chunking) ───────────────────────
wait_for_batch() {
    local batch_id=$1
    local label=$2
    log "Monitoring batch $batch_id ($label)..."
    while true; do
        status=$(python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} status \
            --batch-id "$batch_id" 2>&1 | grep "Status:" | awk '{print $2}')
        case "$status" in
            completed)
                log "Batch $batch_id completed."
                return 0
                ;;
            failed|cancelled|expired)
                log "WARNING: Batch $batch_id $status"
                return 1
                ;;
            *)
                log "  Status: $status — waiting 60s..."
                sleep 60
                ;;
        esac
    done
}

submit_and_process_chunked() {
    local jsonl_file=$1
    local label=$2

    local max_bytes=$((95 * 1024 * 1024))
    local file_size
    file_size=$(stat -c%s "$jsonl_file")

    if [[ $file_size -eq 0 ]]; then
        log "No requests to submit ($label) — all already done"
        return 0
    fi

    if [[ $file_size -le $max_bytes ]]; then
        local batch_id
        batch_id=$(python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} submit \
            --file "$jsonl_file" --description "$label" \
            | grep "Batch ID:" | awk '{print $3}')
        [[ -z "$batch_id" ]] && error "Failed to submit $label"
        log "$label submitted: $batch_id"
        wait_for_batch "$batch_id" "$label"
        python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} download \
            --batch-id "$batch_id" --output "$RESULTS_DIR"
        python "$SCRIPT_DIR/process_results.py" \
            --file "$RESULTS_DIR/${batch_id}_results.jsonl" \
            --stage verify \
            --output "$GRAPHS_DIR" \
            ${PROVIDER:+--provider $PROVIDER}
    else
        local chunk_dir
        chunk_dir=$(dirname "$jsonl_file")/chunks_$(basename "$jsonl_file" .jsonl)
        rm -rf "$chunk_dir"
        mkdir -p "$chunk_dir"
        python - "$jsonl_file" "$chunk_dir" "$max_bytes" <<'PYEOF'
import sys, os
src, dst, max_b = sys.argv[1], sys.argv[2], int(sys.argv[3])
chunk, size, idx = [], 0, 0
with open(src) as f:
    for line in f:
        lb = len(line.encode())
        if size + lb > max_b and chunk:
            with open(f"{dst}/chunk_{idx:03d}.jsonl", "w") as o:
                o.writelines(chunk)
            chunk, size, idx = [], 0, idx + 1
        chunk.append(line); size += lb
if chunk:
    with open(f"{dst}/chunk_{idx:03d}.jsonl", "w") as o:
        o.writelines(chunk)
print(f"Split into {idx+1} chunks")
PYEOF
        local chunk_num=0
        local batch_ids=()
        for chunk_file in "$chunk_dir"/chunk_*.jsonl; do
            chunk_num=$((chunk_num + 1))
            local batch_id
            batch_id=$(python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} submit \
                --file "$chunk_file" --description "$label chunk $chunk_num" \
                | grep "Batch ID:" | awk '{print $3}')
            [[ -z "$batch_id" ]] && error "Failed to submit $label chunk $chunk_num"
            log "$label chunk $chunk_num submitted: $batch_id"
            batch_ids+=("$batch_id")
        done

        log "All $chunk_num chunks submitted — waiting..."
        local pids=()
        for batch_id in "${batch_ids[@]}"; do
            wait_for_batch "$batch_id" "$label" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done

        for batch_id in "${batch_ids[@]}"; do
            python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} download \
                --batch-id "$batch_id" --output "$RESULTS_DIR" || continue
            python "$SCRIPT_DIR/process_results.py" \
                --file "$RESULTS_DIR/${batch_id}_results.jsonl" \
                --stage verify \
                --output "$GRAPHS_DIR" \
                ${PROVIDER:+--provider $PROVIDER}
        done
    fi
}

# ─── Run ──────────────────────────────────────────────────────────────────────
log "Dataset:    $DATASET"
log "Provider:   ${PROVIDER:-openai}"
log "Records:    $RECORDS_DIR"
log "Graphs:     $GRAPHS_DIR"
log "Batch dir:  $BATCH_DIR"
[[ "$DRY_RUN" == "true"    ]] && log "Mode:       dry-run (no submission)"
[[ -n "$MAX_SAMPLES_ARG"    ]] && log "Max samples: $(echo $MAX_SAMPLES_ARG | awk '{print $2}')"

log "Preparing batch requests..."
python "$SCRIPT_DIR/prepare_batches.py" \
    --stage verify \
    --dataset "$DATASET" \
    --records "$RECORDS_DIR" \
    --graphs  "$GRAPHS_DIR" \
    --output  "$BATCH_DIR" \
    --model   "claude-haiku-4-5-20251001" \
    ${PROVIDER:+--provider $PROVIDER} \
    $MAX_SAMPLES_ARG \
    $FORCE

BATCH_FILE="$BATCH_DIR/${DATASET}_stage_verify.jsonl"

if [[ "$DRY_RUN" == "true" ]]; then
    log "Dry run — JSONL ready at $BATCH_FILE (not submitted)"
    exit 0
fi

if [[ ! -f "$BATCH_FILE" ]]; then
    log "WARNING: No batch file created — all scenes already verified or no pruned graphs"
    exit 0
fi

submit_and_process_chunked "$BATCH_FILE" "Graph verification: $DATASET"

log "Done. Results: $GRAPHS_DIR/*_verification.json"
