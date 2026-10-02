#!/bin/bash
#
# Run the complete batch processing pipeline for causal graph generation.
#
# This script orchestrates the entire batch pipeline:
#   Stage 1: Generate initial causal graphs (generator)
#   Stage 2: Refine causal graphs (discriminator)
#   Stage 3a: Generate QA Active questions
#   Stage 3b: Generate QA Dormant questions
#   Stage 4: Generate counterfactual trajectories (not yet implemented)
#
# Usage:
#   ./run_batch_pipeline.sh nuscenes
#   ./run_batch_pipeline.sh openscene --max-samples 10
#   ./run_batch_pipeline.sh argoverse2 --resume stage2
#

set -e  # Exit on error

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default values (empty = will be filled from config.yaml after arg parsing)
DATASET=""
MAX_SAMPLES=""
RESUME_STAGE=""
MODEL=""
REASONING=""
GRAPH_PROMPT_VERSION=""
QA_PROMPT_VERSION=""
PROVIDER=""
OUTPUT_DIR=""
RECORDS_DIR_OVERRIDE=""
BATCH_BASE=""
SKIP_DOWNLOAD=false
DRY_RUN=false
USE_NUPLAN=false
FILTER_FILE=""

# ─── Helper Functions ─────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 DATASET [OPTIONS]"
    echo ""
    echo "DATASET:"
    echo "  nuscenes     Process nuScenes dataset"
    echo "  openscene    Process OpenScene dataset"
    echo "  argoverse2   Process Argoverse V2 dataset"
    echo ""
    echo "OPTIONS:"
    echo "  --max-samples N        Process only N samples (default: all)"
    echo "  --resume STAGE        Resume from stage (1, 2, prune, subgraph, 3a, 3b)"
    echo "  --model MODEL         Model to use (default: gpt-5.2)"
    echo "  --reasoning LEVEL     Reasoning level (default: none)"
    echo "  --graph-prompt-version VER  Graph prompt version (default: from config.yaml)"
    echo "  --qa-prompt-version VER    QA prompt version (default: from config.yaml)"
    echo "  --provider PROVIDER   LLM provider: openai, claude, gemini"
    echo "  --output-dir DIR      Override output directory"
    echo "  --records-dir DIR     Override records directory"
    echo "  --batch-dir DIR       Override batch jobs/results base (default: from config.yaml)"
    echo "  --skip-download       Skip downloading results after completion"
    echo "  --dry-run             Prepare batch JSONL files and estimate cost, but do not submit"
    echo "  --filter FILE         Only process scenes listed in a diverse_subset.json manifest"
    echo "  --nuplan              Use nuPlan simulator for agent interventions (Stage 4np) instead of LLM ego trajectory (Stage 5)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 nuscenes"
    echo "  $0 nuscenes --provider claude"
    echo "  $0 openscene --max-samples 10"
    echo "  $0 argoverse2 --resume stage2"
    exit 0
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "ERROR: $1" >&2
    exit 1
}

# Print cost estimate for all JSONL files generated so far (used by --dry-run)
dry_run_cost_estimate() {
    [[ "$DRY_RUN" != "true" ]] && return
    log "=== Dry-run Cost Estimate ==="
    python3 - "$BATCH_DIR" "$DATASET" "$MODEL" <<'COSTEOF'
import sys, json
from pathlib import Path

batch_dir = Path(sys.argv[1])
dataset   = sys.argv[2]
model     = sys.argv[3]

PRICING = {
    "claude-opus-4-6":   (2.50 / 1_000_000, 12.50 / 1_000_000),
    "claude-sonnet-4-6": (1.50 / 1_000_000,  7.50 / 1_000_000),
    "claude-haiku-4-5":  (0.50 / 1_000_000,  2.50 / 1_000_000),
}

STAGE_INFO = {
    "generator":    ("Stage 1  (causal graph)",    2600),
    "discriminator":("Stage 2  (discriminator)",   3000),
    "active":       ("Stage 3a (active QA)",        3000),
    "dormant":      ("Stage 3b (dormant QA)",       2500),
    "distractor":   ("Stage 3c (distractor QA)",    2500),
    "stage4cf":     ("Stage 4cf (CF trajectory)",   2000),
}

in_price = out_price = None
for prefix, (ip, op) in PRICING.items():
    if model.startswith(prefix):
        in_price, out_price = ip, op
        break

grand_input = grand_output = grand_requests = 0

for jsonl in sorted(batch_dir.glob(f"{dataset}_*.jsonl")):
    stem = jsonl.stem
    label = avg_out = None
    for key, (lbl, tok) in STAGE_INFO.items():
        if key in stem:
            label, avg_out = lbl, tok
            break
    if label is None:
        label, avg_out = stem, 3000

    n_req = file_input = 0
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        req  = json.loads(line)
        body = req.get("body", req.get("params", {}))

        n_images = text_chars = 0
        for msg in body.get("messages", []):
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                t = block.get("type", "")
                if t in ("image", "image_url", "input_image"):
                    n_images += 1
                elif t in ("text", "input_text"):
                    text_chars += len(block.get("text", ""))
        sys_prompt = body.get("system", "")
        if isinstance(sys_prompt, list):
            text_chars += sum(len(s.get("text", "")) for s in sys_prompt)
        else:
            text_chars += len(sys_prompt)

        file_input += text_chars // 4 + n_images * 1_000
        n_req += 1

    if n_req == 0:
        continue

    file_output = n_req * avg_out
    if in_price:
        cost = file_input * in_price + file_output * out_price
        cost_str = f"  est. ${cost:.2f}"
    else:
        cost_str = ""

    print(f"  {label}: {n_req} requests, "
          f"~{file_input/1000:.0f}K in / ~{file_output/1000:.0f}K out tokens{cost_str}")

    grand_input    += file_input
    grand_output   += file_output
    grand_requests += n_req

print()
print(f"  TOTAL: {grand_requests} requests")
print(f"  Input:  ~{grand_input/1_000_000:.2f}M tokens")
print(f"  Output: ~{grand_output/1_000_000:.2f}M tokens")
if in_price:
    total_cost = grand_input * in_price + grand_output * out_price
    print(f"  Cost:   ~${total_cost:.2f}  "
          f"(${in_price*1e6:.2f}/MTok in, ${out_price*1e6:.2f}/MTok out)")
else:
    print(f"  Cost:   unknown model '{model}' -- add to PRICING dict")
COSTEOF
    log "Dry-run complete. Review JSONL files in $BATCH_DIR before submitting."
}

wait_for_batch() {
    local batch_id=$1
    local stage=$2

    log "Monitoring batch $batch_id for Stage $stage..."

    while true; do
        status=$(python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} status --batch-id "$batch_id" 2>&1 | grep "Status:" | awk '{print $2}')

        case "$status" in
            completed)
                log "Batch $batch_id completed successfully!"
                return 0
                ;;
            failed|cancelled|expired)
                log "WARNING: Batch $batch_id $status!"
                return 1
                ;;
            *)
                log "  Status: $status — waiting 60s..."
                sleep 60
                ;;
        esac
    done
}

# Split a JSONL file into chunks under MAX_BYTES and submit each.
# Writes all batch IDs to the variable named by $3 (array).
# Usage: submit_chunked <jsonl_file> <stage_label> <results_dir> <output_dir> <process_stage> [extra_process_args...]
submit_and_process_chunked() {
    local jsonl_file=$1
    local stage_label=$2
    local results_dir=$3
    local output_dir=$4
    local process_stage=$5
    shift 5
    local extra_args=("$@")

    local max_bytes=$((95 * 1024 * 1024))  # 95MB hard limit
    local file_size
    file_size=$(stat -c%s "$jsonl_file")

    if [[ $file_size -eq 0 ]]; then
        log "No requests to submit for $stage_label — all already done"
        return 0
    fi

    if [[ $file_size -le $max_bytes ]]; then
        # Single batch — no splitting needed
        local batch_id
        batch_id=$(python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} submit \
            --file "$jsonl_file" \
            --description "$stage_label" \
            | grep "Batch ID:" | awk '{print $3}')
        [[ -z "$batch_id" ]] && error "Failed to submit $stage_label"
        log "$stage_label submitted: $batch_id"
        wait_for_batch "$batch_id" "$stage_label"
        python "$SCRIPT_DIR/submit_batch.py" ${PROVIDER:+--provider $PROVIDER} download \
            --batch-id "$batch_id" --output "$results_dir"
        python "$SCRIPT_DIR/process_results.py" \
            --file "$results_dir/${batch_id}_results.jsonl" \
            --stage "$process_stage" \
            --output "$output_dir" $PROCESS_EXTRA "${extra_args[@]}"
    else
        # Split into chunks
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
        # Submit all chunks in parallel, collect batch IDs
        local chunk_num=0
        local batch_ids=()
        for chunk_file in "$chunk_dir"/chunk_*.jsonl; do
            chunk_num=$((chunk_num + 1))
            local batch_id
            batch_id=$(python "$SCRIPT_DIR/submit_batch.py" submit \
                --file "$chunk_file" \
                --description "$stage_label chunk $chunk_num" \
                | grep "Batch ID:" | awk '{print $3}')
            [[ -z "$batch_id" ]] && error "Failed to submit $stage_label chunk $chunk_num"
            log "$stage_label chunk $chunk_num submitted: $batch_id"
            batch_ids+=("$batch_id")
        done

        log "All $chunk_num chunks submitted — waiting for all to complete in parallel..."

        # Wait for all chunks in parallel
        local pids=()
        for batch_id in "${batch_ids[@]}"; do
            wait_for_batch "$batch_id" "$stage_label" &
            pids+=($!)
        done
        local failed_chunks=0
        for pid in "${pids[@]}"; do
            wait "$pid" || ((failed_chunks++))
        done
        if [[ $failed_chunks -gt 0 ]]; then
            log "WARNING: $failed_chunks chunk(s) failed — processing successful ones"
        fi

        # Download and process all results sequentially (skip failed)
        for batch_id in "${batch_ids[@]}"; do
            local dl_output
            dl_output=$(python "$SCRIPT_DIR/submit_batch.py" download \
                --batch-id "$batch_id" --output "$results_dir" 2>&1) || {
                log "WARNING: download failed for $batch_id — skipping"
                continue
            }
            local result_file="$results_dir/${batch_id}_results.jsonl"
            if [[ -f "$result_file" ]]; then
                python "$SCRIPT_DIR/process_results.py" \
                    --file "$result_file" \
                    --stage "$process_stage" \
                    --output "$output_dir" $PROCESS_EXTRA "${extra_args[@]}"
            else
                log "WARNING: no results file for $batch_id — skipping"
            fi
        done
    fi
}

# ─── Parse Arguments ─────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    usage
fi

DATASET=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --resume)
            RESUME_STAGE=$2
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --reasoning)
            REASONING=$2
            shift 2
            ;;
        --graph-prompt-version)
            GRAPH_PROMPT_VERSION=$2
            shift 2
            ;;
        --qa-prompt-version)
            QA_PROMPT_VERSION=$2
            shift 2
            ;;
        --provider)
            PROVIDER=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --records-dir)
            RECORDS_DIR_OVERRIDE=$2
            shift 2
            ;;
        --batch-dir)
            BATCH_BASE=$2
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --dry-run)
            SKIP_DOWNLOAD=true
            DRY_RUN=true
            shift
            ;;
        --filter)
            FILTER_FILE=$2
            shift 2
            ;;
        --nuplan)
            USE_NUPLAN=true
            shift
            ;;
        --stop-after)
            STOP_AFTER=$2
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate dataset
if [[ ! "$DATASET" =~ ^(nuscenes|openscene|argoverse2)$ ]]; then
    error "Invalid dataset: $DATASET"
fi

# Set paths from config.yaml (CLI overrides applied below)
# Read dataset-level key: _cfg_path DATASET key
_cfg_path() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', {}).get('$2', ''))
" 2>/dev/null
}
# Read top-level key: _cfg_top key
_cfg_top() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', ''))
" 2>/dev/null
}
# Read batch sub-key: _cfg_batch key
_cfg_batch() {
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$SCRIPT_DIR")')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('batch', {}).get('$1', ''))
" 2>/dev/null
}

RECORDS_DIR="$(_cfg_path "$DATASET" records)"
OUTPUTS_DIR="$(_cfg_path "$DATASET" outputs)"

# Fill MODEL, REASONING, PROVIDER, prompt versions from config if not set by CLI
[[ -z "$MODEL" ]]               && MODEL="$(_cfg_top model)"
[[ -z "$REASONING" ]]           && REASONING="$(_cfg_top reasoning)"
[[ -z "$PROVIDER" ]]            && PROVIDER="$(_cfg_top provider)"
[[ -z "$GRAPH_PROMPT_VERSION" ]] && GRAPH_PROMPT_VERSION="$(_cfg_batch graph_prompt_version)"
[[ -z "$QA_PROMPT_VERSION" ]]   && QA_PROMPT_VERSION="$(_cfg_batch qa_prompt_version)"

# Override output/records dir if explicitly provided via CLI
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUTS_DIR="$OUTPUT_DIR"
fi
if [[ -n "$RECORDS_DIR_OVERRIDE" ]]; then
    RECORDS_DIR="$RECORDS_DIR_OVERRIDE"
fi
GRAPHS_DIR="$OUTPUTS_DIR"
QA_DIR="$OUTPUTS_DIR"

# Batch dirs: CLI --batch-dir overrides config
if [[ -n "$BATCH_BASE" ]]; then
    BATCH_DIR="$BATCH_BASE"
    RESULTS_DIR="$BATCH_BASE/batch_results"
else
    BATCH_DIR="$(_cfg_batch batch_dir)"
    RESULTS_DIR="$(_cfg_batch results_dir)"
    # Default results_dir to batch_dir/batch_results if not set in config
    [[ -z "$RESULTS_DIR" ]] && RESULTS_DIR="$BATCH_DIR/batch_results"
fi

# Build extra args for prepare_batches.py
EXTRA_ARGS=""
[[ -n "$PROVIDER" ]] && EXTRA_ARGS="$EXTRA_ARGS --provider $PROVIDER"
EXTRA_ARGS="$EXTRA_ARGS --layout $GRAPH_PROMPT_VERSION"
[[ -n "$FILTER_FILE" ]] && EXTRA_ARGS="$EXTRA_ARGS --filter $FILTER_FILE"

# Provider flag for process_results.py
PROCESS_EXTRA=""
[[ -n "$PROVIDER" ]] && PROCESS_EXTRA="--provider $PROVIDER"

# Create directories
mkdir -p "$BATCH_DIR" "$RESULTS_DIR" "$GRAPHS_DIR" "$QA_DIR"

log "Starting batch pipeline for $DATASET"
log "Provider: ${PROVIDER:-openai}  |  Graph prompts: $GRAPH_PROMPT_VERSION  |  QA prompts: $QA_PROMPT_VERSION"
log "Records: $RECORDS_DIR"
log "Outputs: $OUTPUTS_DIR"

# ─── Stage 1: Generator ──────────────────────────────────────────────────────

if [[ -z "$RESUME_STAGE" ]] || [[ "$RESUME_STAGE" == "1" ]]; then
    log "=== Stage 1: Causal Graph Generation ==="

    # Prepare batch file
    log "Preparing Stage 1 batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 1 \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage1_generator.jsonl"

    if [[ ! -f "$BATCH_FILE" ]]; then
        error "Failed to create Stage 1 batch file"
    fi

    if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
        submit_and_process_chunked "$BATCH_FILE" "Stage 1: ${DATASET} generator" \
            "$RESULTS_DIR" "$GRAPHS_DIR" "1"
    fi

    RESUME_STAGE="prune"
fi

if [[ -n "$STOP_AFTER" && "$STOP_AFTER" == "1" ]]; then
    log "Stopping after Stage 1 (--stop-after 1)"
    log "Raw graphs: $(ls "$GRAPHS_DIR"/*.json 2>/dev/null | wc -l)"
    dry_run_cost_estimate
    exit 0
fi

# ─── Pruning: Remove unreachable nodes ───────────────────────────────────────

if [[ "$RESUME_STAGE" == "prune" ]]; then
    log "=== Pruning: Removing unreachable nodes ==="
    PRUNE_SCRIPT="$(dirname "$SCRIPT_DIR")/causal_graph_extraction/prune_graphs.py"
    python "$PRUNE_SCRIPT" "$GRAPHS_DIR"
    log "Pruned graphs: $(ls "$GRAPHS_DIR"/*_pruned.json 2>/dev/null | wc -l)"
    RESUME_STAGE="subgraph"
fi

if [[ -n "$STOP_AFTER" && "$STOP_AFTER" == "prune" ]]; then
    log "Stopping after pruning (--stop-after prune)"
    dry_run_cost_estimate
    exit 0
fi

_check_stop_after() {
    if [[ -n "$STOP_AFTER" && "$STOP_AFTER" == "$1" ]]; then
        log "Stopping after Stage $1 (--stop-after $1)"
        dry_run_cost_estimate
        exit 0
    fi
}

# ─── Subgraph Extraction: Identify causal structures ─────────────────────────

if [[ "$RESUME_STAGE" == "subgraph" ]]; then
    log "=== Subgraph Extraction: Identifying causal structures ==="
    SUBGRAPH_SCRIPT="$(dirname "$SCRIPT_DIR")/causal_graph_extraction/extract_subgraphs.py"
    python "$SUBGRAPH_SCRIPT" "$GRAPHS_DIR"
    log "Subgraph analyses: $(ls "$GRAPHS_DIR"/*_subgraphs.md 2>/dev/null | wc -l)"
    RESUME_STAGE="3a"
fi
_check_stop_after "subgraph"

# ─── Stage 3a: QA Active ─────────────────────────────────────────────────────

if [[ "$RESUME_STAGE" == "3a" ]]; then
    log "=== Stage 3a: QA Active Generation ==="

    # Prepare batch file
    log "Preparing Stage 3a batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 3a \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --prompt-version "$QA_PROMPT_VERSION" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage3a_active.jsonl"

    if [[ ! -f "$BATCH_FILE" ]]; then
        error "Failed to create Stage 3a batch file"
    fi

    if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
        submit_and_process_chunked "$BATCH_FILE" "Stage 3a: ${DATASET} QA Active" \
            "$RESULTS_DIR" "$QA_DIR" "3a"
    fi

    RESUME_STAGE="3b"
fi
_check_stop_after "3a"

# ─── Stage 3b: QA Dormant ────────────────────────────────────────────────────────

if [[ "$RESUME_STAGE" == "3b" ]]; then
    log "=== Stage 3b: QA Dormant Generation ==="

    log "Preparing Stage 3b batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 3b \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --prompt-version "$QA_PROMPT_VERSION" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage3b_dormant.jsonl"

    if [[ ! -f "$BATCH_FILE" ]]; then
        error "Failed to create Stage 3b batch file"
    fi

    if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
        submit_and_process_chunked "$BATCH_FILE" "Stage 3b: ${DATASET} QA Dormant" \
            "$RESULTS_DIR" "$QA_DIR" "3b"
    fi

    RESUME_STAGE="3c"
fi
_check_stop_after "3b"

# ─── Stage 3c: Distractor Nodes QA ──────────────────────────────────────────

if [[ "$RESUME_STAGE" == "3c" ]]; then
    log "=== Stage 3c: Distractor Nodes QA Generation ==="

    log "Preparing Stage 3c batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 3c \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        --prompt-version "$QA_PROMPT_VERSION" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage3c_distractor.jsonl"

    if [[ ! -f "$BATCH_FILE" ]]; then
        error "Failed to create Stage 3c batch file"
    fi

    if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
        submit_and_process_chunked "$BATCH_FILE" "Stage 3c: ${DATASET} Distractor QA" \
            "$RESULTS_DIR" "$QA_DIR" "3c"
    fi

    RESUME_STAGE="4cf"
fi
_check_stop_after "3c"

# ─── Stage 4cf: CF Trajectories ──────────────────────────────────────────────

if [[ -z "$RESUME_STAGE" ]] || [[ "$RESUME_STAGE" =~ ^(1|2|3a|3b|3c|4cf|subgraph)$ ]]; then
    log "=== Stage 4cf: Counterfactual Trajectories + nuPlan Configs ==="
    log "Routes: infrastructure→LLM trajectory, agent→nuPlan config"

    log "Preparing Stage 4cf batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 4cf \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --qa-results "$QA_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage4cf.jsonl"
    MANIFEST_FILE="$BATCH_DIR/${DATASET}_stage4cf_manifest.json"

    if [[ -f "$BATCH_FILE" ]]; then
        if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
            submit_and_process_chunked "$BATCH_FILE" "Stage 4cf: ${DATASET} CF trajectories + nuPlan configs" \
                "$RESULTS_DIR" "$QA_DIR" "4cf" \
                --records "$RECORDS_DIR" --manifest "$MANIFEST_FILE"
        fi
    else
        log "WARNING: No Stage 4cf batch file created (no applicable questions)"
    fi
fi

_check_stop_after "4cf"

# ─── Stage 5 or 4np: Agent Intervention Ego Trajectory ───────────────────────
# --nuplan → use nuPlan simulator config (Stage 4np)
# default  → use LLM ego trajectory generation (Stage 5)

if [[ "$USE_NUPLAN" == "true" ]]; then
    log "=== Stage 4np: nuPlan Agent Intervention Config ==="

    log "Preparing Stage 4np batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 4np \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --qa-results "$QA_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage4np.jsonl"
    MANIFEST_FILE="$BATCH_DIR/${DATASET}_stage4np_manifest.json"

    if [[ -f "$BATCH_FILE" ]]; then
        if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
            submit_and_process_chunked "$BATCH_FILE" "Stage 4np: ${DATASET} nuPlan agent intervention" \
                "$RESULTS_DIR" "$QA_DIR" "4np" \
                --records "$RECORDS_DIR" --manifest "$MANIFEST_FILE"
        fi
    else
        log "WARNING: No Stage 4np batch file created (no applicable interventions)"
    fi
else
    log "=== Stage 5: LLM Ego Trajectory Agent Intervention ==="

    log "Preparing Stage 5 batch requests..."
    python "$SCRIPT_DIR/prepare_batches.py" \
        --stage 5 \
        --dataset "$DATASET" \
        --records "$RECORDS_DIR" \
        --graphs "$GRAPHS_DIR" \
        --qa-results "$QA_DIR" \
        --output "$BATCH_DIR" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        $MAX_SAMPLES $EXTRA_ARGS

    BATCH_FILE="$BATCH_DIR/${DATASET}_stage5_ego_traj.jsonl"
    MANIFEST_FILE="$BATCH_DIR/${DATASET}_stage5_ego_traj_manifest.json"

    if [[ -f "$BATCH_FILE" ]]; then
        if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
            submit_and_process_chunked "$BATCH_FILE" "Stage 5: ${DATASET} LLM ego trajectory" \
                "$RESULTS_DIR" "$QA_DIR" "5" \
                --records "$RECORDS_DIR" --manifest "$MANIFEST_FILE"
        fi
    else
        log "WARNING: No Stage 5 batch file created (no applicable agent interventions)"
    fi
fi

# ─── Stage 5 post-processing: stamp ego_trajectory_changed into QA JSONs ──────
log "=== Stage 5 post-processing: flagging unchanged trajectories ==="
python "$SCRIPT_DIR/../counterfactual_trajectory/flag_unchanged_trajectories.py" \
    --outputs-dir "$QA_DIR"

_check_stop_after "5"

# (Stage 4ac removed — merged into 4cf routing)

# ─── Dry-run cost estimate ────────────────────────────────────────────────────

dry_run_cost_estimate
if [[ "$DRY_RUN" == "true" ]]; then
    exit 0
fi

# ─── Summary ──────────────────────────────────────────────────────────────────

log "=== Pipeline Complete ==="
log ""
log "Results:"
log "  Graphs:      $GRAPHS_DIR"
log "  QA Results:  $QA_DIR"
log ""

GRAPH_COUNT=$(find "$GRAPHS_DIR" -name "*_SAMPLED_*_pruned.json" 2>/dev/null | wc -l)
QA_COUNT=$(find "$QA_DIR" -name "result.json" 2>/dev/null | wc -l)
CF_COUNT=$(find "$QA_DIR" -path "*/cf_trajectories/*/result.json" 2>/dev/null | wc -l)
AC_COUNT=$(find "$QA_DIR" -path "*/ac_trajectories/*/result.json" 2>/dev/null | wc -l)
LLM_EGO_COUNT=$(find "$QA_DIR" -name "llm_ego_trajectory.json" 2>/dev/null | wc -l)

log "Statistics:"
log "  Pruned graphs:         $GRAPH_COUNT"
log "  QA results:            $QA_COUNT"
log "  CF trajectories:       $CF_COUNT"
log "  AC trajectories:       $AC_COUNT"
log "  LLM ego trajectories:  $LLM_EGO_COUNT"
log ""
log "Batch pipeline completed successfully!"