#!/bin/bash
# Dry-run validation: checks records, prompts, prepare_batches for all 3 datasets
# Does NOT call the API — only generates batch JSONL files and validates them.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BATCH_PROCESSING="$SCRIPT_DIR/batch_processing"
DRY_DIR="/tmp/dry_run_validate_$$"
mkdir -p "$DRY_DIR"

DATASETS=("nuscenes" "openscene" "argoverse2")

ERRORS=0
WARNINGS=0

log()  { echo -e "\033[1;34m[CHECK]\033[0m $1"; }
ok()   { echo -e "\033[1;32m  ✓\033[0m $1"; }
warn() { echo -e "\033[1;33m  ⚠\033[0m $1"; WARNINGS=$((WARNINGS+1)); }
fail() { echo -e "\033[1;31m  ✗\033[0m $1"; ERRORS=$((ERRORS+1)); }

# Read config via config_loader (mirrors run_batch_pipeline.sh helpers)
_cfg_path() {
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', {}).get('$2', ''))
" 2>/dev/null
}
_cfg_batch() {
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('batch', {}).get('$1', ''))
" 2>/dev/null
}
_cfg_top() {
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from config_loader import load_config
cfg = load_config() or {}
print(cfg.get('$1', ''))
" 2>/dev/null
}

PROMPT_VERSION="$(_cfg_batch prompt_version)"
MODEL="$(_cfg_top model)"
REASONING="$(_cfg_top reasoning)"
PROVIDER="$(_cfg_top provider)"
PROMPTS="$REPO_ROOT/prompts"

log "Config: model=$MODEL  provider=$PROVIDER  prompt_version=$PROMPT_VERSION"

# ─── 1. Check records exist ───────────────────────────────────────────────────
log "Checking records directories..."
for ds in "${DATASETS[@]}"; do
    rdir="$(_cfg_path "$ds" records)"
    if [[ -z "$rdir" ]]; then
        fail "$ds: could not read records path from config.yaml"
        continue
    fi
    if [[ ! -d "$rdir" ]]; then
        fail "$ds: records dir does not exist: $rdir"
        continue
    fi
    n_scenes=$(ls -d "$rdir"/*/ 2>/dev/null | wc -l)
    n_samples=$(find "$rdir" -mindepth 2 -maxdepth 2 -type d -name "SAMPLED_*" | wc -l)
    if [[ $n_samples -eq 0 ]]; then
        fail "$ds: 0 samples found in $rdir"
    else
        ok "$ds: $n_scenes scenes, $n_samples samples ($rdir)"
    fi

    # Check record structure on first sample
    first_sample=$(find "$rdir" -mindepth 2 -maxdepth 2 -type d -name "SAMPLED_*" | head -1)
    if [[ -n "$first_sample" ]]; then
        tp0="$first_sample/Tp0p0"
        for f in bev.png cam_front.jpg cam_front_left.jpg cam_front_right.jpg state.json state_desc.json uuid_map.json; do
            if [[ ! -f "$tp0/$f" ]]; then
                fail "$ds: missing $f in $tp0"
            fi
        done
        for t in Tm1p5 Tm1p0 Tm0p5; do
            if [[ ! -d "$first_sample/$t" ]]; then
                fail "$ds: missing temporal dir $t in $first_sample"
            elif [[ ! -f "$first_sample/$t/bev.png" ]]; then
                fail "$ds: missing bev.png in $first_sample/$t"
            fi
        done
        ok "$ds: record structure looks good (checked $(basename "$first_sample"))"
    fi
done

# ─── 2. Check pruned graphs exist ────────────────────────────────────────────
log "Checking pruned graphs..."
for ds in "${DATASETS[@]}"; do
    odir="$(_cfg_path "$ds" outputs)"
    if [[ -z "$odir" ]]; then
        fail "$ds: could not read outputs path from config.yaml"
        continue
    fi
    if [[ ! -d "$odir" ]]; then
        fail "$ds: outputs dir does not exist: $odir"
        continue
    fi
    n_pruned=$(ls "$odir"/*_pruned.json 2>/dev/null | wc -l)
    n_raw=$(ls "$odir"/*.json 2>/dev/null | grep -v _pruned | wc -l)
    if [[ $n_pruned -eq 0 ]]; then
        fail "$ds: 0 pruned graphs in $odir"
    else
        ok "$ds: $n_pruned pruned graphs, $n_raw raw graphs"
    fi

    first_pruned=$(ls "$odir"/*_pruned.json 2>/dev/null | head -1)
    if [[ -n "$first_pruned" ]]; then
        n_nodes=$(python3 -c "import json; g=json.load(open('$first_pruned')); print(len(g.get('nodes',[])))")
        n_edges=$(python3 -c "import json; g=json.load(open('$first_pruned')); print(len(g.get('ego_edges',g.get('edges',[]))))")
        if [[ "$n_nodes" -eq 0 ]]; then
            warn "$ds: first pruned graph has 0 nodes"
        else
            ok "$ds: sample graph has $n_nodes nodes, $n_edges edges"
        fi
    fi
done

# ─── 3. Check prompt files exist ─────────────────────────────────────────────
log "Checking prompt files (version: $PROMPT_VERSION)..."

# Stage 1 — causal scene graph
for f in system_prompt.md; do
    fp="$PROMPTS/causal_scene_graph/$PROMPT_VERSION/$f"
    [[ -f "$fp" ]] && ok "causal_scene_graph/$PROMPT_VERSION/$f" || fail "MISSING: $fp"
done

# Stage 3a/3b/3c — QA prompts
for f in \
    Causal_Ladder_System_Prompt.md \
    Causal_Ladder_User_Prompt.md \
    Dormant_Nodes_System_Prmpt.md \
    Dormant_Nodes_User_Prompt.md \
    Distractor_Nodes_System_Prompt.md \
    Distractor_Nodes_User_Prompt.md; do
    fp="$PROMPTS/causal_qas/$PROMPT_VERSION/$f"
    [[ -f "$fp" ]] && ok "causal_qas/$PROMPT_VERSION/$f" || fail "MISSING: $fp"
done

# Stage 4cf — counterfactual / intervention
for f in \
    Intervention_System_Prompt.md \
    Intervention_User_Prompt.md \
    CounterFactual_System_Prompt.md \
    CounterFactual_User_Prompt.md; do
    fp="$PROMPTS/alternate_trajectories/$f"
    [[ -f "$fp" ]] && ok "alternate_trajectories/$f" || fail "MISSING: $fp"
done

# nuPlan config generation
for f in system_prompt.md user_prompt.md; do
    fp="$PROMPTS/nuplan_config/$f"
    [[ -f "$fp" ]] && ok "nuplan_config/$f" || fail "MISSING: $fp"
done

# ─── 4. Dry-run prepare_batches for each stage × dataset ─────────────────────
log "Dry-run: preparing batch JSONL files (no API calls)..."
cd "$REPO_ROOT"

for ds in "${DATASETS[@]}"; do
    rdir="$(_cfg_path "$ds" records)"
    odir="$(_cfg_path "$ds" outputs)"
    dry="$DRY_DIR/$ds"
    mkdir -p "$dry"

    if [[ ! -d "$rdir" ]] || [[ ! -d "$odir" ]]; then
        fail "$ds: skipping dry-run (missing records or outputs)"
        continue
    fi

    _run_stage() {
        local stage=$1
        local label=$2
        local expected_file=$3

        log "  $ds Stage $stage ($label)..."
        if python "$BATCH_PROCESSING/prepare_batches.py" \
            --stage "$stage" --dataset "$ds" \
            --records "$rdir" --graphs "$odir" \
            --output "$dry" \
            --model "$MODEL" --reasoning "$REASONING" \
            --provider "$PROVIDER" \
            --prompt-version "$PROMPT_VERSION" \
            2>&1 | tail -3; then
            batch_file="$dry/$expected_file"
            if [[ -f "$batch_file" ]]; then
                n_req=$(wc -l < "$batch_file")
                ok "$ds $stage: $n_req requests generated"
            else
                fail "$ds $stage: batch file not created ($expected_file)"
            fi
        else
            fail "$ds $stage: prepare_batches.py failed"
        fi
    }

    _run_stage 1   "Causal Graph"  "${ds}_stage1_generator.jsonl"

    # Validate image count in stage 3a (most image-heavy stage)
    log "  $ds Stage 3a (Active QA)..."
    if python "$BATCH_PROCESSING/prepare_batches.py" \
        --stage 3a --dataset "$ds" \
        --records "$rdir" --graphs "$odir" \
        --output "$dry" \
        --model "$MODEL" --reasoning "$REASONING" \
        --provider "$PROVIDER" \
        --prompt-version "$PROMPT_VERSION" \
        2>&1 | tail -3; then
        batch_file="$dry/${ds}_stage3a_active.jsonl"
        if [[ -f "$batch_file" ]]; then
            n_req=$(wc -l < "$batch_file")
            n_images=$(head -1 "$batch_file" | python3 -c "
import sys, json
req = json.load(sys.stdin)
body = req.get('body', req.get('params', {}))
msgs = body.get('messages', [])
user_msg = [m for m in msgs if m['role']=='user'][0]
n_img = sum(1 for b in user_msg['content'] if b.get('type')=='image')
print(n_img)
")
            if [[ "$n_images" -eq 20 ]]; then
                ok "$ds 3a: $n_req requests, $n_images images per request (5-cam)"
            elif [[ "$n_images" -eq 24 ]]; then
                ok "$ds 3a: $n_req requests, $n_images images per request (6-cam AV2)"
            elif [[ "$n_images" -gt 0 ]]; then
                warn "$ds 3a: $n_req requests, $n_images images per request (unexpected count)"
            else
                fail "$ds 3a: 0 images in first request"
            fi
        else
            fail "$ds 3a: batch file not created"
        fi
    else
        fail "$ds 3a: prepare_batches.py failed"
    fi

    _run_stage 3b  "Dormant QA"    "${ds}_stage3b_dl.jsonl"
    _run_stage 3c  "Distractor QA" "${ds}_stage3c_distractor.jsonl"
done

# ─── 5. Estimate cost ────────────────────────────────────────────────────────
log "Estimating cost from dry-run JSONL files..."
python3 - "$DRY_DIR" <<'PYEOF'
import sys, json, os
from pathlib import Path

dry_dir = Path(sys.argv[1])
total_input_tok = 0
total_output_tok = 0
total_requests = 0

# Claude batch pricing (claude-opus-4-6)
INPUT_PRICE  = 2.50 / 1_000_000   # $2.50 per MTok
OUTPUT_PRICE = 12.50 / 1_000_000  # $12.50 per MTok
AVG_OUTPUT_TOK = 2600              # measured from 5-sample test batch

for jsonl in sorted(dry_dir.rglob("*.jsonl")):
    n_req = 0
    file_input = 0
    for line in open(jsonl):
        req = json.loads(line)
        body = req.get("body", req.get("params", {}))
        msgs = body.get("messages", [])
        n_images = 0
        text_chars = 0
        for msg in msgs:
            for block in msg.get("content", []):
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        n_images += 1
                    elif block.get("type") == "text":
                        text_chars += len(block.get("text", ""))
                elif isinstance(block, str):
                    text_chars += len(block)
        sys_prompt = body.get("system", "")
        if isinstance(sys_prompt, list):
            for s in sys_prompt:
                text_chars += len(s.get("text", ""))
        else:
            text_chars += len(sys_prompt)
        est_tok = text_chars // 4 + n_images * 1000
        file_input += est_tok
        n_req += 1

    est_output = n_req * AVG_OUTPUT_TOK
    cost = file_input * INPUT_PRICE + est_output * OUTPUT_PRICE
    total_input_tok += file_input
    total_output_tok += est_output
    total_requests += n_req
    print(f"  {jsonl.name}: {n_req} requests, ~{file_input/1000:.0f}K input tok, est cost ${cost:.2f}")

total_cost = total_input_tok * INPUT_PRICE + total_output_tok * OUTPUT_PRICE
print(f"\n  TOTAL: {total_requests} requests")
print(f"  Est input:  ~{total_input_tok/1_000_000:.2f}M tokens")
print(f"  Est output: ~{total_output_tok/1_000_000:.2f}M tokens")
print(f"  Est cost:   ${total_cost:.2f}")
PYEOF

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
if [[ $ERRORS -gt 0 ]]; then
    echo -e "\033[1;31m  FAILED: $ERRORS errors, $WARNINGS warnings\033[0m"
    echo "  Fix the errors above before running the pipeline!"
    exit 1
elif [[ $WARNINGS -gt 0 ]]; then
    echo -e "\033[1;33m  PASSED with $WARNINGS warnings\033[0m"
else
    echo -e "\033[1;32m  ALL CHECKS PASSED\033[0m"
fi
echo "  Dry-run files: $DRY_DIR"
echo "═══════════════════════════════════════════════"

# Cleanup
# rm -rf "$DRY_DIR"  # Uncomment to auto-cleanup
