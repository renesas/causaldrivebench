#!/bin/bash
# End-to-end pipeline: Causal Graph → QA + CF → HTML Rendering
#
# Usage:
#   bash scripts/run_pipeline.sh <dataset> [--skip-graphs] [--skip-qa] [--skip-html]
#
# Dataset: nuscenes | openscene | argoverse2
# Paths are read from config.yaml automatically.
#
# Example:
#   bash scripts/run_pipeline.sh nuscenes
#   bash scripts/run_pipeline.sh openscene --skip-graphs
#   bash scripts/run_pipeline.sh argoverse2 --skip-qa --skip-html
#
# Run all 3 in parallel:
#   bash scripts/run_pipeline.sh nuscenes &
#   bash scripts/run_pipeline.sh openscene &
#   bash scripts/run_pipeline.sh argoverse2 &
#   wait

set -e

DATASET="${1:?Usage: bash scripts/run_pipeline.sh <dataset> [--skip-graphs] [--skip-qa] [--skip-html]}"
shift

# Parse flags
SKIP_GRAPHS=false
SKIP_QA=false
SKIP_HTML=false
SAMPLES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-graphs) SKIP_GRAPHS=true; shift ;;
    --skip-qa)     SKIP_QA=true; shift ;;
    --skip-html)   SKIP_HTML=true; shift ;;
    --samples)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SAMPLES+=("$1"); shift
      done
      ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$REPO_DIR/config.yaml"
RENDER="$SCRIPT_DIR/rendering/render_causal_graph.py"

# Read paths from config.yaml
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config.yaml not found at $CONFIG"
  exit 1
fi

RECORDS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['$DATASET']['records'])")
OUTDIR=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['$DATASET']['outputs'])")
NUM_SCENES=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['$DATASET'].get('num_scenes', 0))")
PROVIDER=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('provider', 'openai'))")
MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('model', 'gpt-5.2'))")
REASONING=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('reasoning', 'none'))")

echo "=== Pipeline: $DATASET ==="
echo "Provider: $PROVIDER  |  Model: $MODEL  |  Reasoning: $REASONING"
echo "Records: $RECORDS"
echo "Output:  $OUTDIR"
echo "Scenes:  $NUM_SCENES"

eval "$(conda shell.bash hook)"

# ─────────────────────────────────────────────
# Step 1: Causal Graph Generation
# ─────────────────────────────────────────────
if [ "$SKIP_GRAPHS" = false ]; then
  conda activate causal_graph
  echo ""
  echo ">>> Step 1: Causal Graph Generation"
  SCENE_ARGS=""
  if [ ${#SAMPLES[@]} -gt 0 ]; then
    SCENE_ARGS="--samples ${SAMPLES[*]}"
  elif [ "$NUM_SCENES" -gt 0 ]; then
    SCENE_ARGS="--scenes $(ls -d "$RECORDS"/*/ | head -n "$NUM_SCENES" | xargs -I{} basename {} | tr '\n' ' ')"
  fi
  python "$SCRIPT_DIR/causal_graph_extraction/v3_pipeline.py" \
    --records "$RECORDS" --out "$OUTDIR" --force \
    --provider "$PROVIDER" --model "$MODEL" --reasoning "$REASONING" $SCENE_ARGS
else
  echo ">>> Skipping Step 1: Causal Graph Generation"
fi

# ─────────────────────────────────────────────
# Step 2: QA + CF Trajectories (per scene)
# ─────────────────────────────────────────────
if [ "$SKIP_QA" = false ]; then
  conda activate causal_graph
  echo ""
  echo ">>> Step 2: QA + CF Trajectories"
  for refined in "$OUTDIR"/*_SAMPLED_*_refined.json; do
    [ -f "$refined" ] || continue
    stem=$(basename "$refined" _refined.json)
    # If --samples given, skip stems not in the list
    if [ ${#SAMPLES[@]} -gt 0 ]; then
      match=false
      for s in "${SAMPLES[@]}"; do [[ "$stem" == "$s" ]] && match=true && break; done
      [[ "$match" == false ]] && continue
    fi
    echo "  >> $stem"
    python "$SCRIPT_DIR/question_answers/generate_causal_qa.py" \
      --records "$RECORDS" --graphs "$OUTDIR" \
      --scene "$stem" --out "$OUTDIR/qa_results" --force \
      --provider "$PROVIDER" --model "$MODEL" --reasoning "$REASONING"
  done
  # Shuffle MCQ options so correct answer is uniformly distributed
  echo "  >> Shuffling MCQ options..."
  python "$SCRIPT_DIR/question_answers/shuffle_mcq_options.py" \
    --qa-dir "$OUTDIR/qa_results" --seed 42
else
  echo ">>> Skipping Step 2: QA + CF"
fi

# ─────────────────────────────────────────────
# Step 3: HTML Rendering
# ─────────────────────────────────────────────
if [ "$SKIP_HTML" = false ]; then
  if [ ! -f "$RENDER" ]; then
    echo "WARNING: render_causal_graph.py not found at $RENDER"
    echo "Skipping HTML rendering."
  else
    conda activate causal_graph
    rm -rf "$OUTDIR/assets"
    echo ""
    echo ">>> Step 3: HTML Rendering"
    count=0
    for refined in "$OUTDIR"/*_SAMPLED_*_refined.json; do
      [ -f "$refined" ] || continue
      stem=$(basename "$refined" _refined.json)
      # If --samples given, skip stems not in the list
      if [ ${#SAMPLES[@]} -gt 0 ]; then
        match=false
        for s in "${SAMPLES[@]}"; do [[ "$stem" == "$s" ]] && match=true && break; done
        [[ "$match" == false ]] && continue
      fi
      qa_dir="$OUTDIR/qa_results/$stem"
      echo "  render $(basename "$refined" .json)"
      if [ -d "$qa_dir" ]; then
        python "$RENDER" "$refined" --html --records "$RECORDS" --model-label "$MODEL" --qa-dir "$qa_dir"
      else
        python "$RENDER" "$refined" --html --records "$RECORDS" --model-label "$MODEL"
      fi
    done
  fi
else
  echo ">>> Skipping Step 3: HTML Rendering"
fi

# ─────────────────────────────────────────────
# Step 4: Package Release (record.json + causal_qa.json)
# ─────────────────────────────────────────────
echo ""
echo ">>> Step 4: Package Release"
conda activate causal_graph
RELEASE_DIR="$(dirname "$OUTDIR")/release/prepared"
PACKAGE_DIR="$(dirname "$OUTDIR")/release/packages"
DOWNLOAD_DIR="$(dirname "$OUTDIR")/outputs/downloads"
mkdir -p "$RELEASE_DIR" "$PACKAGE_DIR" "$DOWNLOAD_DIR"
python "$SCRIPT_DIR/release/prepare_dataset_release.py" \
  --records "$RECORDS" \
  --qa-results "$OUTDIR/qa_results" \
  --output "$RELEASE_DIR" \
  --dataset "$DATASET"
if [ -d "$RELEASE_DIR/$DATASET" ]; then
  cd "$RELEASE_DIR"
  zip -r -q "$PACKAGE_DIR/${DATASET}_release.zip" "$DATASET/"
  cp "$PACKAGE_DIR/${DATASET}_release.zip" "$DOWNLOAD_DIR/"
  cd - > /dev/null
  echo "  Package: $PACKAGE_DIR/${DATASET}_release.zip"
fi

echo ""
echo "=== Pipeline complete: $DATASET ==="
echo "Scenes: $(ls -d "$RECORDS"/*/ 2>/dev/null | wc -l)"
