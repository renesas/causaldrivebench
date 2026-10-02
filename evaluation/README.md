# CausalDriveBench — Model Evaluation Framework

Containerized, multi-model evaluation framework for the CausalDriveBench benchmark.
Each model runs in an isolated Docker container; inference and post-processing are independently resumable stages that read/write JSONL files.

---

## Quick Start

```bash
cd evaluation/models/<your_model>/

# 1. Configure environment
cp dev.env.example dev.env && nano dev.env

# 2. Build
docker compose build

# 3. Download weights
docker compose --profile setup run --rm weight-downloader

# 4. Run inference (single scene to verify)
MODE=single SCENE=nuscenes-scene-0001 docker compose run --rm inference

# 5. Post-process and view metrics
MODE=single SCENE=nuscenes-scene-0001 \
docker compose --profile postprocess run --rm postprocess

cat outputs/report.json
```

---

## Running on Sample Data (no GPU required)

The repo ships a self-contained sample dataset (`evaluation/sample_data/`) with
one representative NuScenes sample — `nuscenes-scene-0072/SAMPLED_1` — that has
all three QA types (ladder/dormant/distractor), `graph_structure.json`, and a
counterfactual trajectory question.  Use it to verify the pipeline end-to-end
without model weights or Docker.

### Step 0 — Prerequisites

```bash
# Python ≥ 3.10 with numpy, pillow, tqdm installed
# From the repo root:
source /datadrive/envs/alpamayo/bin/activate   # or any venv with the deps

export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

### Step 1 — Generate sample outputs (dataloader + postprocess)

The helper script generates `prompts.jsonl`, synthetic `outputs.jsonl`, and all
four tiers of `report.json` under `evaluation/sample_outputs/`.

```bash
python evaluation/generate_sample_outputs.py
```

Example output structure:
```
evaluation/sample_outputs/alpamayo_1_0_sample_<timestamp>/
├── report.json                             # Tier 3: run-level metrics
└── causal_nuscenes/
    ├── report.json                         # Tier 2: dataset-level metrics
    └── nuscenes-scene-0072/
        └── SAMPLED_1/
            ├── prompts.jsonl               # per-question prompt payloads
            ├── outputs.jsonl               # simulated CoC model responses
            └── report.json                 # Tier 1: sample-level metrics

evaluation/sample_reports/
├── index.html                              # Tier 4: root dashboard (open in browser)
└── alpamayo_1_0_sample_<timestamp>/
    ├── index.html                          # run HTML
    └── causal_nuscenes/
        ├── index.html                      # dataset HTML
        └── nuscenes-scene-0072/
            ├── index.html                  # scene HTML
            └── SAMPLED_1/index.html        # sample HTML
```

Open `evaluation/sample_reports/index.html` in a browser to see the full HTML
dashboard with SVG bar charts, accuracy breakdowns by question type, causal rung,
graph structure, and binary F1 metrics.

### Step 2 — Inspect a sample report

```bash
python - <<'EOF'
import json, pathlib

report = json.loads(
    (pathlib.Path("evaluation/sample_outputs")
     .glob("alpamayo_1_0_sample_*/causal_nuscenes/nuscenes-scene-0072/SAMPLED_1/report.json"))
    .__next__().read_text()
)
m = report["metrics"]
print(f"Overall: {m['overall']['accuracy']:.1%}  (n={m['overall']['n']})")
for qt, v in m.get("per_qa_type", {}).items():
    print(f"  {qt:<12}: {v['accuracy']:.1%}  (n={v['n']})")
print()
for qt, v in m.get("per_question_type", {}).items():
    if qt:
        print(f"  {qt:<6} (qtype): {v['accuracy']:.1%}  (n={v['n']})")
EOF
```

### Step 3 — Run postprocess on your own outputs (host, no Docker)

Once you have real `outputs.jsonl` from inference, point postprocess at the
run directory:

```bash
export SCRIPT_DIR=evaluation/models/alpamayo_1_0

# Single scene (fastest — re-generates Tier 1 report only for this scene)
python ${SCRIPT_DIR}/postprocess.py \
    --run-dir evaluation/sample_outputs/<your_run_dir> \
    --bench-dir evaluation/sample_data/causal_nuscenes \
    --mode single --scene nuscenes-scene-0072

# All scenes in sample_data
python ${SCRIPT_DIR}/postprocess.py \
    --run-dir evaluation/sample_outputs/<your_run_dir> \
    --bench-dir evaluation/sample_data/causal_nuscenes \
    --mode full
```

### Step 4 — Run the full alpamayo pipeline on a real machine (GPU required)

```bash
# Set paths
export MODEL_DIR=/path/to/alpamayo/src
export PYTHONPATH="${PYTHONPATH}:${MODEL_DIR}"
export SCRIPT_DIR=evaluation/models/alpamayo_1_0
export CDB_OUTPUTS=/path/to/cdb_outputs

# Inference (creates timestamped run dir under $CDB_OUTPUTS)
python ${SCRIPT_DIR}/inference.py \
    --mode single \
    --scene nuscenes-scene-0072

# Note the printed run directory, then post-process:
python ${SCRIPT_DIR}/postprocess.py \
    --run-dir ${CDB_OUTPUTS}/alpamayo_1_0_<timestamp>

# With LLM judge scoring (requires ANTHROPIC_API_KEY or FOUNDRY vars)
python ${SCRIPT_DIR}/postprocess.py \
    --run-dir ${CDB_OUTPUTS}/alpamayo_1_0_<timestamp> \
    --use-llm-judge \
    --judge-provider anthropic
```

### Step 5 — Generate HTML reports for a completed run

```bash
python evaluation/scripts/generate_html_reports.py \
    --run-name <run_name> \
    --outputs-dir /path/to/cdb_outputs \
    --reports-dir /path/to/cdb_reports \
    --bench-dir /path/to/causal_drive_bench/causal_nuscenes
```

### Step 6 — Run unit tests (no GPU, no Docker)

```bash
# All common framework tests + alpamayo model tests
pytest evaluation/tests/ evaluation/models/alpamayo_1_0/tests/ -v

# LLM judge integration tests — loads API keys from dev.env or DEV_ENV_PATH
# Create evaluation/dev.env with ANTHROPIC_API_KEY=sk-ant-... (gitignored)
pytest evaluation/tests/test_llm_judge.py -v -m integration

# Or point to an existing dev.env:
DEV_ENV_PATH=/path/to/your/dev.env \
    pytest evaluation/tests/test_llm_judge.py -v -m integration

# Sanity-check a completed run against the benchmark manifest
python evaluation/sanity_check.py \
    --outputs-dir /path/to/cdb_outputs \
    --bench-root /path/to/causal_drive_bench
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Framework design, component diagram, base class hierarchy, error handling |
| [Adding a Model](docs/adding_a_model.md) | Step-by-step guide: template → working model |
| [Docker Operations](docs/docker_operations.md) | All five services, build options, GPU config, troubleshooting |
| [Data Formats](docs/data_formats.md) | JSONL schemas, benchmark directory layout, QA types, environment variables |
| [Evaluation Pipeline](docs/evaluation_pipeline.md) | End-to-end flow, resumption, LLM judge, orchestrator usage |

---

## Repository Layout

```
evaluation/
├── README.md                  ← this file
├── eval_config.yaml           # model registry + dataset paths
├── orchestrate.py             # multi-model CLI
├── docs/                      # design documentation
├── common/                    # shared framework (dataset, base classes, metrics)
└── models/
    ├── _template/             # starting point for new models
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   ├── dataloader.py
    │   ├── inference.py
    │   ├── postprocess.py
    │   ├── requirements.in
    │   ├── system_prompt.txt
    │   ├── commands.md
    │   ├── tutorial.ipynb
    │   ├── dev.env.example
    │   ├── setup.sh
    │   └── tests/
    │       ├── test_dataloader.py
    │       └── test_postprocess.py
    └── alpamayo_1_0/          # reference implementation (NVIDIA Alpamayo R1-10B)
```

---

## Adding a New Model

See [docs/adding_a_model.md](docs/adding_a_model.md) for the full guide.

```bash
cp -r evaluation/models/_template evaluation/models/my_model
# Then: rename classes, fill Dockerfile, implement dataloader/inference/postprocess
```

---

## Data Flow

```
BenchmarkDataset → DataLoader (prompts.jsonl) → Inference (outputs.jsonl) → Postprocessor (report.json)
```

Each stage is independently executable. See [docs/evaluation_pipeline.md](docs/evaluation_pipeline.md) for details.