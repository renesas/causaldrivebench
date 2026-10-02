# Alpamayo 1.0 — Run Commands

## Prerequisites

Fill in credentials:
```bash
cp evaluation/models/alpamayo_1_0/dev.env.example evaluation/models/alpamayo_1_0/dev.env
# Edit dev.env: set HF_TOKEN, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR, WEIGHTS_DIR
```

---

## 1. Unit Tests (no GPU required)

Tests live in `evaluation/models/alpamayo_1_0/tests/`.

Run from the **repo root** with a local Python env that has numpy, pillow, pytest:

```bash
# All alpamayo tests (dataloader + postprocessor)
pytest evaluation/models/alpamayo_1_0/tests/ -v

# Dataloader only (rotation math, ego history, image paths, prompt building)
pytest evaluation/models/alpamayo_1_0/tests/test_dataloader.py -v

# Postprocessor only (binary/MCQ parsing, _collect_outputs, compute_metrics)
pytest evaluation/models/alpamayo_1_0/tests/test_postprocess.py -v
```

Or **inside the container** (all deps available via uv):
```bash
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  run --rm inference \
  uv run python -m pytest \
    /workspace/repo/evaluation/models/alpamayo_1_0/tests/ -v
```

Run the **template** tests to validate the base classes:
```bash
pytest evaluation/models/_template/tests/ -v
```

---

## 2. Docker — Build Image

Run from `evaluation/models/alpamayo_1_0/` (docker-compose context is repo root):

```bash
cd evaluation/models/alpamayo_1_0
docker compose build
```

Or specify the compose file from the repo root:
```bash
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml build
```

---

## 3. Docker — Download Model Weights (~22 GB)

Requires `HF_TOKEN` in `dev.env` with access to `nvidia/Alpamayo-R1-10B`.

```bash
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  --profile setup run --rm weight-downloader
```

---

## 4. Docker — Run Inference

### Full dataset
```bash
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  run --rm inference
```

### Subset (e.g. 10 scenes)
```bash
MODE=subset BATCH_SIZE=1 \
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  run --rm -e SUBSET_SIZE=10 inference
```

### Single scene
```bash
MODE=single \
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  run --rm -e SCENE=nuscenes-scene-0159 inference
```

Outputs are written to a **timestamped run directory**:
```
$CDB_OUTPUTS_DIR/alpamayo_1_0_<YYYYMMDD_HHMMSS>/causal_nuscenes/<scene_id>/<sample_id>/
  prompts.jsonl
  outputs.jsonl
```

---

## 5. Docker — Post-process Outputs

Set `RUN_DIR` to the timestamped directory from the inference step:

```bash
RUN_DIR=/cdb_outputs/alpamayo_1_0_20260410_120000 \
docker compose -f evaluation/models/alpamayo_1_0/docker-compose.yml \
  --profile postprocess run --rm postprocess
```

Report written to `$RUN_DIR/report.json`.

---

## 6. Run Inference Directly (local dev, GPU required)

From the repo root, with Alpamayo installed and GPU available:

```bash
python evaluation/models/alpamayo_1_0/inference.py \
  --bench-dir /causal_drive_bench/causal_nuscenes \
  --output-base ./cdb_outputs \
  --weights-path /workspace/weights \
  --device cuda \
  --mode subset --subset-size 5
```

---

## 7. Run Postprocess Directly (local dev)

Provide the timestamped run directory:

```bash
python evaluation/models/alpamayo_1_0/postprocess.py \
  --bench-dir /causal_drive_bench/causal_nuscenes \
  --run-dir ./cdb_outputs/alpamayo_1_0_20260410_120000 \
  --mode full
```

---

## 8. Orchestrator (multi-model evaluation)

From the repo root, using the orchestrate.py framework:

```bash
python evaluation/orchestrate.py \
  --model alpamayo_1_0 \
  --dataset causal_nuscenes \
  --mode subset --subset-size 20
```

---

## Output Files

| Path | Description |
|------|-------------|
| `<run_dir>/causal_nuscenes/<scene>/<sample>/prompts.jsonl` | Per-question prompt payloads |
| `<run_dir>/causal_nuscenes/<scene>/<sample>/outputs.jsonl` | Per-question raw outputs + timing |
| `<run_dir>/report.json` | Accuracy metrics by QA type |
| `<run_dir>/inference.log` | Structured inference log |

### outputs.jsonl schema
```json
{
  "scene_id": "nuscenes-scene-0159",
  "sample_id": "SAMPLED_0",
  "question_id": "D1",
  "prompt_id": "000001",
  "raw_output": {"text": "<think>\nAnswer: No\nReasoning: ...\n</think>"},
  "inference_time_s": 4.21,
  "timestamp": "2026-04-10T10:00:00Z"
}
```

### report.json schema
```json
{
  "overall": {"accuracy": 0.72, "n": 450, "correct": 324},
  "per_qa_type": {
    "ladder":     {"accuracy": 0.68, "n": 200, "correct": 136},
    "dormant":    {"accuracy": 0.80, "n": 150, "correct": 120},
    "distractor": {"accuracy": 0.69, "n": 100, "correct": 69}
  },
  "confusion": {}
}
```
