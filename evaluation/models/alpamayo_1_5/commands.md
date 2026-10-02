# Alpamayo 1.5 — Run Commands

## Prerequisites

Fill in credentials and paths:
```bash
cp evaluation/models/alpamayo_1_5/dev.env.example /path/to/dev.env
# Edit /path/to/dev.env: set HF_TOKEN, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR, WEIGHTS_DIR
```

The `dev.env` file can live anywhere on the host. Pass its path to every
`docker compose` command via `--env-file`.

---

## 1. Unit Tests (no GPU required)

Run from the **repo root** with a local Python env that has numpy, pillow, pytest:

```bash
# All alpamayo_1_5 tests
pytest evaluation/models/alpamayo_1_5/tests/ -v

# Template / base class tests
pytest evaluation/models/_template/tests/ -v
```

Or **inside the container** after building:
```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  run --rm inference \
  uv run python -m pytest /workspace/repo/evaluation/models/alpamayo_1_5/tests/ -v
```

---

## 2. Docker — Build Image

Run from `evaluation/models/alpamayo_1_5/` (compose context is repo root).

```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  build
```

**Note on `physical-ai-av`**: The Alpamayo 1.5 `pyproject.toml` lists
`physical-ai-av==0.2.0`. If this package is unavailable on PyPI the build will
fail at `uv sync --active`. Contact NVIDIA for access or comment it out (our
inference scripts do not import it).

---

## 3. Docker — Download Model Weights (~22 GB)

Requires `HF_TOKEN` in `dev.env` with access to `nvidia/Alpamayo-1.5-10B`.

```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  --profile setup run --rm weight-downloader
```

Weights are saved to `WEIGHTS_DIR` on the host and re-used across containers.

---

## 4. Docker — Run Inference

### Full dataset (default)
```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  run --rm inference
```

### Subset (e.g. 10 scenes)
```bash
MODE=subset \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  run --rm -e SUBSET_SIZE=10 inference
```

### Single scene
```bash
MODE=single \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  run --rm -e SCENE=nuscenes-scene-0159 inference
```

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/alpamayo_1_5_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  run --rm inference
```

Outputs are written to a **timestamped run directory**:
```
$CDB_OUTPUTS_DIR/alpamayo_1_5_<YYYYMMDD_HHMMSS>/causal_nuscenes/<scene_id>/<sample_id>/
  prompts.jsonl
  outputs.jsonl
```

---

## 5. Docker — Run Original Test Script (sanity check)

Runs Alpamayo 1.5's bundled `test_inference.py` on the PhysicalAI-AV dataset.
Requires access to NVIDIA's proprietary `physical-ai-av` dataset package.

```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  --profile original run --rm original
```

---

## 6. Docker — Post-process Outputs

Set `RUN_DIR` to the timestamped directory from the inference step.

```bash
RUN_DIR=/workspace/data/cdb_outputs/alpamayo_1_5_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  --profile postprocess run --rm postprocess
```

With LLM judge enabled:
```bash
RUN_DIR=/workspace/data/cdb_outputs/alpamayo_1_5_20260421_120000 \
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  --profile postprocess \
  run --rm -e USE_LLM_JUDGE=true postprocess
```

---

## 7. Docker — Jupyter Notebook Server

```bash
docker compose \
  --env-file /path/to/dev.env \
  -f evaluation/models/alpamayo_1_5/docker-compose.yml \
  --profile jupyter up jupyter
```

Open http://localhost:8888 — token is printed in the console on startup.

---

## 8. Run Inference Directly (local dev, GPU required)

```bash
cd /path/to/neurips-dataset
export PYTHONPATH="$PYTHONPATH:$(pwd)"

python evaluation/models/alpamayo_1_5/inference.py \
  --bench-dir /datadrive/datasets/causal_drive_bench/causal_nuscenes \
  --output-base /datadrive/datasets/cdb_outputs \
  --weights-path /datadrive/models/Alpamayo-1.5-10B \
  --device cuda \
  --mode subset --subset-size 5
```

---

## 9. Run Postprocess Directly (local dev)

```bash
python evaluation/models/alpamayo_1_5/postprocess.py \
  --bench-dir /datadrive/datasets/causal_drive_bench/causal_nuscenes \
  --run-dir /datadrive/datasets/cdb_outputs/alpamayo_1_5_20260421_120000 \
  --mode full

# Reprocess all runs under cdb_outputs_dir (auto-detects datasets):
python evaluation/models/alpamayo_1_5/postprocess.py --all-runs
```

---

## 10. Orchestrator (multi-model evaluation)

```bash
python evaluation/orchestrate.py \
  --model alpamayo_1_5 \
  --dataset causal_nuscenes \
  --mode subset --subset-size 20
```

---

## Useful Docker Operations

### View live logs of a running container
```bash
docker logs -f <container_id_or_name>
```

### Exec into a running container
```bash
docker exec -it <container_id_or_name> /bin/bash
```

### Inspect a stopped container's logs
```bash
docker ps -a                                          # find container ID
docker logs <container_id>
```

### Check GPU utilisation inside container
```bash
docker exec -it <container_id_or_name> nvidia-smi
```

### Copy outputs from container to host
```bash
docker cp <container_id>:/workspace/data/cdb_outputs/. /local/cdb_outputs/
```

### Remove stopped containers and dangling images
```bash
docker container prune -f
docker image prune -f
```

---

## Output Files

| Path | Description |
|------|-------------|
| `<run_dir>/causal_nuscenes/<scene>/<sample>/prompts.jsonl` | Per-question prompt payloads |
| `<run_dir>/causal_nuscenes/<scene>/<sample>/outputs.jsonl` | Per-question raw outputs + timing |
| `<run_dir>/causal_nuscenes/<scene>/<sample>/report.json` | Tier 1: per-sample metrics |
| `<run_dir>/causal_nuscenes/report.json` | Tier 2: per-dataset metrics |
| `<run_dir>/report.json` | Tier 3: per-run metrics |
| `<run_dir>/inference.log` | Structured inference log |

### outputs.jsonl schema
```json
{
  "scene_id": "nuscenes-scene-0159",
  "sample_id": "SAMPLED_0",
  "question_id": "D1",
  "prompt_id": "000001",
  "raw_output": {"text": "Answer: No\nReasoning: The vehicle is not in the lane."},
  "inference_time_s": 2.14,
  "timestamp": "2026-04-21T10:00:00Z"
}
```

### report.json schema (Tier 3)
```json
{
  "overall": {"accuracy": 0.74, "n": 450, "correct": 333},
  "per_qa_type": {
    "ladder":     {"accuracy": 0.70, "n": 200, "correct": 140},
    "dormant":    {"accuracy": 0.82, "n": 150, "correct": 123},
    "distractor": {"accuracy": 0.70, "n": 100, "correct": 70}
  }
}
```
