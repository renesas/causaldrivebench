# Commands — OpenDriveVLA

Quick reference for development and evaluation commands.

> **Build note:** MMCV and MMDet3D are compiled from source during the Docker build
> (requires CUDA). First build takes ~20-30 minutes.

---

## Prerequisites

```bash
cp evaluation/models/opendrivevla/dev.env.example evaluation/models/opendrivevla/dev.env
# Edit dev.env and set:
#   HF_TOKEN, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, OUTPUT_DIR
#   MODEL_CHECKPOINT, PRECISION (bf16 / fp16 / fp32)
```

All `docker compose` commands must be run from the model directory:

```bash
cd evaluation/models/opendrivevla/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

```bash
# From repo root (no Docker needed)
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/opendrivevla/tests/ -v

# Inside container
docker compose run --rm --entrypoint "" inference \
    uv run python -m pytest /workspace/repo/evaluation/models/opendrivevla/tests/ -v
```

---

## 2. Build Docker Image

```bash
# From repo root
cd ../../..
docker compose -f evaluation/models/opendrivevla/docker-compose.yml build

# Clean rebuild
docker compose -f evaluation/models/opendrivevla/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

Downloads `OpenDriveVLA/OpenDriveVLA-0.5B` from HuggingFace:

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

---

## 4. Run Original Model Inference (Sanity Check)

Runs `drivevla/inference_drivevla.py` with `torchrun` on 1 GPU:

```bash
docker compose --env-file /path/to/dev.env --profile original run --rm original
```

---

## 5. Run Benchmark Inference

Full dataset:

```bash
docker compose --env-file /path/to/dev.env run --rm inference
```

Random subset:

```bash
MODE=subset docker compose --env-file /path/to/dev.env run --rm inference
```

Single scene:

```bash
MODE=single docker compose --env-file /path/to/dev.env run --rm inference
```

Outputs are written to: `$CDB_OUTPUTS_DIR/opendrivevla_<timestamp>/<dataset>/<scene_id>/<sample_id>/`

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/opendrivevla_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/opendrivevla_20260410_120000 \
docker compose --env-file /path/to/dev.env --profile postprocess run --rm postprocess
```

Reports are written to:
- `<run_dir>/<dataset>/<scene>/<sample>/report.json` — Tier 1: per-sample
- `<run_dir>/<dataset>/report.json` — Tier 2: per-dataset
- `<run_dir>/report.json` — Tier 3: per-run

---

## 7. Interactive Development (Jupyter)

```bash
docker compose --env-file /path/to/dev.env --profile jupyter up jupyter
# Open: http://localhost:8888
```

---

## 8. Interactive Shell (Debugging)

```bash
docker compose --env-file /path/to/dev.env run --rm --entrypoint bash inference
```

Inside the shell:

```bash
nvidia-smi
ls /workspace/weights/OpenDriveVLA-0.5B/    # Verify checkpoint

# Run original inference directly (1 GPU)
cd /workspace/model
torchrun --nproc_per_node=1 drivevla/inference_drivevla.py \
  --model-path /workspace/weights/OpenDriveVLA-0.5B \
  --output /outputs/drivevla_debug.json \
  --num-workers 2 \
  --batch-size 1 \
  --bf16
```

---

## 9. Docker Logs

```bash
docker compose --env-file /path/to/dev.env logs -f inference
docker compose --env-file /path/to/dev.env logs --tail=100 inference
```

---

## 10. Exec into a Running Container

```bash
docker ps
docker exec -it <container_name> bash
```

---

## Output File Reference

| File | Description |
|------|-------------|
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `<run_dir>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `<run_dir>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `<run_dir>/report.json` | Tier 3: per-run metrics |
| `<run_dir>/inference.log` | Structured inference log |

Where `<run_dir>` = `$CDB_OUTPUTS_DIR/opendrivevla_<timestamp>/`
