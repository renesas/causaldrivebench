# Commands — OmniDrive

Quick reference for development and evaluation commands.

---

## Prerequisites

Copy and fill in `dev.env` before running any service:

```bash
cp evaluation/models/omnidrive/dev.env.example evaluation/models/omnidrive/dev.env
# Edit dev.env and set: HF_TOKEN, WEIGHTS_DIR, RAW_DATA_DIR, CAUSAL_BENCH_DIR, CDB_OUTPUTS_DIR
# Also set MODEL_CONFIG and MODEL_CHECKPOINT paths.
```

All `docker compose` commands below must be run from the model directory:

```bash
cd evaluation/models/omnidrive/
```

The `--env-file` flag lets you keep `dev.env` anywhere on the host:

```bash
docker compose --env-file /path/to/dev.env <subcommand>
```

---

## 1. Unit Tests

```bash
# From repo root (no Docker required):
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/omnidrive/tests/ -v

# Dataloader tests only
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/omnidrive/tests/test_dataloader.py -v

# Postprocessor tests only
/home/shreedhar_kodate/.local/bin/pytest evaluation/models/omnidrive/tests/test_postprocess.py -v

# Inside the container:
docker compose run --rm --entrypoint "" inference \
    uv run python -m pytest /workspace/repo/evaluation/models/omnidrive/tests/ -v
```

---

## 2. Build Docker Image

```bash
# Must be run from the repo root (build context includes evaluation/)
cd ../../..
docker compose -f evaluation/models/omnidrive/docker-compose.yml build

# Force clean rebuild
docker compose -f evaluation/models/omnidrive/docker-compose.yml build --no-cache
```

---

## 3. Download Model Weights

Downloads exiawsh/OmniDrive, exiawsh/pretrain_qformer, and TinyLlama to `WEIGHTS_DIR`.

```bash
docker compose --env-file /path/to/dev.env --profile setup run --rm weight-downloader
```

> **Manual step required:** Download `eva02_petr_proj.pth` from
> https://github.com/NVlabs/OmniDrive/releases/tag/v1.0
> and place it under `WEIGHTS_DIR/`.

---

## 4. Run Original Model Inference (Sanity Check)

```bash
docker compose --env-file /path/to/dev.env --profile original run --rm original
```

---

## 5. Run Benchmark Inference

Full dataset:

```bash
docker compose --env-file /path/to/dev.env run --rm inference
```

Random subset (N scenes):

```bash
MODE=subset SUBSET_SIZE=5 docker compose --env-file /path/to/dev.env run --rm inference
```

Single scene:

```bash
MODE=single SCENE=nuscenes-scene-0001 docker compose --env-file /path/to/dev.env run --rm inference
```

The inference service creates a timestamped run directory:
```
/workspace/data/cdb_outputs/omnidrive_<YYYYMMDD_HHMMSS>/
```

### Resume an interrupted run
```bash
RUN_DIR=/workspace/data/cdb_outputs/omnidrive_<timestamp> \
docker compose --env-file /path/to/dev.env run --rm inference
```

---

## 6. Post-processing / Metrics

```bash
RUN_DIR=/workspace/data/cdb_outputs/omnidrive_20260410_120000 \
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
nvidia-smi                                  # Verify GPU
ls /workspace/weights                       # Check weights are mounted
ls /workspace/model/tools/                  # Verify OmniDrive tools

# Run benchmark inference manually
uv run python inference.py --mode single --scene nuscenes-scene-0001 \
  --output-base /workspace/data/cdb_outputs \
  --model-config /workspace/model/projects/configs/OmniDrive/eva_base_tinyllama.py

# Run original test.py directly
cd /workspace/model
python tools/test.py \
  projects/configs/OmniDrive/eva_base_tinyllama.py \
  /workspace/weights/OmniDrive/model.pth \
  --eval bbox
```

---

## 9. Docker Logs

```bash
# Follow logs of a running container
docker compose --env-file /path/to/dev.env logs -f inference

# Last 100 lines from a stopped container
docker compose --env-file /path/to/dev.env logs --tail=100 inference
```

---

## 10. Exec into a Running Container

```bash
# Find the container name
docker ps

# Exec into it
docker exec -it <container_name> bash
```

---

## Output File Reference

| File | Description |
|------|-------------|
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/` | Run directory |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/<dataset>/<scene_id>/<sample_id>/prompts.jsonl` | Per-question prompt payloads |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/<dataset>/<scene_id>/<sample_id>/outputs.jsonl` | Per-question raw outputs + timing |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/<dataset>/<scene_id>/<sample_id>/report.json` | Tier 1: per-sample metrics |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/<dataset>/report.json` | Tier 2: per-dataset metrics |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/report.json` | Tier 3: per-run metrics |
| `/workspace/data/cdb_outputs/omnidrive_<timestamp>/inference.log` | Structured inference log |
